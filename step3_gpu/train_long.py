#!/usr/bin/env python3
"""LONGER-training embodied comparison (committee fix #1/#3/#4), non-destructive.

Extends the F1 sensing-limited comparison (train_gpu / torch_env, both UNCHANGED) to 150k env-steps
(10x the original 15k) and additionally:
  * records the embodied LEARNING CURVES (oracle-fraction vs training step) for every learner, so the
    plateau can be shown (answers the unanimous #1 gate);
  * measures the embodied EXPLOITABILITY / Nash-gap for EVERY arm (swarm, random, MLP-IPPO, GNN) on the
    same footing, so the embodied optimum no longer rests only on the IFD-slope proxy (answers #4).

All independent units (16 seeds x 2 operating points x learner family) still run as ONE batched GPU kernel
via train_gpu.train (grouped per-policy weights). Output -> step3_gpu_long.npz (the 15k step3_gpu_full.npz
is left untouched). Run:  python train_long.py --ticks 150000
"""
import os, sys, time, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch_env as TE
from torch_env import TorchForageEnv, N, F_OBS, C
import train_gpu as TG
from train_gpu import train, eval_policy, DEV, _with_oom_retry

_rd = os.path.normpath(os.path.join(HERE, "..", "data", "step3_gpu"))
DATA = os.environ.get("FLOCKING_DATA") or (_rd if os.path.isdir(_rd) else "/mnt/newssd/flocking_ltm_data/step3_gpu")
os.makedirs(DATA, exist_ok=True)

# Relay-decay gamma is REMOVED from Paper 2 (not a Paper-2 claim). gamma=1.0 == no per-hop attenuation
# (strength = gamma^hop = 1 at every hop), i.e. a plain linear-threshold cascade. The original FAVORABLE
# point ran at gamma=0.8 (a silent per-regime swarm-mechanism difference); we pin BOTH points to gamma=1.0
# so nothing is tuned per regime. Effect on the result is negligible (harvest 0.763 vs 0.765). Only the swarm
# arm reads gamma (via cascade()); the learners ignore it, so this does not affect the learner training.
PAPER = dict(TG.PAPER); PAPER["gamma"] = 1.0
FAV = dict(TG.FAV);   FAV["gamma"] = 1.0


def exploit_per_world(rich, occ):
    """Embodied equal-intake gap (a labelled Nash-gap surrogate) of the *realised allocation*, per world.

    The ideal-free / Nash equilibrium of the congestion game is the allocation at which per-capita intake is
    EQUAL across the patches being worked (App. `optimum`); an agent then has nothing to gain by moving. We
    therefore measure how far realised per-capita intake is from equal across worked patches: the
    occupancy-weighted coefficient of variation of v_i = Q_i/n_i over occupied patches (n_i >= 0.5). It is 0
    exactly at the equal-intake equilibrium and grows as the allocation departs from it; being a CV it is
    dimensionless and comparable across fields. This is the direct dispersion version of the IFD-slope claim
    (equal per-capita intake rather than slope-one occupancy), so the embodied optimum no longer rests only on
    the slope proxy.

    Deviations to *unoccupied* patches are excluded: reaching an undiscovered patch is a search / harvesting-
    efficiency matter (the companion paper's subject), not a property of the allocation equilibrium this paper
    claims the swarm reaches; including them would penalise a swarm merely for not yet having found every patch,
    conflating discovery with allocation."""
    rich = np.asarray(rich, float); occ = np.asarray(occ, float)
    occupied = occ >= 0.5
    v = rich / np.clip(occ, 1e-9, None)                               # (nw,C) per-capita intake
    w = occ * occupied                                               # weight by committed agents
    wsum = w.sum(1)
    mu = (w * v).sum(1) / np.clip(wsum, 1e-9, None)                   # weighted mean per-capita intake
    var = (w * (v - mu[:, None]) ** 2).sum(1) / np.clip(wsum, 1e-9, None)
    cv = np.sqrt(np.clip(var, 0.0, None)) / np.clip(mu, 1e-9, None)   # weighted CV = equal-intake gap
    return np.where(wsum > 0, cv, 0.0)                                # (nw,)


def per_seed(x_world, nseed, R):
    """(nseed*R,) world values (one point) -> (nseed,) by averaging the R replicas within each seed."""
    return np.asarray(x_world, float).reshape(nseed, R).mean(1)


def eval_sr(controller, W, ticks, window, op, seed):
    """Like torch_env._eval but also returns per-world (rich, mean-occupancy) for exploitability."""
    env = TorchForageEnv(W, op["phi"], op["patchiness"], op["theta"], op["gamma"], seed=seed)
    ie = torch.zeros(W, N, device=env.device); oe = torch.zeros(W, N, device=env.device)
    gen = torch.Generator(device=env.device); gen.manual_seed(seed + 777)
    hr = torch.zeros(W, device=env.device); hc = torch.zeros(W, device=env.device)
    occ = torch.zeros(W, C, device=env.device); nmeas = 0
    for t in range(ticks):
        capb = env.cap.sum(1)
        if controller == "swarm":
            on, a, _ = env.swarm_actions()
        else:
            on, prox, feats = env.extract(ie, oe)
            a = torch.randint(0, 3, (W, N), generator=gen, device=env.device)
        intake = env.step(a, on)
        ie = 0.9 * ie + 0.1 * intake; oe = 0.95 * oe + 0.05 * on.to(env.dtype)
        if t >= ticks - window:
            hr += intake.sum(1); hc += torch.minimum(torch.tensor(float(N), device=env.device), capb)
            occ += env.occupancy(); nmeas += 1
    frac = (hr / hc.clamp(min=1e-9)).cpu().numpy()
    occm = (occ / max(nmeas, 1)).cpu().numpy(); rich = env.rich.cpu().numpy()
    slope = TE.slope_pooled(rich, occm)
    return frac, slope, rich, occm


def main(ticks, nseed, R_mlp, R_gnn, R_eval, eval_ticks, base, out):
    POINTS = [("PAPER", PAPER), ("FAVORABLE", FAV)]
    ops = [op for _, op in POINTS for _ in range(nseed)]       # M = 2*nseed, PAPER block then FAV block
    res = {name: {} for name, _ in POINTS}
    pslice = {"PAPER": slice(0, nseed), "FAVORABLE": slice(nseed, 2 * nseed)}
    wslice = lambda R: {"PAPER": slice(0, nseed * R), "FAVORABLE": slice(nseed * R, 2 * nseed * R)}
    t0 = time.time()

    def run_cfg(tag, L, R0):
        (tr, R) = _with_oom_retry(lambda R: train(ops, R, ticks=ticks, seed=base, L=L, sample_gpu=True), R0)
        ev, rich, occ = eval_policy(tr["pol"], ops, R_eval, eval_ticks, seed=base + 10000, L=L)
        print(f"[{tag}] R={R} ticks={ticks} eval frac: PAPER {ev[pslice['PAPER']].mean():.3f} "
              f"FAV {ev[pslice['FAVORABLE']].mean():.3f}  util={tr['gpu_util']}% vram={tr['gpu_mem']}MiB "
              f"({time.time()-t0:.0f}s)", flush=True)
        ex60 = exploit_per_world(rich, occ)
        return tr, ev, rich, occ, ex60

    # ---- MLP-IPPO ----
    trm, evm, richm, occm, exm = run_cfg("MLP-IPPO", None, R_mlp)
    for name in res:
        res[name]["mlp_ippo_finals"] = evm[pslice[name]]
        res[name]["mlp_ippo_curve_f"] = trm["curve_f"][:, pslice[name]]; res[name]["mlp_ippo_ct"] = trm["curve_t"]
        ws = wslice(R_eval)[name]
        res[name]["mlp_ippo_slope"] = TE.slope_pooled(richm[ws], occm[ws])
        res[name]["mlp_ippo_exploit"] = per_seed(exm[ws], nseed, R_eval)
        res[name]["mlp_ippo_rich"] = richm[ws]; res[name]["mlp_ippo_occ"] = occm[ws]
    torch.cuda.empty_cache()
    # ---- GNN-IPPO per L ----
    for L in (1, 3, 5):
        trg, evg, richg, occg, exg = run_cfg(f"GNN-L{L}", L, R_gnn)
        for name in res:
            res[name][f"gnn_L{L}_finals"] = evg[pslice[name]]
            res[name][f"gnn_L{L}_curve_f"] = trg["curve_f"][:, pslice[name]]; res[name][f"gnn_L{L}_ct"] = trg["curve_t"]
            ws = wslice(R_eval)[name]
            res[name][f"gnn_L{L}_slope"] = TE.slope_pooled(richg[ws], occg[ws])
            res[name][f"gnn_L{L}_exploit"] = per_seed(exg[ws], nseed, R_eval)
            res[name][f"gnn_L{L}_rich"] = richg[ws]; res[name][f"gnn_L{L}_occ"] = occg[ws]
        torch.cuda.empty_cache()
    # ---- reward-free swarm + random baselines (per world for CIs + exploitability) ----
    for name, op in POINTS:
        sf, ss, srich, socc = eval_sr("swarm", 256, eval_ticks, 2000, op, base + 555)
        rf, _, rrich, rocc = eval_sr("random", 256, eval_ticks, 2000, op, base + 555)
        res[name]["swarm"] = sf; res[name]["swarm_slope"] = ss; res[name]["random"] = rf
        res[name]["swarm_exploit"] = exploit_per_world(srich, socc)
        res[name]["random_exploit"] = exploit_per_world(rrich, rocc)
        res[name]["swarm_rich"] = srich; res[name]["swarm_occ"] = socc
        res[name]["random_rich"] = rrich; res[name]["random_occ"] = rocc
        print(f"[baseline {name}] swarm {sf.mean():.3f} (slope {ss:.2f}, exploit "
              f"{res[name]['swarm_exploit'].mean():.3f})  random {rf.mean():.3f} "
              f"(exploit {res[name]['random_exploit'].mean():.3f})  ({time.time()-t0:.0f}s)", flush=True)

    res["_meta"] = dict(ticks=ticks, nseed=nseed, R_mlp=R_mlp, R_gnn=R_gnn, R_eval=R_eval,
                        eval_ticks=eval_ticks)
    outp = os.path.join(DATA, out)
    np.savez_compressed(outp, res=res)
    print(f"wrote {outp}  ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=150000)
    ap.add_argument("--nseed", type=int, default=16)
    ap.add_argument("--R_mlp", type=int, default=24)
    ap.add_argument("--R_gnn", type=int, default=10)
    ap.add_argument("--R_eval", type=int, default=16)
    ap.add_argument("--eval_ticks", type=int, default=4000)
    ap.add_argument("--base", type=int, default=0)
    ap.add_argument("--out", type=str, default="step3_gpu_long.npz")
    a = ap.parse_args()
    main(a.ticks, a.nseed, a.R_mlp, a.R_gnn, a.R_eval, a.eval_ticks, a.base, a.out)
