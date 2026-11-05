#!/usr/bin/env python3
"""Recurrent (GRU-IPPO) learner for the embodied comparison (committee ask: a learner with LEARNABLE memory,
beyond the hand-crafted EMA features already in the observation). Same sensing-limited torch env, same PPO/GAE,
same shaped reward and eval (true frac-of-oracle) as train_gpu -- the ONLY change is a per-agent GRU policy
whose hidden state is carried across environment ticks, with truncated BPTT over each rollout.

All independent units (nseed x 2 operating points) run as ONE batched GPU kernel via grouped params. We do NOT
minibatch over worlds in the recurrent update (we re-run the full rollout sequence per epoch from a detached
start hidden) -- simpler and less bug-prone than sequence-minibatching; R is kept small and auto-halved on OOM.

  python train_recurrent.py test            # smoke: does the GRU LEARN? (beat random, approach MLP) + GPU check
  python train_recurrent.py full --ticks 150000   # 16 seeds x {PAPER,FAV} x 150k -> gru_recurrent.npz
"""
import os, sys, time, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch_env as TE
from torch_env import TorchForageEnv, N, F_OBS, C
from policies_batched import EnsembleGRU
import train_gpu as TG
from train_gpu import DEV, _world_params

_rd = os.path.normpath(os.path.join(HERE, "..", "data", "step3_gpu"))
DATA = os.environ.get("FLOCKING_DATA") or (_rd if os.path.isdir(_rd) else "/mnt/newssd/flocking_ltm_data/step3_gpu")
os.makedirs(DATA, exist_ok=True)
PAPER = dict(TG.PAPER); PAPER["gamma"] = 1.0        # match train_long: gamma pinned to 1 (no per-regime tuning)
FAV = dict(TG.FAV);     FAV["gamma"] = 1.0


def train_rec(policy_ops, R, ticks, seed, roll=48, h=64, lr=3e-4, shape=0.05, epochs=3,
              ent=0.01, gdisc=0.99, lam=0.95, window=1000, rec=1000, sample_gpu=False):
    M = len(policy_ops); W = M * R; B = R * N
    phi, patch, theta, gam = _world_params(policy_ops, R)
    env = TorchForageEnv(W, phi, patch, theta, gam, seed=seed, device=DEV)
    torch.manual_seed(seed + 12345)
    pol = EnsembleGRU(M, F_OBS, h).to(DEV)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    ie = torch.zeros(W, N, device=DEV); oe = torch.zeros(W, N, device=DEV)
    hlive = pol.init_hidden(M, B, DEV)                      # (M,B,H) carried across rollouts
    hist_tr, hist_ce = [], []; curve_t, curve_f = [], []
    gpu_util, gpu_mem = 0, 0; step = 0; n_upd = ticks // roll

    def feats_MB(feats_WN):                                 # (W,N,F) -> (M,B,F)
        return feats_WN.view(M, R, N, F_OBS).reshape(M, B, F_OBS)

    for up in range(n_upd):
        h0 = hlive.detach()                                 # start hidden for this rollout (truncation point)
        h = h0
        fB, aB, lpB, vB, rB = [], [], [], [], []
        with torch.no_grad():
            for _ in range(roll):
                capb = env.cap.sum(1)
                on_food, prox, feats = env.extract(ie, oe)  # (W,N,F); sets sensing-limited target
                x = feats_MB(feats)                         # (M,B,F)
                logits, val, h = pol.forward_step(x, h)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample(); lp = dist.log_prob(a)    # (M,B)
                intake = env.step(a.view(W, N), on_food)
                r = (intake + shape * prox).view(M, B)
                fB.append(x); aB.append(a); lpB.append(lp); vB.append(val); rB.append(r)
                ie = 0.9 * ie + 0.1 * intake; oe = 0.95 * oe + 0.05 * on_food.to(env.dtype)
                hist_tr.append(intake.sum(1)); hist_ce.append(torch.minimum(torch.tensor(float(N), device=DEV), capb))
                step += 1
                if step % rec == 0:
                    tr = torch.stack(hist_tr[-rec:]).sum(0); ce = torch.stack(hist_ce[-rec:]).sum(0)
                    curve_t.append(step); curve_f.append((tr / ce.clamp(min=1e-9)).view(M, R).mean(1).cpu().numpy())
            hlive = h                                       # carry end-of-rollout hidden to next rollout
            # bootstrap value
            on_food, prox, feats = env.extract(ie, oe)
            _, lastv, _ = pol.forward_step(feats_MB(feats), h)
        # GAE over the rollout, per (M,B)
        V = torch.stack(vB + [lastv]); Rw = torch.stack(rB)         # (roll+1,M,B),(roll,M,B)
        adv = torch.zeros(roll, M, B, device=DEV); gae = torch.zeros(M, B, device=DEV)
        for t in reversed(range(roll)):
            delta = Rw[t] + gdisc * V[t + 1] - V[t]; gae = delta + gdisc * lam * gae; adv[t] = gae
        ret = adv + V[:roll]
        mean = adv.mean(dim=(0, 2), keepdim=True); std = adv.std(dim=(0, 2), keepdim=True)
        advn = (adv - mean) / (std + 1e-8)
        fS = torch.stack(fB); aS = torch.stack(aB); lpS = torch.stack(lpB)   # (roll,M,B,.)
        for _ in range(epochs):
            hh = h0; ploss = vloss = eloss = 0.0
            for t in range(roll):
                logits, val, hh = pol.forward_step(fS[t], hh)
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(aS[t]); ratio = torch.exp(lp - lpS[t])
                a1 = ratio * advn[t]; a2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advn[t]
                ploss = ploss - torch.min(a1, a2).mean()
                vloss = vloss + ((val - ret[t]) ** 2).mean()
                eloss = eloss + dist.entropy().mean()
            loss = (ploss + 0.5 * vloss - ent * eloss) / roll
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(pol.parameters(), 0.5); opt.step()
        if sample_gpu and up % 20 == 0 and DEV == "cuda":
            try:
                q = os.popen("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits").read()
                u, msz = q.strip().split("\n")[0].split(","); gpu_util = max(gpu_util, int(u)); gpu_mem = max(gpu_mem, int(msz))
            except Exception:
                pass
    tr = torch.stack(hist_tr[-window:]).sum(0); ce = torch.stack(hist_ce[-window:]).sum(0)
    finals = (tr / ce.clamp(min=1e-9)).view(M, R).mean(1).cpu().numpy()
    return dict(finals=finals, curve_t=np.array(curve_t), curve_f=np.array(curve_f).reshape(-1, M),
                gpu_util=gpu_util, gpu_mem=gpu_mem, pol=pol)


def eval_rec(pol, policy_ops, R, ticks, seed, window=2000):
    M = len(policy_ops); W = M * R; B = R * N
    phi, patch, theta, gam = _world_params(policy_ops, R)
    env = TorchForageEnv(W, phi, patch, theta, gam, seed=seed, device=DEV)
    ie = torch.zeros(W, N, device=DEV); oe = torch.zeros(W, N, device=DEV)
    h = pol.init_hidden(M, B, DEV)
    hist_tr, hist_ce = [], []; occ = torch.zeros(W, C, device=DEV); nmeas = 0
    with torch.no_grad():
        for t in range(ticks):
            capb = env.cap.sum(1)
            on_food, prox, feats = env.extract(ie, oe)
            logits, _, h = pol.forward_step(feats.view(M, R, N, F_OBS).reshape(M, B, F_OBS), h)
            a = torch.distributions.Categorical(logits=logits).sample()
            intake = env.step(a.view(W, N), on_food)
            ie = 0.9 * ie + 0.1 * intake; oe = 0.95 * oe + 0.05 * on_food.to(env.dtype)
            if t >= ticks - window:
                hist_tr.append(intake.sum(1)); hist_ce.append(torch.minimum(torch.tensor(float(N), device=DEV), capb))
                occ += env.occupancy(); nmeas += 1
    tr = torch.stack(hist_tr).sum(0); ce = torch.stack(hist_ce).sum(0)
    finals = (tr / ce.clamp(min=1e-9)).view(M, R).mean(1).cpu().numpy()
    return finals, env.rich.cpu().numpy(), (occ / max(nmeas, 1)).cpu().numpy()


def _oom_retry(fn, R):
    while True:
        try:
            return fn(R), R
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and R > 2:
                torch.cuda.empty_cache(); R //= 2; print(f"  [OOM] retry R={R}", flush=True)
            else:
                raise


def cmd_test():
    t0 = time.time(); print(f"device={DEV}")
    print("\n[LEARN] GRU-IPPO @PAPER (scarce), 4 seeds, 8000 ticks  (refs: random 0.141, swarm 0.228, MLP ~0.33):")
    r = train_rec([PAPER] * 4, R=8, ticks=8000, seed=0, sample_gpu=True, rec=1000)
    print(f"  finals/seed: {np.round(r['finals'],3)}  mean={r['finals'].mean():.3f}")
    print(f"  curve(mean): {np.round(r['curve_f'].mean(1),3)}")
    print(f"  GPU util(max)={r['gpu_util']}%  VRAM(max)={r['gpu_mem']}MiB   ({time.time()-t0:.0f}s)")
    verdict = "LEARNS (>random)" if r['finals'].mean() > 0.141 + 0.03 else "NOT clearly learning"
    print(f"  --> {verdict}")
    print(f"\ntest done ({time.time()-t0:.0f}s)")


def cmd_full(ticks, nseed, R_train, R_eval, eval_ticks, base, out):
    POINTS = [("PAPER", PAPER), ("FAVORABLE", FAV)]
    ops = [op for _, op in POINTS for _ in range(nseed)]
    pslice = {"PAPER": slice(0, nseed), "FAVORABLE": slice(nseed, 2 * nseed)}
    wslice = lambda R: {"PAPER": slice(0, nseed * R), "FAVORABLE": slice(nseed * R, 2 * nseed * R)}
    t0 = time.time()
    tr, R = _oom_retry(lambda R: train_rec(ops, R, ticks=ticks, seed=base, sample_gpu=True), R_train)
    ev, rich, occ = eval_rec(tr["pol"], ops, R_eval, eval_ticks, seed=base + 10000)
    res = {}
    for name in ["PAPER", "FAVORABLE"]:
        ws = wslice(R_eval)[name]
        res[name] = dict(gru_finals=ev[pslice[name]],
                         gru_curve_f=tr["curve_f"][:, pslice[name]], gru_ct=tr["curve_t"],
                         gru_slope=TE.slope_pooled(rich[ws], occ[ws]),
                         gru_rich=rich[ws], gru_occ=occ[ws])
        print(f"[GRU {name}] frac {ev[pslice[name]].mean():.3f}  slope {res[name]['gru_slope']:.3f}", flush=True)
    res["_meta"] = dict(ticks=ticks, nseed=nseed, R_train=R, R_eval=R_eval, roll=48, util=tr["gpu_util"], mem=tr["gpu_mem"])
    outp = os.path.join(DATA, out); np.savez_compressed(outp, res=res)
    print(f"wrote {outp}  util={tr['gpu_util']}% vram={tr['gpu_mem']}MiB  ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="test", choices=["test", "full"])
    ap.add_argument("--ticks", type=int, default=150000)
    ap.add_argument("--nseed", type=int, default=16)
    ap.add_argument("--R_train", type=int, default=12)
    ap.add_argument("--R_eval", type=int, default=16)
    ap.add_argument("--eval_ticks", type=int, default=4000)
    ap.add_argument("--base", type=int, default=0)
    ap.add_argument("--out", type=str, default="gru_recurrent.npz")
    a = ap.parse_args()
    if a.cmd == "test":
        cmd_test()
    else:
        cmd_full(a.ticks, a.nseed, a.R_train, a.R_eval, a.eval_ticks, a.base, a.out)
