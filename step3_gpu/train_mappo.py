#!/usr/bin/env python3
"""MAPPO (centralized-critic PPO) arm for the F1 sensing-limited comparison, NON-destructive.

Adds the strong standard MARL baseline the committee asked for: centralized training / decentralized
execution (CTDE). The ACTOR is identical to the IPPO actors (EnsembleMLP / EnsembleGNN) and still executes
under the SAME sensing-limited primitives, so the comparison stays fair; only the CRITIC is centralized --
at training time it sees a privileged global state (pooled agent observations + per-patch occupancy and
richness) and returns ONE value per world, used as the GAE baseline for every agent. This is textbook MAPPO
(Yu et al. 2022). If even a centralized-critic learner with a global view does not out-allocate the
reward-free swarm, the central claim is strengthened; if it does, we report it honestly (the cost asymmetry
-- 0 training/samples vs T -- is unaffected either way).

Reuses train_gpu (actor classes, eval_policy, _with_oom_retry, _adj_from_pos, _world_params) and train_long
(exploit_per_world, per_seed, the gamma=1.0 operating points) UNCHANGED. torch_env.py / train_gpu.py /
train_long.py are NOT modified. Output -> step3_gpu_mappo.npz, merged post-hoc with step3_gpu_long.npz
(same seeds/R -> directly comparable). Run:  python train_mappo.py --ticks 150000
"""
import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch_env as TE
from torch_env import TorchForageEnv, N, F_OBS, C
from train_gpu import eval_policy, DEV, _with_oom_retry, _adj_from_pos, _world_params
from policies_batched import EnsembleMLP, EnsembleGNN, GroupedLinear
from train_long import PAPER, FAV, exploit_per_world, per_seed

_rd = os.path.normpath(os.path.join(HERE, "..", "data", "step3_gpu"))
DATA = os.environ.get("FLOCKING_DATA") or (_rd if os.path.isdir(_rd) else "/mnt/newssd/flocking_ltm_data/step3_gpu")
os.makedirs(DATA, exist_ok=True)

# Centralized global-state dim: [mean_feats (F), max_feats (F), occ/N (C), rich_norm (C)].
GDIM = 2 * F_OBS + 2 * C


class CentralCritic(nn.Module):
    """M independent centralized critics (one value PER WORLD, not per agent). g (M,U,GDIM) -> V (M,U)."""
    def __init__(self, M, gdim, h=64):
        super().__init__()
        self.c1 = GroupedLinear(M, gdim, h); self.c2 = GroupedLinear(M, h, h); self.co = GroupedLinear(M, h, 1)

    def forward(self, g):                       # g (M,U,GDIM)
        x = torch.relu(self.c1(g)); x = torch.relu(self.c2(x))
        return self.co(x).squeeze(-1)           # (M,U)


def _global_state(feats, env):
    """Privileged per-world global state for the centralized critic. feats (W,N,F_OBS) -> (W,GDIM).
    Permutation-invariant over agents (mean+max pooling) plus the true per-patch occupancy and richness,
    which no single sensing-limited agent observes -- this is the training-time centralization."""
    mean_f = feats.mean(1)                                   # (W,F_OBS)
    max_f = feats.max(1).values                              # (W,F_OBS)
    occ = env.occupancy() / float(N)                         # (W,C) fraction of agents per patch
    rich = env.rich
    rich = rich / (rich.max(1, keepdim=True).values + 1e-9)  # (W,C) per-world normalized richness
    return torch.cat([mean_f, max_f, occ, rich], dim=-1)     # (W,GDIM)


def train_mappo(policy_ops, R, ticks, seed, L=None, shape=0.05, lr=3e-4, roll=128, h=64, epochs=4, nmb=8,
                ent=0.01, gdisc=0.99, lam=0.95, clip=0.2, window=1000, sample_gpu=False, rec=1000):
    """Mirror of train_gpu.train but with a CENTRALIZED critic. Actor forward uses logits only; the actor's
    own (unused) value head receives no gradient. Returns dict(pol=actor, ...) so eval_policy works as-is."""
    M = len(policy_ops); W = M * R
    phi, patch, theta, gam = _world_params(policy_ops, R)
    env = TorchForageEnv(W, phi, patch, theta, gam, seed=seed, device=DEV)
    torch.manual_seed(seed + 12345)
    actor = (EnsembleMLP(M, F_OBS, h) if L is None else EnsembleGNN(M, F_OBS, h, L)).to(DEV)
    gnn = L is not None
    critic = CentralCritic(M, GDIM, h).to(DEV)
    params = list(actor.parameters()) + list(critic.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    ie = torch.zeros(W, N, device=DEV); oe = torch.zeros(W, N, device=DEV)
    hist_tr, hist_ce = [], []; curve_t, curve_f = [], []
    gpu_util, gpu_mem = 0, 0; step = 0; n_upd = ticks // roll
    for up in range(n_upd):
        fB, pB, gB, aB, lpB, vB, rB = [], [], [], [], [], [], []
        for _ in range(roll):
            capb = env.cap.sum(1)
            on_food, prox, feats = env.extract(ie, oe)        # (W,N,F); sets sensing-limited _target
            pos_obs = env.pos.clone()
            g = _global_state(feats, env)                     # (W,GDIM)
            fr = feats.view(M, R, N, F_OBS)
            adj = _adj_from_pos(pos_obs).view(M, R, N, N) if gnn else None
            with torch.no_grad():
                logits, _ = actor(fr, adj)                    # actor's own value head ignored
                valw = critic(g.view(M, R, GDIM))             # (M,R) one value per world
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample(); lp = dist.log_prob(a)
            intake = env.step(a.view(W, N), on_food)
            r = intake + shape * prox
            fB.append(feats); pB.append(pos_obs); gB.append(g); aB.append(a.view(W, N))
            lpB.append(lp.view(W, N)); vB.append(valw.reshape(W)); rB.append(r)
            ie = 0.9 * ie + 0.1 * intake; oe = 0.95 * oe + 0.05 * on_food.to(env.dtype)
            hist_tr.append(intake.sum(1)); hist_ce.append(torch.minimum(torch.tensor(float(N), device=DEV), capb))
            step += 1
            if step % rec == 0:
                tr = torch.stack(hist_tr[-rec:]).sum(0); ce = torch.stack(hist_ce[-rec:]).sum(0)
                fpp = (tr / ce.clamp(min=1e-9)).view(M, R).mean(1)
                curve_t.append(step); curve_f.append(fpp.cpu().numpy())
        # bootstrap world value (no adjacency needed for the critic)
        with torch.no_grad():
            on_food, prox, feats = env.extract(ie, oe)
            g = _global_state(feats, env)
            lastv = critic(g.view(M, R, GDIM)).reshape(W)     # (W,)
        # per-agent GAE with the world value broadcast as the shared baseline (canonical MAPPO)
        Vw = torch.stack(vB + [lastv])                        # (roll+1, W)
        V = Vw.unsqueeze(-1).expand(roll + 1, W, N)           # (roll+1, W, N)
        Rw = torch.stack(rB)                                  # (roll, W, N)
        adv = torch.zeros(roll, W, N, device=DEV); gae = torch.zeros(W, N, device=DEV)
        for t in reversed(range(roll)):
            delta = Rw[t] + gdisc * V[t + 1] - V[t]; gae = delta + gdisc * lam * gae; adv[t] = gae
        ret = adv + V[:roll]                                  # (roll,W,N) per-agent returns
        retw = ret.mean(2)                                    # (roll,W) per-world value target

        def ppN(x, extra=()):                                 # (roll,W,N,*) -> (M, roll*R, N,*)
            x = x.view(roll, M, R, N, *extra).permute(1, 0, 2, 3, *[4 + i for i in range(len(extra))])
            return x.reshape(M, roll * R, N, *extra)

        def ppG(x):                                           # (roll,W,GDIM) -> (M, roll*R, GDIM)
            return x.view(roll, M, R, GDIM).permute(1, 0, 2, 3).reshape(M, roll * R, GDIM)

        def ppW(x):                                           # (roll,W) -> (M, roll*R)
            return x.view(roll, M, R).permute(1, 0, 2).reshape(M, roll * R)

        fP = ppN(torch.stack(fB), (F_OBS,)); pP = ppN(torch.stack(pB), (2,)); gP = ppG(torch.stack(gB))
        aP = ppN(torch.stack(aB)); lpP = ppN(torch.stack(lpB)); advP = ppN(adv); retwP = ppW(retw)
        mean = advP.mean(dim=(1, 2), keepdim=True); std = advP.std(dim=(1, 2), keepdim=True)
        advP = (advP - mean) / (std + 1e-8)
        U = roll * R; idx = np.arange(U); mb = max(1, U // nmb)
        for _ in range(epochs):
            np.random.shuffle(idx)
            for s in range(0, U, mb):
                j = idx[s:s + mb]
                adjm = _adj_from_pos(pP[:, j].reshape(M * len(j), N, 2)).view(M, len(j), N, N) if gnn else None
                logits, _ = actor(fP[:, j], adjm)
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(aP[:, j]); ratio = torch.exp(lp - lpP[:, j])
                a1 = ratio * advP[:, j]; a2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advP[:, j]
                ploss = -torch.min(a1, a2).mean()
                valw = critic(gP[:, j])                       # (M, len(j)) per-world value
                vloss = ((valw - retwP[:, j]) ** 2).mean()
                loss = ploss + 0.5 * vloss - ent * dist.entropy().mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 0.5); opt.step()
        if sample_gpu and up % 3 == 0 and DEV == "cuda":
            try:
                q = os.popen("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits").read()
                u, msz = q.strip().split("\n")[0].split(",")
                gpu_util = max(gpu_util, int(u)); gpu_mem = max(gpu_mem, int(msz))
            except Exception:
                pass
    tr = torch.stack(hist_tr[-window:]).sum(0); ce = torch.stack(hist_ce[-window:]).sum(0)
    finals = (tr / ce.clamp(min=1e-9)).view(M, R).mean(1).cpu().numpy()
    return dict(finals=finals, curve_t=np.array(curve_t), curve_f=np.array(curve_f).reshape(-1, M),
                gpu_util=gpu_util, gpu_mem=gpu_mem, pol=actor)


ARM_DEFS = {"mappo_mlp": (None, "R_mlp"), "mappo_gnn_L1": (1, "R_gnn"),
            "mappo_gnn_L3": (3, "R_gnn"), "mappo_gnn_L5": (5, "R_gnn")}


def main(ticks, nseed, R_mlp, R_gnn, R_eval, eval_ticks, base, out, arms):
    POINTS = [("PAPER", PAPER), ("FAVORABLE", FAV)]
    ops = [op for _, op in POINTS for _ in range(nseed)]       # M = 2*nseed, PAPER block then FAV block
    res = {name: {} for name, _ in POINTS}
    pslice = {"PAPER": slice(0, nseed), "FAVORABLE": slice(nseed, 2 * nseed)}
    wslice = lambda R: {"PAPER": slice(0, nseed * R), "FAVORABLE": slice(nseed * R, 2 * nseed * R)}
    Rmap = {"R_mlp": R_mlp, "R_gnn": R_gnn}
    t0 = time.time()

    def run_cfg(tag, L, R0):
        (tr, R) = _with_oom_retry(lambda R: train_mappo(ops, R, ticks=ticks, seed=base, L=L, sample_gpu=True), R0)
        ev, rich, occ = eval_policy(tr["pol"], ops, R_eval, eval_ticks, seed=base + 10000, L=L)
        print(f"[{tag}] R={R} ticks={ticks} eval frac: PAPER {ev[pslice['PAPER']].mean():.3f} "
              f"FAV {ev[pslice['FAVORABLE']].mean():.3f}  util={tr['gpu_util']}% vram={tr['gpu_mem']}MiB "
              f"({time.time()-t0:.0f}s)", flush=True)
        return tr, ev, rich, occ, exploit_per_world(rich, occ)

    for key in arms:
        L, Rkey = ARM_DEFS[key]
        tr, ev, rich, occ, ex = run_cfg(key.upper(), L, Rmap[Rkey])
        for name in res:
            res[name][f"{key}_finals"] = ev[pslice[name]]
            res[name][f"{key}_curve_f"] = tr["curve_f"][:, pslice[name]]; res[name][f"{key}_ct"] = tr["curve_t"]
            ws = wslice(R_eval)[name]
            res[name][f"{key}_slope"] = TE.slope_pooled(rich[ws], occ[ws])
            res[name][f"{key}_exploit"] = per_seed(ex[ws], nseed, R_eval)
            res[name][f"{key}_rich"] = rich[ws]; res[name][f"{key}_occ"] = occ[ws]
        if DEV == "cuda":
            torch.cuda.empty_cache()

    res["_meta"] = dict(ticks=ticks, nseed=nseed, R_mlp=R_mlp, R_gnn=R_gnn, R_eval=R_eval,
                        eval_ticks=eval_ticks, arms=arms, gdim=GDIM)
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
    ap.add_argument("--out", type=str, default="step3_gpu_mappo.npz")
    ap.add_argument("--arms", type=str, default="mappo_mlp,mappo_gnn_L3")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU-sized config to validate shapes/learning")
    a = ap.parse_args()
    arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    if a.smoke:
        print(f"[SMOKE] device={DEV} GDIM={GDIM}")
        main(ticks=256, nseed=1, R_mlp=2, R_gnn=2, R_eval=2, eval_ticks=200, base=0,
             out="step3_gpu_mappo_smoke.npz", arms=arms)
    else:
        main(a.ticks, a.nseed, a.R_mlp, a.R_gnn, a.R_eval, a.eval_ticks, a.base, a.out, arms)
