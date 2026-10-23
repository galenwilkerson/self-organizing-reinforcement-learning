#!/usr/bin/env python3
"""Batched-ensemble PPO on the GPU torch env (torch_env.TorchForageEnv). All independent policies (seeds ×
operating points × algo/L) run as ONE batched kernel via grouped params (policies_batched). Per-policy GAE,
advantage norm, and minibatching. Reward = per-capita intake + shape*prox; eval = true frac-of-oracle.

`test` runs the validation gates (G3 ensemble-correctness, G4 PPO-sanity, G5 GPU util, G6 learning) and STOPS.
`full` runs the real comparison (16 seeds, PAPER+FAVORABLE, MLP-IPPO + GNN L∈{1,3,5}). MAPPO skipped (IPPO
only) to limit scope. Data -> /mnt/newssd/flocking_ltm_data/step3_gpu/.
"""
import os, sys, time, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch_env as TE
from torch_env import TorchForageEnv, N, F_OBS, R_NEIGH, L as LSIZE
from policies_batched import EnsembleMLP, EnsembleGNN

_rd = os.path.normpath(os.path.join(HERE, "..", "data", "step3_gpu"))
DATA = os.environ.get("FLOCKING_DATA") or (_rd if os.path.isdir(_rd) else "/mnt/newssd/flocking_ltm_data/step3_gpu")
os.makedirs(DATA, exist_ok=True)
PAPER = dict(phi=0.02, patchiness=0.0375, theta=0.18, gamma=1.0)
FAV = dict(phi=0.08, patchiness=0.02, theta=0.18, gamma=0.8)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _adj_from_pos(pos):                           # pos (W,N,2) -> adj (W,N,N) float {0,1}, no self
    size = torch.tensor([LSIZE, LSIZE], device=pos.device, dtype=pos.dtype)
    diff = pos.unsqueeze(2) - pos.unsqueeze(1)
    diff = diff - size * torch.round(diff / size)
    d = diff.norm(dim=-1)
    W, n, _ = d.shape
    eye = torch.eye(n, dtype=torch.bool, device=pos.device).unsqueeze(0)
    return ((d <= R_NEIGH) & (~eye)).to(pos.dtype)


def _world_params(policy_ops, R):
    """policy_ops: list of op-dicts (len M). Return per-world (W=M*R,) tensors."""
    def col(key):
        return torch.tensor([op[key] for op in policy_ops], dtype=torch.float32).repeat_interleave(R)
    return col("phi"), col("patchiness"), col("theta"), col("gamma")


def train(policy_ops, R, ticks, seed, L=None, shape=0.05, lr=3e-4, roll=128, h=64, epochs=4, nmb=8,
          ent=0.01, gdisc=0.99, lam=0.95, clip=0.2, window=1000, sample_gpu=False, rec=1000):
    M = len(policy_ops); W = M * R
    phi, patch, theta, gam = _world_params(policy_ops, R)
    env = TorchForageEnv(W, phi, patch, theta, gam, seed=seed, device=DEV)
    torch.manual_seed(seed + 12345)
    if L is None:
        pol = EnsembleMLP(M, F_OBS, h).to(DEV); gnn = False
    else:
        pol = EnsembleGNN(M, F_OBS, h, L).to(DEV); gnn = True
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    ie = torch.zeros(W, N, device=DEV); oe = torch.zeros(W, N, device=DEV)
    hist_tr, hist_ce = [], []; curve_t, curve_f = [], []
    gpu_util, gpu_mem = 0, 0; step = 0; n_upd = ticks // roll
    for up in range(n_upd):
        fB, pB, aB, lpB, vB, rB = [], [], [], [], [], []
        for _ in range(roll):
            capb = env.cap.sum(1)
            on_food, prox, feats = env.extract(ie, oe)        # (W,N,F); sets sensing-limited _target
            pos_obs = env.pos.clone()
            fr = feats.view(M, R, N, F_OBS)
            adj = _adj_from_pos(pos_obs).view(M, R, N, N) if gnn else None
            with torch.no_grad():
                logits, val = pol(fr, adj)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample(); lp = dist.log_prob(a)
            intake = env.step(a.view(W, N), on_food)
            r = intake + shape * prox
            fB.append(feats); pB.append(pos_obs); aB.append(a.view(W, N)); lpB.append(lp.view(W, N))
            vB.append(val.view(W, N)); rB.append(r)
            ie = 0.9 * ie + 0.1 * intake; oe = 0.95 * oe + 0.05 * on_food.to(env.dtype)
            hist_tr.append(intake.sum(1)); hist_ce.append(torch.minimum(torch.tensor(float(N), device=DEV), capb))
            step += 1
            if step % rec == 0:
                tr = torch.stack(hist_tr[-rec:]).sum(0); ce = torch.stack(hist_ce[-rec:]).sum(0)
                fpp = (tr / ce.clamp(min=1e-9)).view(M, R).mean(1)       # per-policy frac
                curve_t.append(step); curve_f.append(fpp.cpu().numpy())
        # bootstrap value
        with torch.no_grad():
            capb = env.cap.sum(1)
            on_food, prox, feats = env.extract(ie, oe)
            adj = _adj_from_pos(env.pos).view(M, R, N, N) if gnn else None
            _, lastv = pol(feats.view(M, R, N, F_OBS), adj); lastv = lastv.view(W, N)
        V = torch.stack(vB + [lastv]); Rw = torch.stack(rB)          # (roll+1,W,N),(roll,W,N)
        adv = torch.zeros(roll, W, N, device=DEV); gae = torch.zeros(W, N, device=DEV)
        for t in reversed(range(roll)):
            delta = Rw[t] + gdisc * V[t + 1] - V[t]; gae = delta + gdisc * lam * gae; adv[t] = gae
        ret = adv + V[:roll]
        # reshape to (M, U=roll*R, N, .)
        def pp(x, extra=()):                                          # (roll,W,N,*) -> (M, roll*R, N,*)
            x = x.view(roll, M, R, N, *extra).permute(1, 0, 2, 3, *[4 + i for i in range(len(extra))])
            return x.reshape(M, roll * R, N, *extra)
        fP = pp(torch.stack(fB), (F_OBS,)); pP = pp(torch.stack(pB), (2,))
        aP = pp(torch.stack(aB)); lpP = pp(torch.stack(lpB)); advP = pp(adv); retP = pp(ret)
        mean = advP.mean(dim=(1, 2), keepdim=True); std = advP.std(dim=(1, 2), keepdim=True)
        advP = (advP - mean) / (std + 1e-8)
        U = roll * R; idx = np.arange(U); mb = max(1, U // nmb)
        for _ in range(epochs):
            np.random.shuffle(idx)
            for s in range(0, U, mb):
                j = idx[s:s + mb]
                adjm = _adj_from_pos(pP[:, j].reshape(M * len(j), N, 2)).view(M, len(j), N, N) if gnn else None
                logits, val = pol(fP[:, j], adjm)
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(aP[:, j]); ratio = torch.exp(lp - lpP[:, j])
                a1 = ratio * advP[:, j]; a2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advP[:, j]
                ploss = -torch.min(a1, a2).mean()
                vloss = ((val - retP[:, j]) ** 2).mean()
                loss = ploss + 0.5 * vloss - ent * dist.entropy().mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(pol.parameters(), 0.5); opt.step()
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
                gpu_util=gpu_util, gpu_mem=gpu_mem, pol=pol)


# ---------------- G4: abstract congestion game with this grouped PPO ----------------
def abstract_game(M=4, steps=20000, Kp=80, seed=0, ent=0.0, greedy=True):
    torch.manual_seed(seed); g = torch.Generator(device=DEV); g.manual_seed(seed)
    b = max(2.2 - 1.0, 0.15)
    w = torch.rand(M, Kp, generator=g, device=DEV).clamp(min=1e-6).pow(-1.0 / b); w = w / w.sum(1, keepdim=True)
    cidx = torch.multinomial(w, 80, replacement=True, generator=g)
    Q = torch.zeros(M, Kp, device=DEV); Q.scatter_add_(1, cidx, torch.ones_like(cidx, dtype=torch.float32))
    Fo = 8; emb = torch.randn(M, N, Fo, generator=g, device=DEV)
    from policies_batched import GroupedLinear
    import torch.nn as nn
    actor = nn.Sequential(GroupedLinear(M, Fo, 64)).to(DEV)
    a1 = GroupedLinear(M, Fo, 64).to(DEV); a2 = GroupedLinear(M, 64, Kp).to(DEV)
    c1 = GroupedLinear(M, Fo, 64).to(DEV); c2 = GroupedLinear(M, 64, 1).to(DEV)
    params = list(a1.parameters()) + list(a2.parameters()) + list(c1.parameters()) + list(c2.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)

    def fwd(x):
        h = torch.relu(a1(x)); logits = a2(h); v = c2(torch.relu(c1(x))).squeeze(-1); return logits, v
    roll = 64
    for it in range(steps // roll):
        OB, AC, LP, RW = [], [], [], []
        for _ in range(roll):
            logits, v = fwd(emb)
            dist = torch.distributions.Categorical(logits=logits); a = dist.sample(); lp = dist.log_prob(a)
            oh = torch.zeros(M, N, Kp, device=DEV); oh.scatter_(2, a.unsqueeze(-1), 1.0)
            n = oh.sum(1) / N; na = torch.gather(n, 1, a); f = torch.gather(Q, 1, a) / (N * na).clamp(min=1e-9)
            OB.append(emb); AC.append(a); LP.append(lp.detach()); RW.append(f)
        O = torch.stack(OB); A = torch.stack(AC); LPc = torch.stack(LP); Rr = torch.stack(RW)
        adv = Rr - Rr.mean(dim=(0, 2), keepdim=True); adv = adv / (Rr.std(dim=(0, 2), keepdim=True) + 1e-8)
        for _ in range(4):
            logits, v = fwd(O.reshape(roll * M, N, Fo).view(M, roll * N, Fo) if False else O.permute(1, 0, 2, 3).reshape(M, roll * N, Fo))
            dist = torch.distributions.Categorical(logits=logits)
            Af = A.permute(1, 0, 2).reshape(M, roll * N); LPf = LPc.permute(1, 0, 2).reshape(M, roll * N)
            ADf = adv.permute(1, 0, 2).reshape(M, roll * N)
            lp = dist.log_prob(Af); ratio = torch.exp(lp - LPf)
            l = -torch.min(ratio * ADf, torch.clamp(ratio, 0.8, 1.2) * ADf).mean() - ent * dist.entropy().mean()
            opt.zero_grad(); l.backward(); opt.step()
    with torch.no_grad():
        logits, _ = fwd(emb)
        a = logits.argmax(-1) if greedy else torch.distributions.Categorical(logits=logits).sample()
        occ = torch.zeros(M, Kp, device=DEV); occ.scatter_add_(1, a, torch.ones_like(a, dtype=torch.float32))
    Qn = Q.cpu().numpy(); occn = occ.cpu().numpy(); slopes = []
    for m in range(M):
        r = Qn[m]; o = occn[m]; msk = (r >= 1) & (o >= 0.5)
        if msk.sum() >= 8:
            slopes.append(np.polyfit(np.log(r[msk]), np.log(o[msk]), 1)[0])
    return float(np.mean(slopes))


def eval_policy(pol, policy_ops, R, ticks, seed, L=None, window=2000):
    """Run the FROZEN policy on fresh worlds (disjoint eval seed). Returns per-policy frac-of-oracle (M,),
    plus rich (W,C) and mean window occupancy (W,C) for IFD slopes."""
    M = len(policy_ops); W = M * R; gnn = L is not None
    phi, patch, theta, gam = _world_params(policy_ops, R)
    env = TorchForageEnv(W, phi, patch, theta, gam, seed=seed, device=DEV)
    ie = torch.zeros(W, N, device=DEV); oe = torch.zeros(W, N, device=DEV)
    hist_tr, hist_ce = [], []; occ = torch.zeros(W, TE.C, device=DEV); nmeas = 0
    with torch.no_grad():
        for t in range(ticks):
            capb = env.cap.sum(1)
            on_food, prox, feats = env.extract(ie, oe)
            adj = _adj_from_pos(env.pos).view(M, R, N, N) if gnn else None
            logits, _ = pol(feats.view(M, R, N, F_OBS), adj)
            a = torch.distributions.Categorical(logits=logits).sample()
            intake = env.step(a.view(W, N), on_food)
            ie = 0.9 * ie + 0.1 * intake; oe = 0.95 * oe + 0.05 * on_food.to(env.dtype)
            if t >= ticks - window:
                hist_tr.append(intake.sum(1)); hist_ce.append(torch.minimum(torch.tensor(float(N), device=DEV), capb))
                occ += env.occupancy(); nmeas += 1
    tr = torch.stack(hist_tr).sum(0); ce = torch.stack(hist_ce).sum(0)
    finals = (tr / ce.clamp(min=1e-9)).view(M, R).mean(1).cpu().numpy()
    return finals, env.rich.cpu().numpy(), (occ / max(nmeas, 1)).cpu().numpy()


def _with_oom_retry(fn, R, **kw):
    """Run fn(R=...) halving R on CUDA OOM so the full run can't crash out."""
    while True:
        try:
            return fn(R=R, **kw), R
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and R > 2:
                torch.cuda.empty_cache(); R = R // 2
                print(f"  [OOM] retrying at R={R}", flush=True)
            else:
                raise


def cmd_full(ticks=15000, nseed=16, R_mlp=24, R_gnn=10, R_eval=16, eval_ticks=4000, base=0):
    POINTS = [("PAPER", PAPER), ("FAVORABLE", FAV)]
    ops = [op for _, op in POINTS for _ in range(nseed)]       # M = 2*nseed, PAPER block then FAV block
    res = {name: {} for name, _ in POINTS}
    pslice = {"PAPER": slice(0, nseed), "FAVORABLE": slice(nseed, 2 * nseed)}
    wslice = lambda R: {"PAPER": slice(0, nseed * R), "FAVORABLE": slice(nseed * R, 2 * nseed * R)}
    t0 = time.time()

    def run_cfg(tag, L, R0):
        (tr, R) = _with_oom_retry(lambda R: train(ops, R, ticks=ticks, seed=base, L=L, sample_gpu=True), R0)
        ev, rich, occ = eval_policy(tr["pol"], ops, R_eval, eval_ticks, seed=base + 10000, L=L)
        print(f"[{tag}] R={R} eval frac: PAPER {ev[pslice['PAPER']].mean():.3f} "
              f"FAV {ev[pslice['FAVORABLE']].mean():.3f}  util={tr['gpu_util']}% vram={tr['gpu_mem']}MiB "
              f"({time.time()-t0:.0f}s)", flush=True)
        return tr, ev, rich, occ

    # ---- MLP-IPPO ----
    trm, evm, richm, occm = run_cfg("MLP-IPPO", None, R_mlp)
    for name in res:
        res[name]["mlp_ippo_finals"] = evm[pslice[name]]
        res[name]["mlp_ippo_curve_f"] = trm["curve_f"][:, pslice[name]]; res[name]["mlp_ippo_ct"] = trm["curve_t"]
        ws = wslice(R_eval)[name]
        res[name]["mlp_ippo_slope"] = TE.slope_pooled(richm[ws], occm[ws])
    # ---- GNN-IPPO per L ----
    for L in (1, 3, 5):
        trg, evg, richg, occg = run_cfg(f"GNN-L{L}", L, R_gnn)
        for name in res:
            res[name][f"gnn_L{L}_finals"] = evg[pslice[name]]
            res[name][f"gnn_L{L}_curve_f"] = trg["curve_f"][:, pslice[name]]; res[name][f"gnn_L{L}_ct"] = trg["curve_t"]
            ws = wslice(R_eval)[name]
            res[name][f"gnn_L{L}_slope"] = TE.slope_pooled(richg[ws], occg[ws])
    # ---- reward-free swarm + random baselines (per world for CIs) ----
    for name, op in POINTS:
        sf, ss = TE._eval("swarm", 256, eval_ticks, 2000, op, base + 555)
        rf, _ = TE._eval("random", 256, eval_ticks, 2000, op, base + 555)
        res[name]["swarm"] = sf; res[name]["swarm_slope"] = ss; res[name]["random"] = rf
        print(f"[baseline {name}] swarm {sf.mean():.3f}±{sf.std():.3f} (slope {ss:.2f})  "
              f"random {rf.mean():.3f}  ({time.time()-t0:.0f}s)", flush=True)
    out = os.path.join(DATA, "step3_gpu_full.npz")
    np.savez_compressed(out, res=res)
    print(f"wrote {out}  ({time.time()-t0:.0f}s total)", flush=True)


def cmd_test():
    t0 = time.time()
    print(f"device={DEV}")
    # G6 + G5: short MLP ensemble at PAPER, 4 seeds
    print("\n[G6/G5] MLP-IPPO @PAPER, 4 seeds, 4000 ticks:")
    r = train([PAPER] * 4, R=8, ticks=4000, seed=0, sample_gpu=True, rec=500)
    print(f"  finals per seed: {np.round(r['finals'],3)}  mean={r['finals'].mean():.3f} "
          f"(random 0.141, swarm 0.228)")
    print(f"  curve(frac mean over seeds): {np.round(r['curve_f'].mean(1),3)}")
    print(f"  GPU util(max)={r['gpu_util']}%  VRAM(max)={r['gpu_mem']}MiB   ({time.time()-t0:.0f}s)")
    # G3: ensemble(4) vs single(4), same seeds/op
    print("\n[G3] ensemble(4 seeds) vs 4 single runs @PAPER, 3000 ticks:")
    ens = train([PAPER] * 4, R=8, ticks=3000, seed=0)["finals"]
    sing = np.array([train([PAPER], R=8, ticks=3000, seed=0)["finals"][0] for _ in range(4)])
    print(f"  ensemble finals: {np.round(ens,3)} mean={ens.mean():.3f}")
    print(f"  single finals:   {np.round(sing,3)} mean={sing.mean():.3f}")
    print(f"  |mean diff|={abs(ens.mean()-sing.mean()):.3f} (should be within seed noise ~std {sing.std():.3f})")
    # G6b: GNN short
    print("\n[G6b] GNN(L=3)-IPPO @PAPER, 2 seeds, 3000 ticks:")
    rg = train([PAPER] * 2, R=8, ticks=3000, seed=0, L=3, rec=500)
    print(f"  finals: {np.round(rg['finals'],3)} mean={rg['finals'].mean():.3f} (above random 0.141?)")
    # G4
    print("\n[G4] abstract congestion game (grouped PPO), slope->1?:")
    sl = abstract_game(M=4, steps=20000, ent=0.0, greedy=True)
    print(f"  PPO IFD slope={sl:.3f} (target ~1 with low ent + greedy eval)")
    print(f"\ntest done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="test", choices=["test", "full"])
    a = ap.parse_args()
    if a.cmd == "test":
        cmd_test()
    else:
        cmd_full()
