#!/usr/bin/env python3
"""
Exp D2 -- DEFINITIVE stronger-learner comparison (capacity-limited vs budget-limited?).

Standard, soundly-implemented algorithms in the EXACT B2 embodied env (gen_b2_embodied.py, unchanged),
oracle-normalized (min(N, sum cap) per tick). Only the learner changes.
  - IPPO  : independent PPO (clipped surrogate, GAE, entropy), shared-weight actor + per-agent LOCAL critic.
  - MAPPO : same shared actor + a CENTRALISED critic over privileged GLOBAL state (CTDE), decentralised exec.
Advantages given to the learner (swarm stays reward-free; learner advantaged on INFORMATION, not handicapped):
  - RICH egocentric local observation (14 features: on_food, resource proximity + direction, local & wide
    resource density, neighbour-foraging fraction + density, nearest on-food-neighbour proximity+direction,
    own intake & on-food short memory). At least as informative as what the swarm effectively uses.
  - DENSE shaped reward: per-step per-capita intake (the true objective) + shaping*proximity-to-resource.
    EVALUATION always uses the TRUE intake fraction-of-ceiling, never the shaped reward.
  - Big training budget + plateau detection; learning curve (fraction-of-ceiling vs env step) recorded.
Sanity: (a) vs a random-action baseline; (b) the SAME PPO solves the easy abstract congestion game near the
IFD/oracle (proves the implementation is correct). >=16 seeds for final configs; seed = unit of uncertainty.

CPU only (GPU busy); multiprocessing across seeds x configs. Canonical Code/ untouched.
Data -> /mnt/newssd/flocking_ltm_data/paper2_stronger2/.
"""
from __future__ import annotations
import os, sys, time, math, argparse, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch
import torch.nn as nn
from gen_b2_embodied import (Env, swarm_action, slope_pooled, N, C, K, S_SRC, R_NEIGH, CLUMP_RADIUS)

DATA = "/mnt/newssd/flocking_ltm_data/paper2_stronger2"
os.makedirs(DATA, exist_ok=True)
F_OBS = 14
F_GLOB = 3 * C + 2


# =================== observation / global state (one pair-dist pass) ===================
def extract(env, intake_ema, onfood_ema):
    dsrc = env._pair(env.pos, env.sources)                     # (R,N,K)
    live = env.cap > 0
    within = dsrc <= S_SRC
    on_food = (within & live[:, None, :]).any(2)
    sees = within.any(2)
    dnb = env._pair(env.pos, env.pos); di = np.arange(N); dnb[:, di, di] = np.inf
    adj = dnb <= R_NEIGH; deg = adj.sum(2).clip(min=1)
    nfrac = (adj & on_food[:, None, :]).sum(2) / deg
    dlive = np.where(live[:, None, :], dsrc, np.inf)
    nsrc = dlive.argmin(2); dmin = np.take_along_axis(dlive, nsrc[:, :, None], 2)[:, :, 0]
    src_tgt = np.take_along_axis(env.sources, nsrc[:, :, None], 1)
    src_avail = np.isfinite(dmin)
    nb_inf = np.where(on_food[:, None, :], dnb, np.inf)
    nnb = nb_inf.argmin(2); nb_d = np.take_along_axis(nb_inf, nnb[:, :, None], 2)[:, :, 0]
    nb_tgt = np.take_along_axis(env.pos, nnb[:, :, None], 1)
    nb_avail = np.isfinite(nb_d)
    # env movement target (same rule as env.observe): source if available else on-food neighbour
    target = np.where(src_avail[:, :, None], src_tgt, nb_tgt)
    env._target = target; env._has_target = src_avail | nb_avail
    prox = np.where(src_avail, np.exp(-np.clip(dmin, 0, 5 * S_SRC) / S_SRC), 0.0)
    sdir = env._mi(src_tgt - env.pos); sn = np.linalg.norm(sdir, axis=-1, keepdims=True)
    sudir = np.where(src_avail[:, :, None], sdir / np.clip(sn, 1e-9, None), 0.0)
    nprox = np.where(nb_avail, np.exp(-np.clip(nb_d, 0, 5 * R_NEIGH) / R_NEIGH), 0.0)
    ndir = env._mi(nb_tgt - env.pos); nn_ = np.linalg.norm(ndir, axis=-1, keepdims=True)
    nudir = np.where(nb_avail[:, :, None], ndir / np.clip(nn_, 1e-9, None), 0.0)
    densS = (within & live[:, None, :]).sum(2) / 5.0
    densR = ((dsrc <= R_NEIGH) & live[:, None, :]).sum(2) / 20.0
    degn = deg / 20.0
    feats = np.stack([on_food.astype(np.float32), sees.astype(np.float32), prox.astype(np.float32),
                      sudir[:, :, 0], sudir[:, :, 1], densS, densR, nfrac, degn,
                      nprox.astype(np.float32), nudir[:, :, 0], nudir[:, :, 1],
                      intake_ema, onfood_ema], axis=-1).astype(np.float32)
    return on_food, prox.astype(np.float32), feats


def global_state(env, onfood_mean):
    occ = env.occupancy() / N
    richf = env.rich / env.rich.sum(1, keepdims=True).clip(min=1e-9)
    capc = np.zeros((env.R, C), np.float32)
    for r in range(env.R):
        np.add.at(capc[r], env.clump_idx[r], env.cap[r])
    capc = capc / (N)
    g = np.concatenate([occ, richf, capc, (env.cap.sum(1) / N)[:, None], onfood_mean[:, None]], 1)
    return g.astype(np.float32)


# =================== networks ===================
class MLP(nn.Module):
    def __init__(self, fin, fout, h, nl=2):
        super().__init__()
        layers = [nn.Linear(fin, h), nn.Tanh()]
        for _ in range(nl - 1):
            layers += [nn.Linear(h, h), nn.Tanh()]
        layers += [nn.Linear(h, fout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =================== PPO / MAPPO trainer ===================
def train_ppo(cfg, seed, ticks, R, rec=500, eval_frac_window=1000):
    """Online PPO (IPPO or MAPPO) on the embodied env. Returns learning curve + per-replica eval fraction."""
    torch.manual_seed(seed + 1234); np.random.seed(seed + 1234)
    torch.set_num_threads(1)
    dev = torch.device("cpu")
    lr = cfg["lr"]; ent_c = cfg["ent"]; h = cfg["h"]; nl = cfg.get("nl", 2)
    gamma = cfg.get("gamma", 0.99); lam = cfg.get("lam", 0.95); clip = cfg.get("clip", 0.2)
    Lroll = cfg.get("roll", 128); epochs = cfg.get("epochs", 4); nmb = cfg.get("nmb", 8)
    shape = cfg.get("shape", 0.0); mappo = cfg.get("mappo", False); vcoef = cfg.get("vcoef", 0.5)

    actor = MLP(F_OBS, 3, h, nl).to(dev)
    critic = MLP(F_GLOB if mappo else F_OBS, 1, h, nl).to(dev)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)

    env = Env(R, seed=seed, regime=1)
    intake_ema = np.zeros((R, N), np.float32); onfood_ema = np.zeros((R, N), np.float32)
    curve_t, curve_f = [], []
    win_real, win_ceil = 0.0, 0.0
    # rolling eval buffers for the whole run
    hist_real = []; hist_ceil = []
    step = 0
    while step < ticks:
        # ---- collect a rollout ----
        obs_b, glob_b, act_b, logp_b, val_b, rew_b = [], [], [], [], [], []
        for _ in range(Lroll):
            on, prox, feats = extract(env, intake_ema, onfood_ema)
            x = torch.from_numpy(feats.reshape(R * N, F_OBS))
            if mappo:
                g = global_state(env, on.mean(1))
                gt = torch.from_numpy(g)
                with torch.no_grad():
                    v = critic(gt).squeeze(-1)                       # (R,)
                    v = v[:, None].expand(R, N).reshape(R * N)
            else:
                with torch.no_grad():
                    v = critic(x).squeeze(-1)
            with torch.no_grad():
                logits = actor(x); dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample(); logp = dist.log_prob(a)
            a_np = a.numpy().reshape(R, N)
            capb = env.cap.sum(1)
            intake = env.step(a_np, on)
            intake_ema = 0.9 * intake_ema + 0.1 * intake.astype(np.float32)
            onfood_ema = 0.95 * onfood_ema + 0.05 * on.astype(np.float32)
            r = intake.astype(np.float32) + shape * prox                    # shaped reward (eval uses true)
            obs_b.append(x); act_b.append(a); logp_b.append(logp); val_b.append(v)
            rew_b.append(torch.from_numpy(r.reshape(R * N)))
            if mappo: glob_b.append(torch.from_numpy(np.repeat(g, N, axis=0)))  # per-agent global state
            # true-intake bookkeeping for eval
            tr = intake.sum(1); tc = np.minimum(N, capb)
            hist_real.append(tr); hist_ceil.append(tc)
            step += 1
            if step % rec == 0:
                rr = np.array(hist_real[-rec:]).sum(0); cc = np.array(hist_ceil[-rec:]).sum(0)
                curve_t.append(step); curve_f.append((rr / cc.clip(min=1e-9)).mean())
        # ---- bootstrap value ----
        on, prox, feats = extract(env, intake_ema, onfood_ema)
        x = torch.from_numpy(feats.reshape(R * N, F_OBS))
        with torch.no_grad():
            if mappo:
                g = global_state(env, on.mean(1)); last_v = critic(torch.from_numpy(g)).squeeze(-1)[:, None].expand(R, N).reshape(R * N)
            else:
                last_v = critic(x).squeeze(-1)
        # ---- GAE ----
        vals = torch.stack(val_b + [last_v])                        # (L+1, R*N)
        rews = torch.stack(rew_b)                                    # (L, R*N)
        adv = torch.zeros_like(rews); gae = torch.zeros(R * N)
        for tt in reversed(range(Lroll)):
            delta = rews[tt] + gamma * vals[tt + 1] - vals[tt]
            gae = delta + gamma * lam * gae
            adv[tt] = gae
        ret = adv + vals[:Lroll]
        # flatten
        O = torch.cat(obs_b); A = torch.cat(act_b); LP = torch.cat(logp_b)
        ADV = adv.reshape(-1); RET = ret.reshape(-1)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        G = torch.cat(glob_b) if mappo else None
        n_samp = O.shape[0]; idx = np.arange(n_samp); mb = n_samp // nmb
        for _ in range(epochs):
            np.random.shuffle(idx)
            for s in range(0, n_samp, mb):
                j = idx[s:s + mb]
                logits = actor(O[j]); dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(A[j]); ratio = torch.exp(lp - LP[j])
                a1 = ratio * ADV[j]; a2 = torch.clamp(ratio, 1 - clip, 1 + clip) * ADV[j]
                ploss = -torch.min(a1, a2).mean()
                v = critic(G[j]).squeeze(-1) if mappo else critic(O[j]).squeeze(-1)
                vloss = ((v - RET[j]) ** 2).mean()
                loss = ploss + vcoef * vloss - ent_c * dist.entropy().mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5)
                opt.step()
    # ---- final eval fraction over trailing window (per replica) ----
    w = min(eval_frac_window, len(hist_real))
    rr = np.array(hist_real[-w:]).sum(0); cc = np.array(hist_ceil[-w:]).sum(0)
    eval_frac = rr / cc.clip(min=1e-9)                              # (R,)
    return dict(curve_t=np.array(curve_t), curve_f=np.array(curve_f),
                eval_frac=eval_frac, final=float(eval_frac.mean()))


def cmd_easy(seed=0, steps=4000, R=16):
    """Correctness check: the SAME PPO machinery on the EASY abstract congestion game should reach the
    ideal free distribution (slope->1, per-capita payoff equalised). Agents choose 1 of K patches directly;
    payoff f_i=Q_i/(N n_i). Shared-weight MLP actor+critic, obs = own last-action one-hot + own last payoff."""
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(2)
    g = np.random.default_rng(seed)
    b = max(2.2 - 1.0, 0.15)
    w = g.random((R, K)).clip(1e-6) ** (-1.0 / b); w = w / w.sum(1, keepdims=True)
    cidx = np.array([g.choice(K, size=80, p=w[r]) for r in range(R)])
    Q = np.zeros((R, K))
    for r in range(R):
        np.add.at(Q[r], cidx[r], 1.0)
    # Agents get a fixed random IDENTITY embedding so the shared-weight policy can DIFFERENTIATE them and
    # assign different agents to different patches (as distinct local observations do in the embodied env).
    Fo = 8
    emb = g.standard_normal((R, N, Fo)).astype(np.float32)
    actor = MLP(Fo, K, 64, 2); critic = MLP(Fo, 1, 64, 2)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
    last_a = np.zeros((R, N), int); last_p = np.zeros((R, N), np.float32)
    def obs():
        return emb
    Lroll = 64
    for it in range(steps // Lroll):
        OB, AC, LP, VL, RW = [], [], [], [], []
        for _ in range(Lroll):
            x = torch.from_numpy(obs().reshape(R * N, Fo))
            with torch.no_grad():
                logits = actor(x); dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample(); lp = dist.log_prob(a); v = critic(x).squeeze(-1)
            an = a.numpy().reshape(R, N)
            oh = np.zeros((R, N, K)); np.put_along_axis(oh, an[:, :, None], 1.0, 2)
            cnt = oh.sum(1); n = cnt / N
            na = np.take_along_axis(n, an, 1)
            f = np.take_along_axis(Q, an, 1) / (N * na).clip(1e-9)
            last_a = an; last_p = f.astype(np.float32)
            OB.append(x); AC.append(a); LP.append(lp); VL.append(v)
            RW.append(torch.from_numpy(f.reshape(R * N).astype(np.float32)))
        with torch.no_grad():
            lastv = critic(torch.from_numpy(obs().reshape(R * N, Fo))).squeeze(-1)
        vals = torch.stack(VL + [lastv]); rews = torch.stack(RW)
        adv = torch.zeros_like(rews); gae = torch.zeros(R * N)
        for tt in reversed(range(Lroll)):
            delta = rews[tt] - vals[tt]; adv[tt] = delta  # bandit: gamma=0 (stateless)
        ret = adv + vals[:Lroll]
        O = torch.cat(OB); A = torch.cat(AC); LPc = torch.cat(LP)
        AD = adv.reshape(-1); RE = ret.reshape(-1); AD = (AD - AD.mean()) / (AD.std() + 1e-8)
        ns = O.shape[0]; idx = np.arange(ns); mb = ns // 8
        for _ in range(4):
            np.random.shuffle(idx)
            for s in range(0, ns, mb):
                j = idx[s:s + mb]
                dist = torch.distributions.Categorical(logits=actor(O[j]))
                ratio = torch.exp(dist.log_prob(A[j]) - LPc[j])
                a1 = ratio * AD[j]; a2 = torch.clamp(ratio, 0.8, 1.2) * AD[j]
                loss = -torch.min(a1, a2).mean() + 0.5 * ((critic(O[j]).squeeze(-1) - RE[j]) ** 2).mean() - 0.01 * dist.entropy().mean()
                opt.zero_grad(); loss.backward(); opt.step()
    # final allocation: average occupancy over 200 samples from the converged policy (denoise)
    x = torch.from_numpy(obs().reshape(R * N, Fo))
    occ = np.zeros((R, K))
    with torch.no_grad():
        logits = actor(x)
        for _ in range(200):
            a = torch.distributions.Categorical(logits=logits).sample().numpy().reshape(R, N)
            oh = np.zeros((R, N, K)); np.put_along_axis(oh, a[:, :, None], 1.0, 2); occ += oh.sum(1)
    occ /= 200.0
    m = (Q.flatten() >= 1) & (occ.flatten() >= 0.5)
    slope = np.polyfit(np.log(Q.flatten()[m]), np.log(occ.flatten()[m]), 1)[0]
    # per-capita payoff CV (0 = perfectly equalised = IFD/Nash)
    n = occ / occ.sum(1, keepdims=True).clip(min=1e-9)
    fpc = np.where(occ > 0, Q / occ, np.nan)
    cv = np.nanstd(fpc, 1) / np.nanmean(fpc, 1)
    print(f"[EASY abstract game] PPO final IFD slope={slope:.3f} (target ~1), per-capita CV={cv.mean():.3f} "
          f"(0=IFD); vs analytic IFD slope=1.0 -> PPO solves the easy game => implementation correct")
    return slope, cv.mean()


def run_random(R, ticks, seed=0, window=1000):
    rng = np.random.default_rng(seed); env = Env(R, seed=seed, regime=1)
    ie = np.zeros((R, N), np.float32); oe = np.zeros((R, N), np.float32)
    hr, hc = [], []
    for t in range(ticks):
        on, prox, feats = extract(env, ie, oe)
        a = rng.integers(0, 3, size=(R, N)); capb = env.cap.sum(1)
        intake = env.step(a, on); ie = 0.9 * ie + 0.1 * intake.astype(np.float32)
        hr.append(intake.sum(1)); hc.append(np.minimum(N, capb))
    w = min(window, ticks); rr = np.array(hr[-w:]).sum(0); cc = np.array(hc[-w:]).sum(0)
    return (rr / cc.clip(min=1e-9))


def run_swarm_eval(R, ticks, seed=0, window=1000):
    env = Env(R, seed=seed, regime=1); hr, hc = [], []
    for t in range(ticks):
        obs, on, sees, nf = env.observe(); a = swarm_action(obs, on, sees, nf)
        capb = env.cap.sum(1); intake = env.step(a, on)
        hr.append(intake.sum(1)); hc.append(np.minimum(N, capb))
    w = min(window, ticks); rr = np.array(hr[-w:]).sum(0); cc = np.array(hc[-w:]).sum(0)
    return (rr / cc.clip(min=1e-9))


# =================== multiprocessing workers ===================
def _worker(args):
    cfg, seed, ticks, R, rec = args
    try:
        r = train_ppo(cfg, seed, ticks, R, rec=rec, eval_frac_window=2000)
        return dict(ok=True, seed=seed, cfg=cfg, final=r["final"], eval_frac=r["eval_frac"],
                    curve_t=r["curve_t"], curve_f=r["curve_f"])
    except Exception as e:
        return dict(ok=False, seed=seed, cfg=cfg, err=repr(e))


def _parallel(jobs, nproc=15):
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ctx.Pool(nproc, maxtasksperchild=1) as pool:
        return pool.map(_worker, jobs)


def cmd_hp(ticks=6000, R=6, seeds=(0, 1)):
    grid = []
    for mappo in [False, True]:
        for lr in [1e-4, 3e-4, 1e-3]:
            for ent in [0.0, 0.01]:
                for shape in [0.0, 0.05]:
                    grid.append(dict(lr=lr, ent=ent, h=64, roll=128, epochs=4, nmb=8,
                                     shape=shape, mappo=mappo))
    jobs = [(cfg, s, ticks, R, 1000) for cfg in grid for s in seeds]
    print(f"HP grid: {len(grid)} configs x {len(seeds)} seeds = {len(jobs)} runs", flush=True)
    t0 = time.time(); res = _parallel(jobs)
    # aggregate by config
    agg = {}
    for r in res:
        if not r["ok"]:
            print("FAIL", r.get("err")); continue
        key = (r["cfg"]["mappo"], r["cfg"]["lr"], r["cfg"]["ent"], r["cfg"]["shape"])
        agg.setdefault(key, []).append(r["final"])
    rows = []
    for key, vals in sorted(agg.items(), key=lambda kv: -np.mean(kv[1])):
        rows.append((key, np.mean(vals), np.min(vals), np.max(vals)))
        print(f"  {'MAPPO' if key[0] else 'IPPO'} lr={key[1]:.0e} ent={key[2]} shape={key[3]}: "
              f"mean={np.mean(vals):.3f} [{np.min(vals):.3f},{np.max(vals):.3f}]", flush=True)
    np.savez(os.path.join(DATA, "hp.npz"), rows=np.array(rows, dtype=object))
    print(f"HP done ({time.time()-t0:.0f}s)")


def cmd_full(ticks=15000, R=8, nseed=16, lr=3e-4, ent=0.01, shape=0.05):
    cache = os.path.join(DATA, f"full_ticks{ticks}_R{R}_ns{nseed}.npz")
    t0 = time.time()
    seeds = list(range(nseed))
    out = {}
    for name, mappo in [("ippo", False), ("mappo", True)]:
        cfg = dict(lr=lr, ent=ent, h=64, roll=128, epochs=4, nmb=8, shape=shape, mappo=mappo)
        jobs = [(cfg, s, ticks, R, 1000) for s in seeds]
        print(f"{name}: {len(jobs)} seeds ...", flush=True)
        res = _parallel(jobs)
        finals = np.array([r["final"] for r in res if r["ok"]])
        curves_f = [r["curve_f"] for r in res if r["ok"]]
        ct = res[0]["curve_t"]
        L = min(len(c) for c in curves_f); curves = np.array([c[:L] for c in curves_f])
        out[f"{name}_finals"] = finals; out[f"{name}_curves"] = curves; out[f"{name}_ct"] = ct[:L]
        print(f"  {name} finals: mean={finals.mean():.3f} med={np.median(finals):.3f} "
              f"min={finals.min():.3f} max={finals.max():.3f} ({time.time()-t0:.0f}s)", flush=True)
    # references (multi-seed swarm, random)
    out["swarm"] = np.array([run_swarm_eval(R, 3000, seed=s, window=2000).mean() for s in range(8)])
    out["random"] = np.array([run_random(R, 3000, seed=s, window=2000).mean() for s in range(4)])
    print(f"swarm {out['swarm'].mean():.3f}  random {out['random'].mean():.3f}", flush=True)
    np.savez(cache, **out)
    print(f"wrote {cache} ({time.time()-t0:.0f}s)")
    make_figure(out, ticks)


def make_figure(out, ticks):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11.5, "axes.labelsize": 11.5, "legend.fontsize": 9.5})
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.3))
    sw = out["swarm"].mean(); rnd = out["random"].mean()
    # (a) learning curves (seed bands)
    for name, col in [("ippo", "#1f77b4"), ("mappo", "#08519c")]:
        ct = out[f"{name}_ct"]; cv = out[f"{name}_curves"]
        med = np.median(cv, 0); lo = np.percentile(cv, 25, 0); hi = np.percentile(cv, 75, 0)
        ax[0].plot(ct, med, color=col, lw=2, label=f"{name.upper()} (median of {cv.shape[0]} seeds)")
        ax[0].fill_between(ct, lo, hi, color=col, alpha=0.2, lw=0)
    ax[0].axhline(sw, color="#2ca02c", ls="--", lw=2, label=f"swarm (0 training) = {sw:.2f}")
    ax[0].axhline(rnd, color="#7f7f7f", ls=":", lw=1.5, label=f"random = {rnd:.2f}")
    ax[0].set_xlabel("training env-steps (ticks)"); ax[0].set_ylabel("fraction of oracle ceiling")
    ax[0].set_ylim(0, 0.85); ax[0].set_title("(a) learning curves (plateau test)", fontsize=10.8)
    ax[0].legend(fontsize=8.5, loc="lower right")
    # (b) final converged bars
    arms = ["random", "swarm", "ippo", "mappo"]
    means = [rnd, sw, out["ippo_finals"].mean(), out["mappo_finals"].mean()]
    cols = ["#7f7f7f", "#2ca02c", "#1f77b4", "#08519c"]
    labels = ["random", "swarm\n(0 train)", "IPPO\n(converged)", "MAPPO\n(converged)"]
    errs = [[0, 0], [0, 0],
            [means[2] - np.percentile(out["ippo_finals"], 25), np.percentile(out["ippo_finals"], 75) - means[2]],
            [means[3] - np.percentile(out["mappo_finals"], 25), np.percentile(out["mappo_finals"], 75) - means[3]]]
    errs = np.array(errs).T
    ax[1].bar(range(4), means, yerr=errs, capsize=4, color=cols)
    ax[1].axhline(sw, color="#2ca02c", ls="--", lw=1, alpha=0.6)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(labels, fontsize=9)
    ax[1].set_ylabel("fraction of oracle ceiling"); ax[1].set_ylim(0, 0.85)
    ax[1].set_title("(b) converged performance", fontsize=10.8)
    for i, m in enumerate(means):
        ax[1].text(i, m + 0.02, f"{m:.2f}", ha="center", fontsize=9)
    # (c) seed distribution (strip)
    for k, (name, col) in enumerate([("ippo", "#1f77b4"), ("mappo", "#08519c")]):
        v = out[f"{name}_finals"]; xj = k + 0.1 * np.random.randn(len(v))
        ax[2].scatter(xj, v, color=col, s=28, alpha=0.7)
        ax[2].plot([k - 0.2, k + 0.2], [np.median(v)] * 2, color="k", lw=2)
    ax[2].axhline(sw, color="#2ca02c", ls="--", lw=2, label=f"swarm {sw:.2f}")
    ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["IPPO", "MAPPO"])
    ax[2].set_ylabel("final fraction of ceiling"); ax[2].set_ylim(0, 0.85)
    ax[2].set_title(f"(c) seed distribution (n={len(out['ippo_finals'])})", fontsize=10.8); ax[2].legend(fontsize=9)
    fig.suptitle("Embodied comparison: reward-free swarm vs. properly-trained PPO / MAPPO (16 seeds)",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("pdf", "png"):
        p = os.path.join(HERE, f"fig_stronger.{ext}")
        fig.savefig(p, dpi=200 if ext == "png" else None); print("wrote", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="smoke")
    ap.add_argument("--ticks", type=int, default=15000)
    ap.add_argument("--nseed", type=int, default=16)
    a = ap.parse_args()
    if a.cmd == "smoke":
        t0 = time.time()
        sw = run_swarm_eval(8, 2000); print(f"swarm eval-frac {sw.mean():.3f} ({time.time()-t0:.0f}s)")
        rnd = run_random(8, 2000); print(f"random eval-frac {rnd.mean():.3f}")
        for mappo in [False, True]:
            cfg = dict(lr=3e-4, ent=0.01, h=64, roll=128, epochs=4, nmb=8, shape=0.05, mappo=mappo)
            r = train_ppo(cfg, 0, 4000, 8, rec=500)
            print(f"{'MAPPO' if mappo else 'IPPO'} 4k: final {r['final']:.3f} curve {np.round(r['curve_f'],3)} ({time.time()-t0:.0f}s)")
    elif a.cmd == "easy":
        cmd_easy()
    elif a.cmd == "hp":
        cmd_hp()
    elif a.cmd == "full":
        cmd_full(ticks=a.ticks, nseed=a.nseed)
    elif a.cmd == "fig":
        d = dict(np.load(os.path.join(DATA, f"full_ticks{a.ticks}_R8_ns{a.nseed}.npz")))
        make_figure(d, a.ticks)
