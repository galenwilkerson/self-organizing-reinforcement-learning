#!/usr/bin/env python3
"""
Exp D -- stronger-learner comparison (EXPERIMENT_DESIGN_D.md). Does the swarm's embodied advantage survive a
HIGHER-CAPACITY learner? Honest either way.

Reuses the EXACT B2 embodied env (gen_b2_embodied.py) unchanged; only the learner changes. Same env, action
set {explore, approach, stay}, movement budget, oracle ceiling (min(N, sum cap) per tick). Arms:
  swarm   -- fixed recruitment+blocking (reward-free; ~55% of ceiling at T=600).
  simple  -- independent per-agent tabular softmax REINFORCE, coarse 12-state obs (B2 baseline; ~34%).
  mlp     -- shared-weight MLP actor over a RICHER continuous local obs (9 features), REINFORCE + per-agent
             EMA baseline.
  ctde    -- same shared MLP actor + a CENTRALISED critic V(global patch-level state) as the advantage
             baseline (MAPPO/COMA-lite); decentralised execution. The real "more capacity" test.

Metric: fraction of the oracle ceiling (cumulative over horizon AND over the last 30% = trained performance)
+ cumulative intake. B>=32, bootstrap CIs. CPU (small MLPs over 200 agents). Light lr/entropy/hidden sweep
reported. Data -> /mnt/newssd/.../paper2_stronger/.
"""
from __future__ import annotations
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch
import torch.nn as nn
from gen_b2_embodied import (Env, swarm_action, Learner, slope_pooled, N, C, S_SRC, R_NEIGH, CLUMP_RADIUS)

torch.set_num_threads(16)
DATA = "/mnt/newssd/flocking_ltm_data/paper2_stronger"
os.makedirs(DATA, exist_ok=True)
F_OBS = 9
F_GLOB = 2 * C + 2


# ---------------- combined obs + richer features (pair distances computed ONCE) ----------------
def obs_and_features(env, intake_ema):
    """Replicates Env.observe()'s signals AND builds the richer continuous features from the SAME pair
    distances (no double O(N^2) work). Returns on_food, and the (R,N,F_OBS) feature tensor."""
    dsrc = env._pair(env.pos, env.sources)                # (R,N,K)
    live = env.cap > 0
    within = dsrc <= S_SRC
    on_food = (within & live[:, None, :]).any(2)
    sees = within.any(2)
    dnb = env._pair(env.pos, env.pos); di = np.arange(N); dnb[:, di, di] = np.inf
    adj = dnb <= R_NEIGH; deg = adj.sum(2).clip(min=1)
    nfrac = (adj & on_food[:, None, :]).sum(2) / deg
    # nearest live source + direction
    dlive = np.where(live[:, None, :], dsrc, np.inf)
    nsrc = dlive.argmin(2); dmin = np.take_along_axis(dlive, nsrc[:, :, None], 2)[:, :, 0]
    src_tgt = np.take_along_axis(env.sources, nsrc[:, :, None], 1)
    src_avail = np.isfinite(dmin)
    nb_inf = np.where(on_food[:, None, :], dnb, np.inf)
    nnb = nb_inf.argmin(2); nb_tgt = np.take_along_axis(env.pos, nnb[:, :, None], 1)
    nb_avail = nb_inf.min(2) < np.inf
    target = np.where(src_avail[:, :, None], src_tgt, nb_tgt)
    has_target = src_avail | nb_avail
    env._target = target; env._has_target = has_target     # so env movement uses same target
    prox = np.where(src_avail, np.exp(-np.clip(dmin, 0, 5 * S_SRC) / S_SRC), 0.0)
    tdir = env._mi(target - env.pos); tn = np.linalg.norm(tdir, axis=-1, keepdims=True)
    udir = np.where(has_target[:, :, None], tdir / np.clip(tn, 1e-9, None), 0.0)
    dens = (within & live[:, None, :]).sum(2) / 5.0
    deg_n = deg / 20.0
    feats = np.stack([on_food.astype(np.float32), sees.astype(np.float32), prox.astype(np.float32),
                      udir[:, :, 0], udir[:, :, 1], dens, nfrac, deg_n, intake_ema], axis=-1)
    return on_food, feats.astype(np.float32)


def global_features(env):
    occ = env.occupancy()                                 # (R,C) committed
    occf = occ / max(1.0, N)
    richf = env.rich / env.rich.sum(1, keepdims=True).clip(min=1e-9)
    totcap = (env.cap.sum(1) / N)[:, None]
    g = np.concatenate([occf, richf, totcap, np.zeros((env.R, 1))], axis=1)
    return g.astype(np.float32)                           # (R, F_GLOB)


class MLP(nn.Module):
    def __init__(self, fin, fout, h=48):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(fin, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, fout))

    def forward(self, x):
        return self.net(x)


def run_swarm(T, R, tau=None, seed=0):
    env = Env(R, seed=seed, regime=1)
    return _loop(env, T, tau, controller="swarm")


def run_simple(T, R, tau=None, seed=0, lr=0.3):
    env = Env(R, seed=seed, regime=1)
    learner = Learner(R, lr=lr, seed=seed + 100)
    return _loop(env, T, tau, controller="simple", simple=learner)


def _loop(env, T, tau, controller, simple=None):
    R = env.R
    real = np.zeros((T, R)); ceil = np.zeros((T, R))
    occ_tail = np.zeros((R, C)); nm = 0
    for t in range(T):
        if tau is not None and t > 0 and t % tau == 0:
            env.relocate()
        obs, on, sees, nf = env.observe()
        if controller == "swarm":
            a = swarm_action(obs, on, sees, nf)
        else:
            a = simple.act(obs)
        capb = env.cap.sum(1)
        intake = env.step(a, on)
        if controller == "simple":
            simple.update(intake)
        real[t] = intake.sum(1); ceil[t] = np.minimum(N, capb)
        if t >= int(0.7 * T):
            occ_tail += env.occupancy(); nm += 1
    return dict(real=real, ceil=ceil, occ=occ_tail / max(nm, 1), rich=env.rich)


def run_mlp(T, R, tau=None, seed=0, lr=1e-2, ent=0.01, h=48, use_critic=False, crit_lr=1e-2):
    torch.manual_seed(seed + 7)
    env = Env(R, seed=seed, regime=1)
    actor = MLP(F_OBS, 3, h)
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    if use_critic:
        critic = MLP(F_GLOB, 1, h)
        copt = torch.optim.Adam(critic.parameters(), lr=crit_lr)
    base = np.zeros((R, N))                                # per-agent EMA baseline (non-critic)
    intake_ema = np.zeros((R, N))
    real = np.zeros((T, R)); ceil = np.zeros((T, R))
    occ_tail = np.zeros((R, C)); nm = 0
    for t in range(T):
        if tau is not None and t > 0 and t % tau == 0:
            env.relocate()
        on, feats = obs_and_features(env, intake_ema)                # (R,N,F), one pair-dist pass
        x = torch.from_numpy(feats.reshape(R * N, F_OBS))
        logits = actor(x)
        dist = torch.distributions.Categorical(logits=logits)
        a_t = dist.sample()
        logp = dist.log_prob(a_t)
        ent_t = dist.entropy()
        a = a_t.numpy().reshape(R, N)
        capb = env.cap.sum(1)
        intake = env.step(a, on)
        r = intake.astype(np.float32)                                # (R,N) reward
        intake_ema = 0.9 * intake_ema + 0.1 * intake
        # advantage
        if use_critic:
            g = torch.from_numpy(global_features(env))               # (R,F_GLOB)
            V = critic(g).squeeze(-1)                                 # (R,)
            Vexp = V.detach().numpy()[:, None]                        # (R,1) baseline per replica
            adv = r - Vexp
        else:
            adv = r - base
            base = 0.98 * base + 0.02 * intake
        adv = adv / (adv.std(axis=1, keepdims=True) + 1e-6)
        adv_t = torch.from_numpy(adv.reshape(R * N).astype(np.float32))
        loss = -(adv_t * logp).mean() - ent * ent_t.mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if use_critic:
            target = torch.from_numpy(r.mean(axis=1).astype(np.float32))   # per-replica mean reward
            closs = ((V - target) ** 2).mean()
            copt.zero_grad(); closs.backward(); copt.step()
        real[t] = intake.sum(1); ceil[t] = np.minimum(N, capb)
        if t >= int(0.7 * T):
            occ_tail += env.occupancy(); nm += 1
    return dict(real=real, ceil=ceil, occ=occ_tail / max(nm, 1), rich=env.rich)


def frac_metrics(d):
    real, ceil = d["real"], d["ceil"]
    T = real.shape[0]; l0 = int(0.7 * T)
    cum = real.sum(0) / ceil.sum(0).clip(min=1e-9)                    # (R,) cumulative fraction
    late = real[l0:].sum(0) / ceil[l0:].clip(min=1e-9).sum(0)         # (R,) trained-performance fraction
    return cum, late


def boot(v, draws):
    return v.mean(), np.percentile(v[draws].mean(1), 2.5), np.percentile(v[draws].mean(1), 97.5)


def main(R=32, T=1500):
    cache = os.path.join(DATA, f"stronger_R{R}_T{T}.npz")
    if os.path.exists(cache) and os.environ.get("FORCE", "") != "1":
        print(f"loading {cache}"); res = dict(np.load(cache, allow_pickle=True))["res"].item()
    else:
        t0 = time.time(); res = {}
        print(f"running arms (R={R}, T={T}) ...", flush=True)
        # Hyperparameters: lr=3e-3, h=48. ent=0.01 is the STABLE, reliable setting (seed std ~0.015):
        # the entropy-free (ent=0) config can spike toward the swarm on a lucky seed but COLLAPSES on most
        # seeds (MLP 0.06-0.59, CTDE 0.08-0.61 across 4 seeds) -> not reliable; reported in RESULTS_D.md.
        MLP_LR, MLP_ENT, MLP_H = 3e-3, 0.01, 48
        res["swarm"] = run_swarm(T, R); print(f"  swarm   ({time.time()-t0:.0f}s)", flush=True)
        res["simple"] = run_simple(T, R); print(f"  simple  ({time.time()-t0:.0f}s)", flush=True)
        res["mlp"] = run_mlp(T, R, lr=MLP_LR, ent=MLP_ENT, h=MLP_H, use_critic=False)
        print(f"  mlp     ({time.time()-t0:.0f}s)", flush=True)
        res["ctde"] = run_mlp(T, R, lr=MLP_LR, ent=MLP_ENT, h=MLP_H, use_critic=True)
        print(f"  ctde    ({time.time()-t0:.0f}s)", flush=True)
        np.savez(cache, res=res); print(f"wrote {cache} ({time.time()-t0:.0f}s)")

    rng = np.random.default_rng(0); draws = rng.integers(0, R, size=(3000, R))
    print(f"\n=== Exp D stronger-learner (R={R}, T={T}) ===")
    print("arm       cum-frac [95% CI]        late-frac [95% CI]      cum-intake   IFD-slope")
    summ = {}
    for arm in ["swarm", "simple", "mlp", "ctde"]:
        d = res[arm]; cum, late = frac_metrics(d)
        cm, clo, chi = boot(cum, draws); lm, llo, lhi = boot(late, draws)
        ci = d["real"].sum(0).mean(); sl = slope_pooled(d["rich"], d["occ"])
        summ[arm] = dict(cum=cm, clo=clo, chi=chi, late=lm, llo=llo, lhi=lhi, intake=ci, slope=sl)
        print(f"{arm:8s}  {cm:.3f} [{clo:.3f},{chi:.3f}]   {lm:.3f} [{llo:.3f},{lhi:.3f}]   "
              f"{ci:8.0f}   {sl:.3f}")
    print(f"\n  swarm/best-learner (late-frac): swarm {summ['swarm']['late']:.3f} vs "
          f"max({summ['simple']['late']:.3f},{summ['mlp']['late']:.3f},{summ['ctde']['late']:.3f})")
    make_figure(summ, R, T)
    return summ


def make_figure(summ, R, T):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "legend.fontsize": 10})
    arms = ["swarm", "simple", "mlp", "ctde"]
    labels = ["swarm\n(reward-free)", "simple\nREINFORCE", "richer-obs\nMLP", "centralised-critic\nCTDE"]
    cols = ["#2ca02c", "#9ecae1", "#4292c6", "#08519c"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(len(arms))
    for k, (metric, lo, hi, title) in enumerate([
            ("cum", "clo", "chi", "(a) cumulative fraction of oracle ceiling"),
            ("late", "llo", "lhi", "(b) trained (last 30%) fraction of ceiling")]):
        vals = [summ[a][metric] for a in arms]
        el = [summ[a][metric] - summ[a][lo] for a in arms]
        eh = [summ[a][hi] - summ[a][metric] for a in arms]
        ax[k].bar(x, vals, yerr=[el, eh], capsize=4, color=cols)
        ax[k].axhline(summ["swarm"][metric], color="#2ca02c", ls="--", lw=1, alpha=0.7)
        ax[k].axhline(1.0, color="k", ls=":", lw=1, label="oracle ceiling")
        ax[k].set_xticks(x); ax[k].set_xticklabels(labels, fontsize=9)
        ax[k].set_ylabel("fraction of oracle ceiling"); ax[k].set_ylim(0, 1.02)
        ax[k].set_title(title, fontsize=10.8)
        for i, a in enumerate(arms):
            ax[k].text(i, summ[a][metric] + 0.02, f"{summ[a][metric]:.2f}", ha="center", fontsize=9)
    # annotate the entropy-free INSTABILITY on the trained-fraction panel (from the 4-seed sweep)
    unstable = {"mlp": (0.057, 0.590, 0.287), "ctde": (0.075, 0.613, 0.222)}  # (min,max,mean) ent=0
    for a, (lo, hi, mn) in unstable.items():
        i = arms.index(a)
        ax[1].plot([i + 0.30, i + 0.30], [lo, hi], color="#d62728", lw=1.5, zorder=7)
        ax[1].plot([i + 0.22, i + 0.38], [mn, mn], color="#d62728", lw=1.5, zorder=7)
        for y in (lo, hi):
            ax[1].plot([i + 0.26, i + 0.34], [y, y], color="#d62728", lw=1.2, zorder=7)
    ax[1].plot([], [], color="#d62728", lw=1.5, label="entropy-free: unstable\n(4-seed range; mostly collapses)")
    ax[1].legend(fontsize=8, loc="upper right")
    ax[0].legend(fontsize=9)
    fig.suptitle(f"Exp D — does the swarm's embodied edge survive a higher-capacity learner? (R={R}, T={T})",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("pdf", "png"):
        p = os.path.join(HERE, f"fig_stronger.{ext}")
        fig.savefig(p, dpi=200 if ext == "png" else None); print("wrote", p)


if __name__ == "__main__":
    import sys as _s
    if len(_s.argv) > 1 and _s.argv[1] == "smoke":
        for arm, fn in [("swarm", lambda: run_swarm(400, 8)), ("simple", lambda: run_simple(400, 8)),
                        ("mlp", lambda: run_mlp(400, 8)), ("ctde", lambda: run_mlp(400, 8, use_critic=True))]:
            d = fn(); cum, late = frac_metrics(d)
            print(f"{arm}: cum-frac {cum.mean():.3f} late-frac {late.mean():.3f} slope {slope_pooled(d['rich'],d['occ']):.3f}")
    else:
        main()
