#!/usr/bin/env python3
"""
MARL baseline experiment for paper2_rl_v2 (design: MARL_EXPERIMENT_DESIGN.md).

Question: does the learner-free swarm reach the SAME congestion-game Nash / ideal-free optimum that a
TRAINED multi-agent RL learner reaches, on the SAME decision problem -- and where does that break?

Shared task = abstract CONGESTION ALLOCATION GAME:
  K patches, heavy-tailed richness Q_i (the swarm's own Pareto draws, clump_beta=2.2), N agents.
  Each step every agent picks a patch i; per-capita payoff f_i = Q_i/(N n_i)  (payoff falls with crowding).
  Optimum = ideal free distribution / Nash equilibrium: n_i proportional to Q_i (equalises f_i).

Methods:
  * Analytic IFD            -- ground-truth target n_i = Q_i/sum(Q). Reference line, not a learner.
  * Learner-free swarm      -- the spatial recruitment+blocking mechanism. Payoff-BLIND: its allocation
                               is fixed by its physics (input-matching ~ Q_i, slope 0.94). Reused from the
                               B=64 bridge cache on MATCHED (Q_i, N, K). Zero training.
  * Independent softmax-PG  -- PRIMARY learner. Each agent keeps a softmax policy over K arms, updates by
                               REINFORCE from its OWN per-capita payoff (local info, per-agent baseline).
  * Independent Q-learning  -- secondary recognisable MARL baseline (eps-greedy, own payoff).

Two regimes:
  Regime 1  CONGESTION    f_i = Q_i/(N n_i)     optimum = IFD (spread ~ Q)       -> expect swarm ~ learner
  Regime 2  NON-congestion f_i = Q_i (fixed)    optimum = concentrate on argmax Q -> swarm sub-optimal,
                                                                                     learner adapts
The swarm allocation is identical in both regimes (payoff-blind); only the optimality criterion changes.

Metrics: IFD matching slope ->1 ; distance-to-IFD (L1) ; Nash-gap / exploitability ; welfare efficiency
(achieved / optimal) for the regime; interactions-to-90%-of-final. >=16 replicas, bootstrap CIs. CPU.

Raw data cached to /mnt/newssd/flocking_ltm_data/paper2_marl/. Figures + this script in the paper repo.
"""
from __future__ import annotations
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/mnt/newssd/flocking_ltm_data/paper2_marl"
os.makedirs(DATA, exist_ok=True)
BRIDGE_CACHE = "/mnt/newssd/flocking_ltm_data/paper2_bridge/bridge_B64_T800_seed0.npz"

N = 200
K = 10


# ------------------------------------------------------------------ richness / allocations
def load_swarm_Q_and_alloc():
    """Q_i (R,K) = the swarm's own Pareto richness per replica; swarm allocation = steady-state
    committed-forager occupancy, normalised to sum 1 per replica."""
    z = np.load(BRIDGE_CACHE)
    rich = z["rich"].astype(float)                      # (R,K) sources per clump
    count = z["count"].astype(float)                    # (T,R,K) committed count
    occ = count[400:].mean(axis=0)                      # steady-state occupancy (R,K)
    alloc = occ / occ.sum(axis=1, keepdims=True).clip(min=1e-9)
    return rich, alloc


def ifd_alloc(Q):
    return Q / Q.sum(axis=1, keepdims=True).clip(min=1e-9)


# ------------------------------------------------------------------ metrics
def matching_slope_pooled(Q, n, boot_idx=None):
    """Regress log n_i on log Q_i pooled over (replica, patch); masks Q>=1 and n>0."""
    q = Q[boot_idx] if boot_idx is not None else Q
    a = n[boot_idx] if boot_idx is not None else n
    q = q.flatten(); a = a.flatten()
    m = (q >= 1) & (a > 1.0 / (N * 50))
    if m.sum() < 8:
        return np.nan
    return np.polyfit(np.log(q[m]), np.log(a[m]), 1)[0]


def l1_to_ifd(Q, n):
    """Mean over replicas of 0.5*sum|n_i - Q_i/sumQ| (total variation)."""
    t = ifd_alloc(Q)
    return 0.5 * np.abs(n - t).sum(axis=1)              # (R,)


def nash_gap(Q, n):
    """Exploitability: mean best-response improvement in per-capita payoff. ~0 at IFD/Nash.
    f_i = Q_i/(N n_i); deviating to i gives ~Q_i/(N n_i + 1). Returns (R,)."""
    x = n * N
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(x > 0, Q / x.clip(min=1e-9), 0.0)          # current per-capita (occupied)
    f_dev = Q / (x + 1.0)                                       # payoff if one more agent joins i
    best_dev = f_dev.max(axis=1, keepdims=True)                 # (R,1)
    imp = np.clip(best_dev - f, 0.0, None)                      # improvement from each agent's patch
    return (n * imp).sum(axis=1)                                # occupancy-weighted mean improvement


def welfare_efficiency(Q, n, regime):
    """Achieved total payoff / optimal total payoff under the regime. (R,)"""
    if regime == 1:      # congestion: welfare = sum over OCCUPIED Q_i ; optimum = all occupied
        w = np.where(n > 0, Q, 0.0).sum(axis=1)
        wopt = Q.sum(axis=1)
    else:                # fixed per-patch reward f_i=Q_i : welfare = N*sum n_i Q_i ; opt = N*max Q_i
        w = (n * Q).sum(axis=1)
        wopt = Q.max(axis=1)
    return w / wopt.clip(min=1e-9)


# ------------------------------------------------------------------ sanity test
def _sanity():
    rich, alloc = load_swarm_Q_and_alloc()
    R = rich.shape[0]
    print(f"loaded swarm richness + allocation: R={R} replicas, K={rich.shape[1]} patches, N={N}")
    print(f"Q_i per replica: mean sum={rich.sum(1).mean():.1f} (should ~ K_sources=80), "
          f"range {rich.min():.0f}..{rich.max():.0f}")
    # 1) IFD is the Nash: at n_i = Q_i/sumQ, per-capita f_i must be EQUAL across occupied patches
    t = ifd_alloc(rich)
    x = t * N
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(x > 0, rich / x, np.nan)
    cv = np.nanstd(f, axis=1) / np.nanmean(f, axis=1)
    print(f"[sanity] at analytic IFD: per-capita payoff f_i CV across patches = {np.nanmean(cv):.2e} "
          f"(should be ~0 => equalised => Nash)")
    print(f"[sanity] analytic IFD matching slope (pooled) = {matching_slope_pooled(rich, t):.3f} (==1)")
    print(f"[sanity] analytic IFD Nash-gap = {nash_gap(rich, t).mean():.2e} (should be ~0)")
    print(f"[sanity] analytic IFD L1-to-IFD = {l1_to_ifd(rich, t).mean():.2e} (==0)")
    # 2) swarm allocation
    print(f"[swarm]  matching slope (pooled, steady-state alloc) = {matching_slope_pooled(rich, alloc):.3f} "
          f"(paper reports ~0.94)")
    print(f"[swarm]  Nash-gap = {nash_gap(rich, alloc).mean():.3e} ; L1-to-IFD = {l1_to_ifd(rich, alloc).mean():.3f}")
    print(f"[swarm]  R1 welfare-eff = {welfare_efficiency(rich, alloc, 1).mean():.3f} ; "
          f"R2 welfare-eff = {welfare_efficiency(rich, alloc, 2).mean():.3f}")
    # 3) uniform allocation reference
    uni = np.full_like(rich, 1.0 / rich.shape[1])
    print(f"[uniform] slope={matching_slope_pooled(rich, uni):.3f} Nash-gap={nash_gap(rich, uni).mean():.3e} "
          f"R1-eff={welfare_efficiency(rich, uni, 1).mean():.3f} R2-eff={welfare_efficiency(rich, uni, 2).mean():.3f}")


# ------------------------------------------------------------------ learners (vectorised over replicas)
def _payoff(Q, a, regime):
    """Per-agent payoff for chosen arms a (R,N). R1 congestion: Q_i/(N n_i); R2 fixed: Q_i."""
    R, Narb = a.shape
    oh = np.zeros((R, Narb, Q.shape[1]))
    np.put_along_axis(oh, a[:, :, None], 1.0, axis=2)
    counts = oh.sum(axis=1)                                  # (R,K)
    n = counts / Narb
    Qa = np.take_along_axis(Q, a, axis=1)                   # (R,N) richness of chosen arm
    if regime == 1:
        na = np.take_along_axis(n, a, axis=1)              # (R,N) occupancy of chosen arm (incl self)
        r = Qa / (Narb * na).clip(min=1e-9)
    else:
        r = Qa.astype(float)
    return r, n


def _checkpoints(steps, every):
    cks = list(range(0, steps, every))
    if cks[-1] != steps - 1:
        cks.append(steps - 1)
    return np.array(cks)


def run_pg(Q, regime, steps=4000, lr=0.5, beta=0.02, ent_coef=0.0, every=25, seed=0):
    """Independent softmax policy-gradient (REINFORCE) learners; per-agent EMA baseline; per-step
    per-replica advantage normalisation (standard trick). Own-payoff info only.
    Returns checkpoints and alloc_traj (ncks,R,K)."""
    rng = np.random.default_rng(seed)
    R, Kk = Q.shape
    theta = np.zeros((R, N, Kk))
    base = np.zeros((R, N))                                  # per-agent reward EMA baseline (local)
    cks = _checkpoints(steps, every)
    ckset = set(cks.tolist())
    alloc_traj = []
    for s in range(steps):
        th = theta - theta.max(axis=2, keepdims=True)
        ex = np.exp(th); pi = ex / ex.sum(axis=2, keepdims=True)      # (R,N,K)
        g = -np.log(-np.log(rng.random((R, N, Kk)).clip(1e-12, 1)))   # Gumbel-max sampling
        a = (np.log(pi.clip(1e-12, 1)) + g).argmax(axis=2)           # (R,N)
        r, n = _payoff(Q, a, regime)
        adv = r - base
        base = (1 - beta) * base + beta * r
        adv_n = adv / (adv.std(axis=1, keepdims=True) + 1e-8)
        oh = np.zeros((R, N, Kk)); np.put_along_axis(oh, a[:, :, None], 1.0, axis=2)
        grad = adv_n[:, :, None] * (oh - pi)
        if ent_coef > 0:
            grad += ent_coef * (-(np.log(pi.clip(1e-12, 1)) + 1.0)) * pi
        theta += lr * grad
        if s in ckset:
            alloc_traj.append(n.copy())
    return cks, np.array(alloc_traj)


def run_ql(Q, regime, steps=4000, alpha=0.1, eps0=0.3, eps_decay=0.999, every=25, seed=0):
    """Independent Q-learning, eps-greedy, own-payoff bandit update. Secondary MARL baseline."""
    rng = np.random.default_rng(seed)
    R, Kk = Q.shape
    Qval = np.zeros((R, N, Kk))
    cks = _checkpoints(steps, every)
    ckset = set(cks.tolist())
    eps = eps0
    alloc_traj = []
    for s in range(steps):
        greedy = Qval.argmax(axis=2)
        rand = rng.integers(0, Kk, size=(R, N))
        explore = rng.random((R, N)) < eps
        a = np.where(explore, rand, greedy)
        r, n = _payoff(Q, a, regime)
        oh_r = np.take_along_axis(Qval, a[:, :, None], axis=2)[:, :, 0]
        newv = oh_r + alpha * (r - oh_r)
        np.put_along_axis(Qval, a[:, :, None], newv[:, :, None], axis=2)
        eps *= eps_decay
        if s in ckset:
            alloc_traj.append(n.copy())
    return cks, np.array(alloc_traj)


# ------------------------------------------------------------------ curve analysis + bootstrap
N_BOOT = 2000


def tv_to_opt(Q, n, regime):
    if regime == 1:
        target = ifd_alloc(Q)
    else:
        target = np.zeros_like(Q)
        target[np.arange(Q.shape[0]), Q.argmax(axis=1)] = 1.0
    return 0.5 * np.abs(n - target).sum(axis=1)            # (R,)


def boot_mean(valsR, draws):
    """mean + 95% CI over replicas given bootstrap replica-index draws (nboot,R)."""
    m = valsR.mean()
    bm = valsR[draws].mean(axis=1)
    return m, np.percentile(bm, 2.5), np.percentile(bm, 97.5)


def curves(alloc_traj, Q, regime, draws):
    """Per-checkpoint metric means + CIs. slope via pooled regression (bootstrap over replicas)."""
    nck = alloc_traj.shape[0]
    out = {k: np.zeros((nck, 3)) for k in ("slope", "nash", "l1", "weff", "opt")}
    for c in range(nck):
        n = alloc_traj[c]
        # pooled slope + bootstrap (resample replicas)
        s = matching_slope_pooled(Q, n)
        bs = np.array([matching_slope_pooled(Q, n, draws[b]) for b in range(0, N_BOOT, 4)])
        bs = bs[np.isfinite(bs)]
        out["slope"][c] = [s, np.percentile(bs, 2.5), np.percentile(bs, 97.5)]
        out["nash"][c] = boot_mean(nash_gap(Q, n), draws)
        out["l1"][c] = boot_mean(l1_to_ifd(Q, n), draws)
        out["weff"][c] = boot_mean(welfare_efficiency(Q, n, regime), draws)
        out["opt"][c] = boot_mean(1.0 - tv_to_opt(Q, n, regime), draws)
    return out


def tte(cks, curve_mean, frac=0.90, rising=True):
    """interactions (steps) to reach frac of final value (plateau = last 20%)."""
    plateau = curve_mean[-max(2, len(curve_mean) // 5):].mean()
    if rising:
        target = frac * plateau
        hit = curve_mean >= target
    else:
        start = curve_mean[0]
        target = start - frac * (start - plateau)
        hit = curve_mean <= target
    for i in range(len(cks)):
        if hit[i] and hit[i:].all():
            return int(cks[i]), plateau
    return None, plateau


def run_all(steps=4000, seed=0):
    cache = os.path.join(DATA, f"marl_results_steps{steps}_seed{seed}.npz")
    if os.path.exists(cache) and os.environ.get("FORCE", "") != "1":
        print(f"loading {cache}")
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    rich, swarm_alloc = load_swarm_Q_and_alloc()
    t0 = time.time()
    res = dict(rich=rich, swarm_alloc=swarm_alloc)
    for regime in (1, 2):
        ck_pg, pg = run_pg(rich, regime, steps=steps, lr=0.5, every=25, seed=seed)
        ck_ql, ql = run_ql(rich, regime, steps=steps, alpha=0.1, every=25, seed=seed)
        res[f"cks"] = ck_pg
        res[f"pg_r{regime}"] = pg
        res[f"ql_r{regime}"] = ql
        print(f"  regime {regime} done ({time.time()-t0:.0f}s)")
    np.savez_compressed(cache, **res)
    print(f"wrote {cache} ({time.time()-t0:.0f}s)")
    return res


def make_figures(res):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "legend.fontsize": 10,
                         "xtick.labelsize": 10.5, "ytick.labelsize": 10.5})
    Q = res["rich"]; swarm = res["swarm_alloc"]; cks = res["cks"]
    rng = np.random.default_rng(99)
    draws = rng.integers(0, Q.shape[0], size=(N_BOOT, Q.shape[0]))

    C_PG, C_QL, C_SW, C_AN = "#1f77b4", "#ff7f0e", "#2ca02c", "k"

    # ---- precompute curves ----
    pg1 = curves(res["pg_r1"], Q, 1, draws)
    ql1 = curves(res["ql_r1"], Q, 1, draws)
    # swarm scalars (zero training)
    sw_slope = matching_slope_pooled(Q, swarm)
    sw_nash = nash_gap(Q, swarm).mean()
    # analytic
    an_alloc = ifd_alloc(Q)

    # ======================= Figure 1: Regime-1 comparison vs interactions =======================
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    # (a) matching slope
    for cv, col, lab in [(pg1, C_PG, "softmax-PG learner"), (ql1, C_QL, "Q-learning")]:
        ax[0].plot(cks, cv["slope"][:, 0], color=col, lw=2, label=lab)
        ax[0].fill_between(cks, cv["slope"][:, 1], cv["slope"][:, 2], color=col, alpha=0.18, lw=0)
    ax[0].axhline(sw_slope, color=C_SW, ls="-", lw=2, label=f"swarm (0 training), slope={sw_slope:.2f}")
    ax[0].axhline(1.0, color=C_AN, ls="--", lw=1, label="analytic IFD / Nash (=1)")
    ax[0].set_xlabel("learning steps (rounds; each = N interactions)")
    ax[0].set_ylabel("IFD matching slope")
    ax[0].set_title("(a) quality: reaching the IFD", loc="left", fontsize=11.5, fontweight="bold")
    ax[0].set_ylim(-0.05, 1.25); ax[0].legend(loc="lower right", frameon=False)

    # (b) Nash-gap (exploitability), log y
    for cv, col, lab in [(pg1, C_PG, "softmax-PG learner"), (ql1, C_QL, "Q-learning")]:
        ax[1].plot(cks, cv["nash"][:, 0].clip(min=1e-4), color=col, lw=2, label=lab)
        ax[1].fill_between(cks, cv["nash"][:, 1].clip(min=1e-4), cv["nash"][:, 2].clip(min=1e-4),
                           color=col, alpha=0.18, lw=0)
    ax[1].axhline(sw_nash, color=C_SW, ls="-", lw=2, label=f"swarm (0 training) = {sw_nash:.2f}")
    ax[1].axhline(1e-3, color=C_AN, ls="--", lw=1, label="analytic Nash ($\\approx$0)")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("learning steps (rounds; each = N interactions)")
    ax[1].set_ylabel("Nash-gap / exploitability")
    ax[1].set_title("(b) optimality: distance from Nash", loc="left", fontsize=11.5, fontweight="bold")
    ax[1].legend(loc="upper right", frameon=False)
    fig.suptitle("Regime 1 — congestion game (optimum = ideal free distribution)", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        p = os.path.join(HERE, f"fig_marl_compare.{ext}")
        fig.savefig(p, dpi=200 if ext == "png" else None); print("wrote", p)

    # ======================= Figure 2: Regime-1 vs Regime-2 optimality summary =======================
    # final-step optimality (1 - TV to each regime's optimum) for each method, both regimes
    methods = []
    def final_opt(alloc_traj, regime):
        n = alloc_traj[-1]
        v = 1.0 - tv_to_opt(Q, n, regime)
        return boot_mean(v, draws)
    def scalar_opt(n, regime):
        return boot_mean(1.0 - tv_to_opt(Q, n, regime), draws)

    data = {
        "analytic\noptimum": {1: scalar_opt(an_alloc, 1), 2: scalar_opt(_onehot_argmax(Q), 2)},
        "softmax-PG\n(trained)": {1: final_opt(res["pg_r1"], 1), 2: final_opt(res["pg_r2"], 2)},
        "Q-learning\n(trained)": {1: final_opt(res["ql_r1"], 1), 2: final_opt(res["ql_r2"], 2)},
        "swarm\n(0 training)": {1: scalar_opt(swarm, 1), 2: scalar_opt(swarm, 2)},
    }
    fig2, ax2 = plt.subplots(figsize=(8.2, 4.6))
    labels = list(data.keys())
    xpos = np.arange(len(labels)); w = 0.38
    col1, col2 = "#2c7fb8", "#d95f0e"
    for j, regime in enumerate((1, 2)):
        vals = np.array([data[l][regime][0] for l in labels])
        lo = np.array([data[l][regime][1] for l in labels])
        hi = np.array([data[l][regime][2] for l in labels])
        ax2.bar(xpos + (j - 0.5) * w, vals, w,
                yerr=[vals - lo, hi - vals], capsize=3,
                color=(col1 if regime == 1 else col2),
                label=("Regime 1: congestion (opt = IFD / spread)" if regime == 1
                       else "Regime 2: fixed reward (opt = concentrate)"))
    ax2.set_xticks(xpos); ax2.set_xticklabels(labels)
    ax2.set_ylabel("closeness to regime optimum\n(1 $-$ total-variation distance)")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(1.0, color="k", lw=0.8, ls=":")
    ax2.set_title("Learning buys generality: the swarm is optimal only for its matched (congestion) structure",
                  fontsize=10.8)
    ax2.legend(loc="lower left", frameon=False, fontsize=9.5)
    fig2.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(HERE, f"fig_marl_regimes.{ext}")
        fig2.savefig(p, dpi=200 if ext == "png" else None); print("wrote", p)

    # ---- numeric report ----
    print("\n================ MARL RESULTS ================")
    tpg, plpg = tte(cks, pg1["slope"][:, 0], 0.90, rising=True)
    tnpg, _ = tte(cks, pg1["nash"][:, 0], 0.90, rising=False)
    print(f"[Regime 1] swarm: slope={sw_slope:.3f}, Nash-gap={sw_nash:.3f}, L1-to-IFD="
          f"{l1_to_ifd(Q, swarm).mean():.3f}  (ZERO training)")
    print(f"[Regime 1] softmax-PG final: slope={pg1['slope'][-1,0]:.3f}, Nash-gap={pg1['nash'][-1,0]:.3e}, "
          f"L1={pg1['l1'][-1,0]:.3f}; steps to 90% slope={tpg}, to 90% Nash-drop={tnpg}")
    print(f"[Regime 1] Q-learning final: slope={ql1['slope'][-1,0]:.3f}, Nash-gap={ql1['nash'][-1,0]:.3e}, "
          f"L1={ql1['l1'][-1,0]:.3f}")
    pg2 = curves(res["pg_r2"], Q, 2, draws); ql2 = curves(res["ql_r2"], Q, 2, draws)
    print(f"[Regime 2] swarm welfare-eff={welfare_efficiency(Q, swarm, 2).mean():.3f}, "
          f"closeness-to-opt={scalar_opt(swarm,2)[0]:.3f}  (payoff-blind; unchanged allocation)")
    print(f"[Regime 2] softmax-PG welfare-eff={welfare_efficiency(Q, res['pg_r2'][-1], 2).mean():.3f}, "
          f"closeness={final_opt(res['pg_r2'],2)[0]:.3f}")
    print(f"[Regime 2] Q-learning welfare-eff={welfare_efficiency(Q, res['ql_r2'][-1], 2).mean():.3f}, "
          f"closeness={final_opt(res['ql_r2'],2)[0]:.3f}")
    print(f"[Regime 1] closeness-to-opt: swarm={scalar_opt(swarm,1)[0]:.3f}, "
          f"PG={final_opt(res['pg_r1'],1)[0]:.3f}, QL={final_opt(res['ql_r1'],1)[0]:.3f}")


def _onehot_argmax(Q):
    t = np.zeros_like(Q); t[np.arange(Q.shape[0]), Q.argmax(1)] = 1.0; return t


if __name__ == "__main__":
    import sys as _s
    if len(_s.argv) > 1 and _s.argv[1] == "sanity":
        _sanity()
    else:
        _sanity()
        print("\n######## RUN + FIGURES ########")
        res = run_all(steps=4000, seed=0)
        make_figures(res)
