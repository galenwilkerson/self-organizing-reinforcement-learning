#!/usr/bin/env python3
"""
Mechanism-validation ("bridge") experiment for paper2_rl_v2.

Tests the paper's CENTRAL ASSUMPTION: that the coarse-grained recruitment dynamics of the swarm are
reward-proportional replicator dynamics (so the IFD is the replicator / policy-gradient rest point),
rather than the IFD being reached by some other route (e.g. input-matching via source density + blocking).

Model: Code/mobile_cascade.py, the Code/ifd_test.py config, coupling ON (converge=True), B=64 replicas,
Pareto clumps (clump_beta=2.2). Committed foragers = ACTIVE agents assigned to their nearest clump
(same definition as ifd_test / the convergence figure). Richness Q_i = sources assigned to clump i.

Per patch i, per tick t, per replica we log:
  count_i(t)  committed-forager count (occupancy n_i = count_i/N)
  Jplus_i(t)  recruitment INFLOW   = agents whose committed-label becomes i this tick
                                     (inactive->i OR j->i)
  Jminus_i(t) RELEASE              = agents whose committed-label leaves i this tick
Derived:
  n_i = count_i/N ;  dn_i/dt = n_i(t)-n_i(t-1) = (Jplus_i-Jminus_i)/N   (identity)
  f_i = Q_i/count_i   (per-capita payoff: richness shared among committed foragers; defined count_i>0)
  fbar = sum_i n_i f_i   (= sum over OCCUPIED clumps of Q_i/N)

Two tests / figures:
 1. fig_recruit_vs_payoff : what does recruitment inflow Jplus_i scale with?
       Jplus vs f_i (reward-proportional hypothesis) vs Q_i (richness) vs n_i (crowd/contagion).
 2. fig_replicator_check  : does measured dn_i/dt follow the replicator RHS n_i(f_i-fbar)?
       Only informative in the TRANSIENT (ticks ~1-200); at steady state dn/dt~0.

Replicas B=64 (>=16). Bootstrap CIs over replicas. Raw data cached to /mnt/newssd (NEVER Dropbox).
CPU only.

Usage:
  python3 gen_bridge_figs.py            # simulate if no cache, then analyze + plot
  FORCE=1 python3 gen_bridge_figs.py    # re-simulate
"""
from __future__ import annotations
import os, sys, time
import numpy as np

CODE = "/home/galen/Dropbox_2025/Dropbox/projects/flocking_LTM/Code"
sys.path.insert(0, CODE)
import torch  # noqa: E402
from mobile_cascade import MobileCascade, MobileCascadeConfig  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/mnt/newssd/flocking_ltm_data/paper2_bridge"
os.makedirs(DATA, exist_ok=True)

B = int(sys.argv[1]) if len(sys.argv) > 1 else 64
T = int(sys.argv[2]) if len(sys.argv) > 2 else 800
SEED = 0
N_AGENTS = 200
TRANSIENT = (1, 200)      # window (ticks) for the replicator check
N_BOOT = 2000
CACHE = os.path.join(DATA, f"bridge_B{B}_T{T}_seed{SEED}.npz")


def simulate():
    cfg = MobileCascadeConfig(B=B, N=N_AGENTS, K=80, L=800.0, rho_r=2.1, phi=0.02, theta=0.18,
                              levy_alpha=1.0, clump_count=10, clump_beta=2.2, clump_radius=30.0,
                              capacity=40.0, consumption=1.0, step=20.0, pull=0.12,
                              converge=True, device="cpu", seed=SEED)
    sim = MobileCascade(cfg)
    C = sim._n_clumps
    idx = sim._clump_idx
    rich = torch.zeros(sim.B, C)
    rich.scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float32))   # (B,C) richness Q_i

    count = torch.empty(T, sim.B, C)
    Jplus = torch.empty(T, sim.B, C)         # TOTAL commits to i (uncommitted->i AND j->i)
    Jnew = torch.empty(T, sim.B, C)          # STRICT recruitment: uncommitted->i only (genuine new)
    Jminus = torch.empty(T, sim.B, C)        # TOTAL releases from i
    UNC = C                                   # "uncommitted" bin index
    prev_label = torch.full((sim.B, sim.N), UNC, dtype=torch.long)
    for t in range(T):
        sim.step()
        active = sim._active_state                           # (B,N) bool
        d = sim._pair_dist(sim.pos, sim.clump_centers)       # (B,N,C)
        near = d.argmin(dim=2)                               # (B,N) in [0,C)
        label = torch.where(active, near, torch.full_like(near, UNC))   # (B,N) in [0,C]
        # counts (drop the uncommitted bin)
        cf = torch.zeros(sim.B, C + 1)
        cf.scatter_add_(1, label, torch.ones_like(label, dtype=torch.float32))
        count[t] = cf[:, :C]
        # inflow / release vs previous tick's label
        changed = label != prev_label
        commit = changed & (label < C)                       # became committed to label (i)
        commit_new = commit & (prev_label == UNC)            # from uncommitted only (strict recruitment)
        release = changed & (prev_label < C)                 # left prev_label (j)
        jp = torch.zeros(sim.B, C + 1)
        jp.scatter_add_(1, label, commit.to(torch.float32))
        Jplus[t] = jp[:, :C]
        jn = torch.zeros(sim.B, C + 1)
        jn.scatter_add_(1, label, commit_new.to(torch.float32))
        Jnew[t] = jn[:, :C]
        jm = torch.zeros(sim.B, C + 1)
        jm.scatter_add_(1, prev_label, release.to(torch.float32))
        Jminus[t] = jm[:, :C]
        prev_label = label
    return (rich.numpy(), count.numpy(), Jplus.numpy(), Jnew.numpy(), Jminus.numpy())


def get_data():
    if os.path.exists(CACHE) and os.environ.get("FORCE", "") != "1":
        print(f"loading cache {CACHE}")
        z = np.load(CACHE)
        return z["rich"], z["count"], z["Jplus"], z["Jnew"], z["Jminus"]
    print(f"simulating coupling-ON B={B} T={T} seed={SEED} (CPU) ...")
    t0 = time.time()
    rich, count, Jplus, Jnew, Jminus = simulate()
    np.savez_compressed(CACHE, rich=rich, count=count, Jplus=Jplus, Jnew=Jnew, Jminus=Jminus)
    print(f"wrote {CACHE} ({time.time()-t0:.0f}s)")
    return rich, count, Jplus, Jnew, Jminus


# ------------------------------------------------------------------ analysis helpers
def pearson(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    return np.corrcoef(x[m], y[m])[0, 1]


def spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    from scipy.stats import rankdata
    return np.corrcoef(rankdata(x[m]), rankdata(y[m]))[0, 1]


def boot_corr(xb, yb, repl, func):
    """Bootstrap a correlation over replicas. xb,yb,repl are 1-D pooled arrays; repl = replica id."""
    rng = np.random.default_rng(7)
    uids = np.unique(repl)
    out = []
    for _ in range(N_BOOT):
        pick = rng.choice(uids, size=len(uids), replace=True)
        mask = np.isin(repl, pick)
        out.append(func(xb[mask], yb[mask]))
    out = np.array(out)
    out = out[np.isfinite(out)]
    return np.percentile(out, [2.5, 97.5])


def binned(x, y, nbins=12, logx=False):
    """Quantile-binned mean(y) vs mean(x) with per-bin SEM-ish via std/sqrt(n)."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    xb = np.log10(x) if logx else x
    qs = np.quantile(xb, np.linspace(0, 1, nbins + 1))
    qs[-1] += 1e-9
    xm, ym, yl, yh = [], [], [], []
    for b in range(nbins):
        sel = (xb >= qs[b]) & (xb < qs[b + 1])
        if sel.sum() < 5:
            continue
        xm.append(x[sel].mean()); ym.append(y[sel].mean())
        se = y[sel].std() / np.sqrt(sel.sum())
        yl.append(y[sel].mean() - 1.96 * se); yh.append(y[sel].mean() + 1.96 * se)
    return map(np.array, (xm, ym, yl, yh))


def _explore():
    rich, count, Jplus, Jnew, Jminus = get_data()
    Tn, Bn, C = count.shape
    N = N_AGENTS
    Q = np.broadcast_to(rich[None], count.shape)
    n = count / N
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(count > 0, Q / count, np.nan)
    occupied = count > 0
    realpatch = Q >= 1                                          # clump actually has sources
    fbar = (np.where(occupied, Q, 0.0).sum(axis=2) / N)         # (T,B) = sum_i n_i f_i

    def pool(window, cond):
        lo, hi = window
        tmask = np.zeros(Tn, bool); tmask[lo:hi] = True
        return cond & tmask[:, None, None]

    print("shapes count", count.shape, "rich", rich.shape)
    print("Q range per clump: %.0f..%.0f ; empty clumps (Q=0) per replica avg %.1f/%d"
          % (rich.min(), rich.max(), (rich < 1).sum(1).mean(), C))
    print("mean total inflow/tick %.2f ; mean STRICT(new) inflow/tick %.2f (frac new %.2f)"
          % (Jplus.mean() * C, Jnew.mean() * C, Jnew.sum() / max(Jplus.sum(), 1)))

    for jname, J in [("Jplus(total)", Jplus), ("Jnew(strict recruit)", Jnew)]:
        print(f"\n===== inflow = {jname} =====")
        for name, win in [("all t>=5", (5, Tn)), ("transient 1-200", (1, 200)), ("steady 400-800", (400, Tn))]:
            sel = pool(win, occupied & realpatch)
            jp = J[sel]; ff = f[sel]; qq = Q[sel]; nn = n[sel]
            print(f"[{name}] n={jp.size}  "
                  f"corr(J,f)={pearson(jp, ff):+.3f}  corr(J,Q)={pearson(jp, qq):+.3f}  "
                  f"corr(J,n)={pearson(jp, nn):+.3f}  sp(J,f)={spearman(jp, ff):+.3f} "
                  f"sp(J,Q)={spearman(jp, qq):+.3f} sp(J,n)={spearman(jp, nn):+.3f}")

    # ---------- replicator check ----------
    dn = np.full_like(n, np.nan); dn[1:] = n[1:] - n[:-1]
    g = np.full_like(n, np.nan)
    g[1:] = n[:-1] * (f[:-1] - fbar[:-1][:, :, None])          # predictor from previous state
    prevpos = np.zeros_like(dn, bool); prevpos[1:] = count[:-1] > 0

    def repl_fit(win, label):
        lo, hi = win
        v = np.zeros_like(dn, bool); v[lo:hi] = True
        v &= np.isfinite(dn) & np.isfinite(g) & (count > 0) & prevpos & realpatch
        gg, dd = g[v], dn[v]
        A = np.polyfit(gg, dd, 1); r = pearson(gg, dd)
        s0 = np.sum(gg * dd) / np.sum(gg * gg)
        print(f"[replicator {label}] n={gg.size}  slope={A[0]:.3f} (through-origin {s0:.3f}) "
              f"r={r:+.3f} R2={r**2:.3f}")
        return gg, dd

    print("\n===== replicator: dn/dt vs n(f-fbar) =====")
    repl_fit(TRANSIENT, "tick-resolution transient 1-200")
    repl_fit((1, Tn), "tick-resolution all")

    # block-averaged (reduce tick noise): average n & g over BLK-tick blocks per (replica,patch)
    BLK = 10
    nb = Tn // BLK
    nn_blk = n[:nb * BLK].reshape(nb, BLK, Bn, C).mean(axis=1)          # (nb,B,C)
    f_blk = np.where(count[:nb * BLK].reshape(nb, BLK, Bn, C).sum(1) > 0,
                     Q[:nb * BLK].reshape(nb, BLK, Bn, C).mean(1)
                     / (count[:nb * BLK].reshape(nb, BLK, Bn, C).mean(1) + 1e-9), np.nan)
    fbar_blk = np.nansum(np.where(nn_blk > 0, Q[0][None] / N, 0.0), axis=2)  # approx, (nb,B)
    dn_blk = np.full_like(nn_blk, np.nan); dn_blk[1:] = nn_blk[1:] - nn_blk[:-1]
    g_blk = np.full_like(nn_blk, np.nan)
    g_blk[1:] = nn_blk[:-1] * (f_blk[:-1] - fbar_blk[:-1][:, :, None])
    realb = Q[0][None] >= 1
    tb = int(np.ceil(TRANSIENT[1] / BLK))
    vb = np.zeros_like(dn_blk, bool); vb[1:tb] = True
    vb &= np.isfinite(dn_blk) & np.isfinite(g_blk) & realb
    gg, dd = g_blk[vb], dn_blk[vb]
    A = np.polyfit(gg, dd, 1); r = pearson(gg, dd)
    print(f"[replicator BLOCK-avg({BLK}) transient] n={gg.size}  slope={A[0]:.3f} "
          f"r={r:+.3f} R2={r**2:.3f}")


def bin_boot(x, y, repl, nbins, Bn, logx=False):
    """Quantile-binned mean(y) vs mean(x), with 95% CI by bootstrap OVER replicas.
    Efficient: precompute per-(replica,bin) sums, resample replicas."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y, repl = x[m], y[m], repl[m]
    xb = np.log10(x) if logx else x
    edges = np.quantile(xb, np.linspace(0, 1, nbins + 1)); edges[-1] += 1e-12
    bidx = np.clip(np.digitize(xb, edges) - 1, 0, nbins - 1)
    key = repl * nbins + bidx
    sy = np.bincount(key, weights=y, minlength=Bn * nbins).reshape(Bn, nbins)
    sx = np.bincount(key, weights=x, minlength=Bn * nbins).reshape(Bn, nbins)
    cc = np.bincount(key, minlength=Bn * nbins).reshape(Bn, nbins).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        xm = sx.sum(0) / cc.sum(0)
        ym = sy.sum(0) / cc.sum(0)
    rng = np.random.default_rng(11)
    draws = rng.integers(0, Bn, size=(N_BOOT, Bn))
    S = sy[draws].sum(1); Cc = cc[draws].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        bm = S / Cc
    ylo = np.nanpercentile(bm, 2.5, axis=0)
    yhi = np.nanpercentile(bm, 97.5, axis=0)
    return xm, ym, ylo, yhi


def boot_r(x, y, repl, Bn, nb=600):
    """Bootstrap 95% CI for Pearson r over replicas (grouped resample)."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y, repl = x[m], y[m], repl[m]
    groups = [np.where(repl == r)[0] for r in range(Bn)]
    rng = np.random.default_rng(3)
    out = []
    for _ in range(nb):
        pick = rng.integers(0, Bn, size=Bn)
        idx = np.concatenate([groups[p] for p in pick])
        out.append(np.corrcoef(x[idx], y[idx])[0, 1])
    out = np.array(out)
    return np.percentile(out, [2.5, 97.5])


def load_derived():
    rich, count, Jplus, Jnew, Jminus = get_data()
    Tn, Bn, C = count.shape
    N = N_AGENTS
    Q = np.broadcast_to(rich[None], count.shape).astype(float)
    n = count / N
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(count > 0, Q / count, np.nan)
    fbar = np.where(count > 0, Q, 0.0).sum(axis=2) / N
    return dict(rich=rich, count=count, Jplus=Jplus, Jnew=Jnew, Jminus=Jminus,
                Q=Q, n=n, f=f, fbar=fbar, Tn=Tn, Bn=Bn, C=C, N=N)


def make_fig1(D):
    """Recruitment inflow vs per-capita payoff f_i, richness Q_i, occupancy n_i."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12.5, "axes.labelsize": 12.5, "legend.fontsize": 10.5,
                         "xtick.labelsize": 11, "ytick.labelsize": 11})
    Tn, Bn, C = D["Tn"], D["Bn"], D["C"]
    tmask = np.zeros(Tn, bool); tmask[5:] = True
    sel = (D["count"] > 0) & (D["Q"] >= 1) & tmask[:, None, None]
    J = D["Jnew"][sel]                                        # strict recruitment inflow
    f = D["f"][sel]; Q = D["Q"][sel]; n = D["n"][sel]
    repl = np.broadcast_to(np.arange(Bn)[None, :, None], D["count"].shape)[sel]

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.9))
    specs = [(f, "per-capita payoff  $f_i = Q_i/(N n_i)$", "#8e44ad", False,
              "reward-proportional?"),
             (Q, "patch richness  $Q_i$ (sources)", "#2c7fb8", False, "richness-proportional?"),
             (n, "occupancy  $n_i$ (committed fraction)", "#e67e22", False, "crowd / contagion?")]
    for k, (xv, xlab, col, logx, qlab) in enumerate(specs):
        xm, ym, ylo, yhi = bin_boot(xv, J, repl, 12, Bn, logx=logx)
        ax[k].plot(xm, ym, "-o", color=col, ms=4, lw=1.8, zorder=3)
        ax[k].fill_between(xm, ylo, yhi, color=col, alpha=0.22, lw=0, zorder=2)
        r = pearson(J, xv); rlo, rhi = boot_r(J, xv, repl, Bn)
        ax[k].set_title(f"{qlab}\nPearson $r$={r:+.2f}  [{rlo:+.2f}, {rhi:+.2f}]", fontsize=11)
        ax[k].set_xlabel(xlab)
        ax[k].grid(alpha=0.25)
    ax[0].set_ylabel("recruitment inflow $J^+_i$\n(new commits / tick)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(HERE, f"fig_recruit_vs_payoff.{ext}")
        fig.savefig(p, dpi=200 if ext == "png" else None)
        print("wrote", p)


def make_fig2(D):
    """Replicator check: measured dn/dt vs n(f-fbar), transient; binned + fit + y=x."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12.5, "axes.labelsize": 12.5, "legend.fontsize": 10.5})
    Tn, Bn, C, N = D["Tn"], D["Bn"], D["C"], D["N"]
    n, f, fbar, count, Q = D["n"], D["f"], D["fbar"], D["count"], D["Q"]
    dn = np.full_like(n, np.nan); dn[1:] = n[1:] - n[:-1]
    g = np.full_like(n, np.nan); g[1:] = n[:-1] * (f[:-1] - fbar[:-1][:, :, None])
    prevpos = np.zeros_like(dn, bool); prevpos[1:] = count[:-1] > 0
    lo, hi = TRANSIENT
    v = np.zeros_like(dn, bool); v[lo:hi] = True
    v &= np.isfinite(dn) & np.isfinite(g) & (count > 0) & prevpos & (Q >= 1)
    gg, dd = g[v], dn[v]
    repl = np.broadcast_to(np.arange(Bn)[None, :, None], D["count"].shape)[v]

    A = np.polyfit(gg, dd, 1); r = pearson(gg, dd)
    # bootstrap slope over replicas
    groups = [np.where(repl == rr)[0] for rr in range(Bn)]
    rng = np.random.default_rng(5); slopes = []
    for _ in range(600):
        pick = rng.integers(0, Bn, size=Bn)
        idx = np.concatenate([groups[p] for p in pick])
        slopes.append(np.polyfit(gg[idx], dd[idx], 1)[0])
    slo, shi = np.percentile(slopes, [2.5, 97.5])

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    xm, ym, ylo, yhi = bin_boot(gg, dd, repl, 14, Bn)
    ax.axhline(0, color="k", lw=0.6, alpha=0.5); ax.axvline(0, color="k", lw=0.6, alpha=0.5)
    # y=x reference (slope-1 replicator with unit rate)
    lim = np.nanpercentile(np.abs(gg), 99)
    xs = np.linspace(-lim, lim, 50)
    ax.plot(xs, xs, "k--", lw=1, label="$y=x$ (unit-rate replicator)")
    ax.plot(xs, A[0] * xs + A[1], color="#c0392b", lw=2,
            label=f"fit: slope={A[0]:.2f} [{slo:.2f},{shi:.2f}]")
    ax.errorbar(xm, ym, yerr=[ym - ylo, yhi - ym], fmt="o", color="#2c7fb8", ms=5, lw=1,
                capsize=2, zorder=4, label="binned mean $\\pm$95% CI")
    ax.set_xlim(-lim, lim)
    yl = np.nanpercentile(np.abs(dd), 99)
    ax.set_ylim(-yl, yl)
    ax.set_xlabel("replicator prediction  $n_i\\,(f_i-\\bar f)$")
    ax.set_ylabel("measured  $dn_i/dt$")
    ax.set_title(f"transient ticks {lo}-{hi}:  Pearson $r$={r:+.2f},  $R^2$={r**2:.3f}", fontsize=11.5)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(HERE, f"fig_replicator_check.{ext}")
        fig.savefig(p, dpi=200 if ext == "png" else None)
        print("wrote", p)
    return A[0], (slo, shi), r


if __name__ == "__main__":
    print("\n######## EXPLORATION ########")
    _explore()
    print("\n######## FIGURES ########")
    D = load_derived()
    make_fig1(D)
    slope, sci, r = make_fig2(D)
    print(f"\nreplicator transient fit: slope={slope:.3f} CI{sci}  r={r:+.3f} R2={r**2:.3f}")
