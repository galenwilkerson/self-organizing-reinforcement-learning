#!/usr/bin/env python3
"""
Convergence / time-to-equilibrium figure for paper2_rl_v2.

Claim supported: a recruiting foraging swarm reaches the Ideal Free Distribution (IFD)
*dynamically* -- committed-forager allocation across patches converges OVER TIME to be
proportional to patch richness; the ablated swarm (recruitment->motion coupling OFF) does
not. The paper already reports the END-STATE match (matching slope ~0.84 ON / ~0.09 OFF,
from Code/ifd_test.py); this figure shows the TIME COURSE of that match.

Model: Code/mobile_cascade.py (same config as Code/ifd_test.py). The only change here is
instrumentation: committed-forager occupancy per clump is recorded EVERY tick (ifd_test only
averages over the tail). Convergence metrics are then evaluated in a trailing window.

Two convergence metrics (both plotted):
  (a) IFD matching slope: regress log(active agents at clump) on log(patch richness), pooling
      (replica, clump). slope -> 1 == ideal free distribution. Ties directly to the paper's
      0.84 / 0.09 end-state numbers.
  (b) Distance to IFD: total-variation distance between the committed-forager distribution
      n_i/sum(n) and the target richness distribution Q_i/sum(Q), averaged over replicas.
      distance -> 0 == IFD.

Replicas = the batch dimension B (each b has an independent clump layout/richness and
trajectory). Confidence bands are bootstrap over replicas (the project rule: always bootstrap).

Raw simulation data is cached under /mnt/newssd (NEVER in Dropbox). The figure PDF/PNG and this
script live in the paper repo.

CPU only. Usage:
  python3 gen_convergence_fig.py [B] [T]       # simulate (if no cache) then plot
  FORCE=1 python3 gen_convergence_fig.py        # re-simulate even if cache exists
"""
from __future__ import annotations
import os, sys, time
import numpy as np

CODE = "/home/galen/Dropbox_2025/Dropbox/projects/flocking_LTM/Code"
sys.path.insert(0, CODE)
import torch  # noqa: E402
from mobile_cascade import MobileCascade, MobileCascadeConfig  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/mnt/newssd/flocking_ltm_data/paper2_convergence"
os.makedirs(DATA, exist_ok=True)

# ---- experiment configuration (matches Code/ifd_test.py exactly, B configurable) ----
B = int(sys.argv[1]) if len(sys.argv) > 1 else 64      # replicas
T = int(sys.argv[2]) if len(sys.argv) > 2 else 800     # ticks
SEED = 0
WINDOW = 60          # trailing-window width (ticks) for the allocation estimate
CKPT_STEP = 10       # evaluate a metric checkpoint every this many ticks
N_BOOT = 2000        # bootstrap resamples over replicas
CACHE = os.path.join(DATA, f"convergence_B{B}_T{T}_seed{SEED}.npz")


def simulate(converge: bool):
    """Run one condition; return rich (B,C) and occ_series (T,B,C) of committed foragers/clump."""
    cfg = MobileCascadeConfig(B=B, N=200, K=80, L=800.0, rho_r=2.1, phi=0.02, theta=0.18,
                              levy_alpha=1.0, clump_count=10, clump_beta=2.2, clump_radius=30.0,
                              capacity=40.0, consumption=1.0, step=20.0, pull=0.12,
                              converge=converge, device="cpu", seed=SEED)
    sim = MobileCascade(cfg)
    C = sim._n_clumps
    # richness of each clump = number of sources assigned to it (fixed per replica)
    idx = sim._clump_idx                                      # (B,K) clump of each source
    rich = torch.zeros(sim.B, C)
    rich.scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float32))   # (B,C)
    occ_series = torch.empty(T, sim.B, C)
    for t in range(T):
        sim.step()
        # committed foragers only = ACTIVE agents (seeds + recruited), assigned to nearest clump
        active = sim._active_state                           # (B,N) bool
        d = sim._pair_dist(sim.pos, sim.clump_centers)       # (B,N,C)
        near = d.argmin(dim=2)                               # (B,N)
        o = torch.zeros(sim.B, C)
        o.scatter_add_(1, near, active.to(torch.float32))
        occ_series[t] = o
    return rich.numpy(), occ_series.numpy()


def get_data():
    if os.path.exists(CACHE) and os.environ.get("FORCE", "") != "1":
        print(f"loading cache {CACHE}")
        d = np.load(CACHE)
        return d["rich_on"], d["occ_on"], d["rich_off"], d["occ_off"]
    print(f"simulating B={B} T={T} seed={SEED} (CPU) ...")
    t0 = time.time()
    rich_on, occ_on = simulate(True)
    print(f"  coupling ON done ({time.time()-t0:.0f}s)")
    t1 = time.time()
    rich_off, occ_off = simulate(False)
    print(f"  coupling OFF done ({time.time()-t1:.0f}s)")
    np.savez_compressed(CACHE, rich_on=rich_on, occ_on=occ_on,
                        rich_off=rich_off, occ_off=occ_off)
    print(f"wrote {CACHE} ({time.time()-t0:.0f}s total)")
    return rich_on, occ_on, rich_off, occ_off


# ---------------------------- metrics ----------------------------
def window_occ(occ_series, t, w):
    """Trailing-window mean occupancy (B,C) over ticks (t-w, t]."""
    lo = max(0, t - w)
    return occ_series[lo:t].mean(axis=0)


def slope_pooled(rich, occw, boot_idx=None):
    """Matching slope: regress log(occ) on log(rich), pooling (replica,clump).
    boot_idx: optional array of replica indices (bootstrap resample)."""
    r = rich[boot_idx] if boot_idx is not None else rich
    o = occw[boot_idx] if boot_idx is not None else occw
    r = r.flatten(); o = o.flatten()
    m = (r >= 1) & (o >= 0.5)
    if m.sum() < 8:
        return np.nan
    x, y = np.log(r[m]), np.log(o[m])
    return np.polyfit(x, y, 1)[0]


def tv_per_replica(rich, occw):
    """Total-variation distance to IFD per replica: 0.5*sum|n_i/sum n - Q_i/sum Q|, (B,)."""
    q = rich / rich.sum(axis=1, keepdims=True).clip(min=1e-9)
    s = occw.sum(axis=1, keepdims=True)
    p = np.where(s > 0, occw / s.clip(min=1e-9), 0.0)
    return 0.5 * np.abs(p - q).sum(axis=1)


def analyze(rich, occ_series):
    Bn = rich.shape[0]
    ticks = np.arange(WINDOW, T + 1, CKPT_STEP)
    slope_med = np.full(len(ticks), np.nan)
    slope_lo = np.full(len(ticks), np.nan)
    slope_hi = np.full(len(ticks), np.nan)
    tv_med = np.full(len(ticks), np.nan)
    tv_lo = np.full(len(ticks), np.nan)
    tv_hi = np.full(len(ticks), np.nan)
    rng = np.random.default_rng(12345)
    boots = rng.integers(0, Bn, size=(N_BOOT, Bn))
    for i, t in enumerate(ticks):
        occw = window_occ(occ_series, t, WINDOW)
        # slope: point estimate on full sample + bootstrap over replicas
        slope_med[i] = slope_pooled(rich, occw)
        bs = np.array([slope_pooled(rich, occw, boots[b]) for b in range(N_BOOT)])
        bs = bs[~np.isnan(bs)]
        slope_lo[i], slope_hi[i] = np.percentile(bs, [2.5, 97.5])
        # tv: per-replica -> bootstrap mean over replicas
        tv = tv_per_replica(rich, occw)
        tv_med[i] = tv.mean()
        tvb = tv[boots].mean(axis=1)
        tv_lo[i], tv_hi[i] = np.percentile(tvb, [2.5, 97.5])
    return dict(ticks=ticks, slope=slope_med, slope_lo=slope_lo, slope_hi=slope_hi,
                tv=tv_med, tv_lo=tv_lo, tv_hi=tv_hi)


def time_to_equilibrium(ticks, curve, frac=0.90):
    """First tick at which curve reaches frac of its plateau (mean of last 10% of checkpoints)
    and stays >= that level thereafter."""
    plateau = np.nanmean(curve[-max(3, len(curve) // 10):])
    target = frac * plateau
    reached = curve >= target
    for i in range(len(ticks)):
        if reached[i] and reached[i:].all():
            return int(ticks[i]), plateau
    return None, plateau


# ---------------------------- figure ----------------------------
def make_figure(res_on, res_off, teq_on):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13,
                         "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11.5})
    C_ON, C_OFF = "#2ca02c", "#c0392b"
    fig, ax = plt.subplots(figsize=(6.6, 4.3))

    for res, col, lab in [(res_on, C_ON, "recruiting (coupling on)"),
                          (res_off, C_OFF, "ablated (coupling off)")]:
        ax.plot(res["ticks"], res["slope"], color=col, lw=2.2, label=lab, zorder=3)
        ax.fill_between(res["ticks"], res["slope_lo"], res["slope_hi"], color=col, alpha=0.20,
                        linewidth=0, zorder=2)
    # IFD reference
    ax.axhline(1.0, ls="--", color="k", lw=1, zorder=1)
    ax.text(res_on["ticks"][-1], 1.005, "ideal free distribution (slope 1)",
            va="bottom", ha="right", fontsize=10.5)
    # time-to-equilibrium marker (coupling on)
    if teq_on is not None:
        ax.axvline(teq_on, ls=":", color=C_ON, lw=1.2, zorder=1)
        ax.text(teq_on + 8, 0.08, f"90% of plateau\nby {teq_on} ticks",
                color=C_ON, fontsize=10, va="bottom", ha="left")

    ax.set_xlabel("time (ticks)")
    ax.set_ylabel("IFD matching slope\n($\\log n_i$ vs $\\log Q_i$)")
    ax.set_ylim(-0.08, 1.08)
    ax.set_xlim(0, res_on["ticks"][-1])
    ax.legend(loc="center right", frameon=False)
    fig.tight_layout()
    pdf = os.path.join(HERE, "fig_convergence.pdf")
    png = os.path.join(HERE, "fig_convergence.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    print("wrote", pdf)
    print("wrote", png)


def main():
    rich_on, occ_on, rich_off, occ_off = get_data()
    res_on = analyze(rich_on, occ_on)
    res_off = analyze(rich_off, occ_off)

    # report
    teq, plat_on = time_to_equilibrium(res_on["ticks"], res_on["slope"], 0.90)
    print("\n=== convergence summary (matching slope) ===")
    print(f"coupling ON : final slope ~ {np.nanmean(res_on['slope'][-5:]):.3f}  "
          f"(plateau {plat_on:.3f}); time-to-90%-of-plateau = {teq} ticks")
    print(f"coupling OFF: final slope ~ {np.nanmean(res_off['slope'][-5:]):.3f}")
    print("=== convergence summary (TV distance to IFD) ===")
    print(f"coupling ON : start {res_on['tv'][0]:.3f} -> final {np.nanmean(res_on['tv'][-5:]):.3f}")
    print(f"coupling OFF: start {res_off['tv'][0]:.3f} -> final {np.nanmean(res_off['tv'][-5:]):.3f}")
    print(f"replicas B={B}, bootstrap N={N_BOOT} over replicas, window={WINDOW}, ckpt_step={CKPT_STEP}")

    # persist the analyzed curves alongside the raw cache
    np.savez(os.path.join(DATA, f"convergence_curves_B{B}_T{T}.npz"),
             **{f"on_{k}": v for k, v in res_on.items()},
             **{f"off_{k}": v for k, v in res_off.items()})
    make_figure(res_on, res_off, teq)


if __name__ == "__main__":
    main()
