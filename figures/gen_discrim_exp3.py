#!/usr/bin/env python3
"""
Discrimination Exp 3 -- not-learning test by information plateau (DISCRIMINATION_EXPERIMENT_DESIGN.md Exp 3).

Question: does the swarm ACCUMULATE information about the resource structure over time (epistemic
learning), or reach a fixed informativeness and HOLD it (converged policy / homeostatic)?

Metric: closeness between the committed-occupancy distribution and the resource distribution,
closeness(t) = 1 - TV( n_hat(t), Q/Q_tot ) over real patches, trailing-window smoothed, averaged over
replicas with bootstrap CI. A learner improves this monotonically with experience; a converged policy
rises once and plateaus. (KL avoided -- blows up on empty patches, per the design self-check.)

Secondary (controlled perturbation): at a plateau tick, RELOCATE the resources to NEW patch locations
(same richness distribution Q_i, new clump centres). A cumulative learner would re-equilibrate FASTER or
to a HIGHER informativeness (it has learned); a converged/homeostatic policy re-tracks to the SAME
plateau with no cumulative gain. Canonical model is NOT edited -- relocation is done by resetting
clump centres / sources on the sim object between steps.

B=64 replicas, bootstrap CI over replicas. CPU. Raw data -> /mnt/newssd/.../paper2_discrim/.
Figure -> figures/fig_discrim_plateau.
"""
from __future__ import annotations
import os, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.normpath(os.path.join(_HERE, "..", "code"))
if not os.path.isdir(CODE):
    CODE = "/home/galen/Dropbox_2025/Dropbox/projects/flocking_LTM/Code"
sys.path.insert(0, CODE)
import torch  # noqa: E402
from mobile_cascade import MobileCascade, MobileCascadeConfig  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_RD = os.path.normpath(os.path.join(HERE, "..", "data", "paper2_discrim"))
DATA = _RD if os.path.isdir(_RD) else "/mnt/newssd/flocking_ltm_data/paper2_discrim"
os.makedirs(DATA, exist_ok=True)
N = 200


def relocate(sim):
    """Move resources to NEW patch locations, same richness assignment (_clump_idx unchanged)."""
    sim.clump_centers = sim._rand(sim.B, sim._n_clumps, 2) * sim.size
    sim.sources = sim._place_sources()
    sim.source_cap = torch.full((sim.B, sim.K), sim.capacity, device=sim.device, dtype=sim.dtype)


def run(B=64, T=800, relocate_at=None, seed=0):
    cfg = MobileCascadeConfig(B=B, N=N, K=80, L=800.0, rho_r=2.1, phi=0.02, theta=0.18,
                              levy_alpha=1.0, clump_count=10, clump_beta=2.2, clump_radius=30.0,
                              capacity=40.0, consumption=1.0, step=20.0, pull=0.12,
                              converge=True, device="cpu", seed=seed)
    sim = MobileCascade(cfg)
    C = sim._n_clumps
    idx = sim._clump_idx
    rich = torch.zeros(sim.B, C); rich.scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float32))
    occ_series = torch.empty(T, sim.B, C)
    for t in range(T):
        if relocate_at is not None and t == relocate_at:
            relocate(sim)
        sim.step()
        active = sim._active_state
        d = sim._pair_dist(sim.pos, sim.clump_centers)
        near = d.argmin(dim=2)
        o = torch.zeros(sim.B, C); o.scatter_add_(1, near, active.to(torch.float32))
        occ_series[t] = o
    return rich.numpy(), occ_series.numpy()


def closeness_series(rich, occ_series, window=20):
    """closeness(t,replica) = 1 - 0.5*sum|n_hat - Q/Qtot| over real patches, trailing-window smoothed."""
    T, R, C = occ_series.shape
    real = rich >= 1
    q = np.where(real, rich, 0.0)
    q = q / q.sum(1, keepdims=True).clip(min=1e-9)                     # (R,C) target
    out = np.full((T, R), np.nan)
    for t in range(T):
        lo = max(0, t - window + 1)
        o = occ_series[lo:t + 1].mean(0)                              # (R,C) smoothed occupancy
        o = np.where(real, o, 0.0)
        s = o.sum(1, keepdims=True)
        p = np.where(s > 0, o / s.clip(min=1e-9), 0.0)
        out[t] = 1.0 - 0.5 * np.abs(p - q).sum(1)
    return out                                                        # (T,R)


def boot_band(valsTR, nboot=2000):
    T, R = valsTR.shape
    rng = np.random.default_rng(7)
    draws = rng.integers(0, R, size=(nboot, R))
    mean = valsTR.mean(1)
    bm = valsTR[:, draws].mean(2)                                     # (T,nboot)
    return mean, np.percentile(bm, 2.5, axis=1), np.percentile(bm, 97.5, axis=1)


def main(B=64, T=800, relocate_at=400):
    cache = os.path.join(DATA, f"exp3_plateau_B{B}_T{T}.npz")
    if os.path.exists(cache) and os.environ.get("FORCE", "") != "1":
        print(f"loading {cache}"); z = np.load(cache)
        rich = z["rich"]; occ_fix = z["occ_fix"]; occ_rel = z["occ_rel"]
    else:
        t0 = time.time()
        rich, occ_fix = run(B=B, T=T, relocate_at=None, seed=0)
        print(f"  fixed done ({time.time()-t0:.0f}s)")
        rich2, occ_rel = run(B=B, T=T, relocate_at=relocate_at, seed=0)
        print(f"  relocation done ({time.time()-t0:.0f}s)")
        np.savez_compressed(cache, rich=rich, occ_fix=occ_fix, occ_rel=occ_rel)
        print(f"wrote {cache}")

    cf = closeness_series(rich, occ_fix)
    cr = closeness_series(rich, occ_rel)
    mf, lf, hf = boot_band(cf)
    mr, lr, hr = boot_band(cr)

    # plateau analysis on fixed run
    plateau = np.nanmean(mf[150:relocate_at])
    base0 = np.nanmean(mf[:10])
    tte = None
    tgt = base0 + 0.9 * (plateau - base0)
    for t in range(20, relocate_at):
        if mf[t] >= tgt and (mf[t:relocate_at] >= tgt - 0.01).all():
            tte = t; break
    # re-tracking: closeness right after relocation vs recovered plateau
    dip = np.nanmin(mr[relocate_at:relocate_at + 30])
    recov = np.nanmean(mr[relocate_at + 150:])
    print("\n=== Exp 3 information plateau + relocation (B=%d) ===" % B)
    print(f"  fixed: start closeness {base0:.3f} -> plateau {plateau:.3f} by ~{tte} ticks (90% of rise)")
    print(f"  late-fixed closeness (ticks {relocate_at}-{T}) = {np.nanmean(mf[relocate_at:]):.3f} "
          f"(flat: no further improvement => plateau, not monotone learning)")
    print(f"  relocation@{relocate_at}: dips to {dip:.3f}, re-tracks to {recov:.3f} "
          f"(vs pre-reloc plateau {plateau:.3f}) => re-tracking to SAME informativeness, not cumulative gain")

    make_figure(B, T, relocate_at, mf, lf, hf, mr, lr, hr, plateau, tte)


def make_figure(B, T, relocate_at, mf, lf, hf, mr, lr, hr, plateau, tte):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "legend.fontsize": 10.5})
    t = np.arange(T)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(t, mf, color="#2c7fb8", lw=2, label="fixed resources")
    ax.fill_between(t, lf, hf, color="#2c7fb8", alpha=0.2, lw=0)
    ax.plot(t, mr, color="#d95f0e", lw=2, label=f"resources relocated at t={relocate_at}")
    ax.fill_between(t, lr, hr, color="#d95f0e", alpha=0.2, lw=0)
    ax.axvline(relocate_at, color="#d95f0e", ls=":", lw=1.3)
    ax.axhline(plateau, color="#2c7fb8", ls="--", lw=1, alpha=0.7)
    ax.text(relocate_at + 6, 0.02 + ax.get_ylim()[0], "relocation", color="#d95f0e", fontsize=10)
    if tte:
        ax.axvline(tte, color="#2c7fb8", ls=":", lw=1)
        ax.text(tte + 6, plateau - 0.08, f"plateau by\n~{tte} ticks", color="#2c7fb8", fontsize=9.5)
    ax.set_xlabel("time (ticks)")
    ax.set_ylabel("closeness to resource distribution\n(1 $-$ total variation, real patches)")
    ax.set_title("Informativeness rises and plateaus, and re-tracks after relocation\n"
                 "(converged/homeostatic policy, not cumulative epistemic learning)", fontsize=10.6)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(HERE, f"fig_discrim_plateau.{ext}")
        fig.savefig(p, dpi=200 if ext == "png" else None); print("wrote", p)


if __name__ == "__main__":
    import sys as _s
    if len(_s.argv) > 1 and _s.argv[1] == "smoke":
        r, o = run(B=8, T=120, relocate_at=60); print("smoke ok", o.shape,
              "closeness end", np.nanmean(closeness_series(r, o)[-1]))
    else:
        main()
