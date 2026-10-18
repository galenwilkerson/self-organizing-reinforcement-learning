#!/usr/bin/env python3
"""
Exp C -- principled strengthener (EXPERIMENT_DESIGN_C.md; referee M1).

Replaces the single anecdotal 1.67x embodied result with (1) an ORACLE-normalized headline and (2) a
CROSSOVER MAP over horizon T x nonstationarity tau. Reuses the EXACT B2 env (gen_b2_embodied.py) unchanged;
varies only T and the resource-relocation interval tau, and adds an analytic ceiling.

ORACLE CEILING (per tick, per replica): the maximum total intake achievable under FREE placement given the
current source/depletion state, ignoring movement/search. The B2 intake rule gives total intake this tick =
sum_k min(cap_k, n_on_k) (CONSUMPTION=1; per-capita share <= 1). With one agent per source-slot and free
placement, the optimum is min(N, sum_k cap_k): give N agents a live-resource unit each; if total live
resource < N, all of it is taken. Verified empirically to be a strict upper bound on realized intake (max
realized/tick 177 < N=200; 0 violations). Both controllers report cumulative intake / cumulative ceiling, so
the comparison does not hinge on the learner's strength.

CROSSOVER: T in {200,600,1500,4000} x tau in {50,150,400,inf}, B=32 replicas (paired env fields per cell);
per cell record swarm & learner fraction-of-ceiling and Delta = swarm_frac - learner_frac. The B2 point
(T=600, tau=inf) must reproduce learner/swarm ~0.60.

CPU; canonical Code/ untouched; per-cell cache to /mnt/newssd/.../paper2_crossover/; T=4000 cells run last.
"""
from __future__ import annotations
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_b2_embodied as B2   # the EXACT B2 env / controllers
from gen_b2_embodied import Env, swarm_action, Learner, N

DATA = "/mnt/newssd/flocking_ltm_data/paper2_crossover"
os.makedirs(DATA, exist_ok=True)

TS = [200, 600, 1500, 4000]
TAUS = [50, 150, 400, None]          # None = stationary (inf)
R = 32
LR = 0.3


def tau_label(tau):
    return "inf" if tau is None else str(tau)


def run_cell(controller, T, tau, seed=0, lr=LR):
    """Returns per-replica fraction-of-ceiling (R,) and cumulative intake (R,). Paired env seed."""
    env = Env(R, seed=seed, regime=1)
    learner = Learner(R, lr=lr, seed=seed + 100) if controller == "learner" else None
    real_cum = np.zeros(R); ceil_cum = np.zeros(R)
    for t in range(T):
        if tau is not None and t > 0 and t % tau == 0:
            env.relocate()
        obs, on, sees, nf = env.observe()
        a = swarm_action(obs, on, sees, nf) if controller == "swarm" else learner.act(obs)
        cap_before = env.cap.sum(1)                     # (R,) live resource before extraction
        intake = env.step(a, on)
        if controller == "learner":
            learner.update(intake)
        real_cum += intake.sum(1)
        ceil_cum += np.minimum(N, cap_before)
    frac = real_cum / ceil_cum.clip(min=1e-9)
    return frac, real_cum, ceil_cum


def cell_path(T, tau):
    return os.path.join(DATA, f"cell_T{T}_tau{tau_label(tau)}_R{R}.npz")


def main():
    # order cells by T ascending (T=4000 last = cost driver)
    t0 = time.time()
    for T in TS:
        for tau in TAUS:
            p = cell_path(T, tau)
            if os.path.exists(p) and os.environ.get("FORCE", "") != "1":
                continue
            sw_frac, sw_cum, ceil = run_cell("swarm", T, tau, seed=0)
            ln_frac, ln_cum, _ = run_cell("learner", T, tau, seed=0)
            np.savez(p, sw_frac=sw_frac, ln_frac=ln_frac, sw_cum=sw_cum, ln_cum=ln_cum, ceil=ceil)
            print(f"  T={T:5d} tau={tau_label(tau):>4s}: swarm {sw_frac.mean():.3f}  "
                  f"learner {ln_frac.mean():.3f}  Delta {sw_frac.mean()-ln_frac.mean():+.3f}  "
                  f"ln/sw_intake {ln_cum.mean()/sw_cum.mean():.3f}  ({time.time()-t0:.0f}s)", flush=True)
    analyze()


def analyze():
    rng = np.random.default_rng(0); draws = rng.integers(0, R, size=(3000, R))
    sw = np.full((len(TS), len(TAUS)), np.nan); ln = np.full_like(sw, np.nan)
    dl = np.full_like(sw, np.nan); dllo = np.full_like(sw, np.nan); dlhi = np.full_like(sw, np.nan)
    cells = {}
    for i, T in enumerate(TS):
        for j, tau in enumerate(TAUS):
            p = cell_path(T, tau)
            if not os.path.exists(p):
                print(f"  MISSING {p}"); continue
            z = np.load(p); sf = z["sw_frac"]; lf = z["ln_frac"]
            sw[i, j] = sf.mean(); ln[i, j] = lf.mean()
            d = sf - lf; dl[i, j] = d.mean()
            db = d[draws].mean(1); dllo[i, j] = np.percentile(db, 2.5); dlhi[i, j] = np.percentile(db, 97.5)
            cells[(T, tau)] = dict(sw=sf, ln=lf)
    print("\n=== Crossover: fraction-of-ceiling (swarm / learner), Delta=sw-ln ===")
    print("rows T =", TS, " cols tau =", [tau_label(t) for t in TAUS])
    for i, T in enumerate(TS):
        print(f"  T={T:5d}: " + "  ".join(
            f"[{tau_label(TAUS[j])}] sw{sw[i,j]:.2f}/ln{ln[i,j]:.2f} D{dl[i,j]:+.2f}" for j in range(len(TAUS))))
    # B2 reproduction
    if (600, None) in cells:
        c = cells[(600, None)]
        print(f"\n  B2 point (T=600, tau=inf): swarm {c['sw'].mean():.3f}, learner {c['ln'].mean():.3f}, "
              f"learner/swarm intake-ratio = {c['ln'].mean()/c['sw'].mean():.3f} (target ~0.60, 1.67x)")
    # crossover boundary
    anyflip = np.nanmin(dl) < 0
    print(f"\n  Delta>0 everywhere (swarm wins): {not anyflip}; min Delta={np.nanmin(dl):+.3f} at "
          f"cell {np.unravel_index(np.nanargmin(dl), dl.shape)} (T,tau idx)")
    make_figure(sw, ln, dl, dllo, dlhi)
    return sw, ln, dl


def make_figure(sw, ln, dl, dllo, dlhi):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({"font.size": 11.5, "axes.labelsize": 11.5, "legend.fontsize": 9.5})
    taulab = [tau_label(t) for t in TAUS]
    fig = plt.figure(figsize=(14, 4.3))

    # (a) phase map of Delta
    ax = fig.add_subplot(1, 3, 1)
    vmax = np.nanmax(np.abs(dl))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(dl, cmap="RdBu_r", norm=norm, origin="lower", aspect="auto")
    ax.set_xticks(range(len(TAUS))); ax.set_xticklabels(taulab)
    ax.set_yticks(range(len(TS))); ax.set_yticklabels(TS)
    ax.set_xlabel(r"relocation interval $\tau$  (smaller = more nonstationary)")
    ax.set_ylabel("horizon $T$")
    for i in range(len(TS)):
        for j in range(len(TAUS)):
            ax.text(j, i, f"{dl[i,j]:+.2f}", ha="center", va="center", fontsize=8.5,
                    color="k" if abs(dl[i, j]) < 0.6 * vmax else "w")
    # Delta=0 contour (if any sign change)
    try:
        ax.contour(dl, levels=[0.0], colors="k", linewidths=2)
    except Exception:
        pass
    fig.colorbar(im, ax=ax, label=r"$\Delta$ = swarm $-$ learner (frac. of ceiling)")
    ax.set_title("(a) crossover map", fontsize=11)

    # (b) slice vs T at fixed tau=inf
    ax2 = fig.add_subplot(1, 3, 2)
    j = TAUS.index(None)
    ax2.plot(TS, sw[:, j], "-o", color="#2ca02c", lw=2, label="swarm")
    ax2.plot(TS, ln[:, j], "-s", color="#1f77b4", lw=2, label="learner")
    ax2.axhline(1.0, color="k", ls=":", lw=1, label="oracle ceiling")
    ax2.scatter([600], [sw[TS.index(600), j]], s=120, facecolors="none", edgecolors="#d62728",
                linewidths=2, zorder=6, label="B2 point")
    ax2.set_xscale("log"); ax2.set_xlabel("horizon $T$ (log)")
    ax2.set_ylabel("fraction of oracle ceiling"); ax2.set_ylim(0, 1.02)
    ax2.set_title(r"(b) vs horizon $T$ (stationary $\tau=\infty$)", fontsize=10.8); ax2.legend(fontsize=8.5)

    # (c) slice vs tau at fixed T=600
    ax3 = fig.add_subplot(1, 3, 3)
    i = TS.index(600)
    xt = [50, 150, 400, 900]  # map inf -> 900 for plotting position
    ax3.plot(xt, sw[i, :], "-o", color="#2ca02c", lw=2, label="swarm")
    ax3.plot(xt, ln[i, :], "-s", color="#1f77b4", lw=2, label="learner")
    ax3.axhline(1.0, color="k", ls=":", lw=1)
    ax3.set_xticks(xt); ax3.set_xticklabels(taulab)
    ax3.set_xlabel(r"relocation interval $\tau$"); ax3.set_ylabel("fraction of oracle ceiling")
    ax3.set_ylim(0, 1.02)
    ax3.set_title("(c) vs nonstationarity ($T=600$)", fontsize=10.8); ax3.legend(fontsize=8.5)
    fig.suptitle("Exp C — oracle-normalized performance and the fixed-vs-learned crossover map", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("pdf", "png"):
        p = os.path.join(HERE, f"fig_crossover.{ext}")
        fig.savefig(p, dpi=200 if ext == "png" else None); print("wrote", p)


if __name__ == "__main__":
    import sys as _s
    if len(_s.argv) > 1 and _s.argv[1] == "smoke":
        for ctrl in ["swarm", "learner"]:
            f, c, cl = run_cell(ctrl, 600, None, seed=0)
            print(ctrl, "T=600 tau=inf frac-of-ceiling", f.mean(), "cum", c.mean())
    elif len(_s.argv) > 1 and _s.argv[1] == "analyze":
        analyze()
    else:
        main()
