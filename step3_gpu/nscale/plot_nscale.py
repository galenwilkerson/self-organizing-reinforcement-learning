#!/usr/bin/env python3
"""Plot the finite-size N-scaling: IFD slope and harvest vs N, both fields. -> figures/fig_nscaling.pdf"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "/mnt/newssd/flocking_ltm_data/step3_gpu/nscale"
OUT = "fig_nscaling.pdf"
NS = [50, 100, 200, 400, 800]
FIELDS = [("PAPER", "scarce field", "#1f77b4", "o"), ("FAVORABLE", "clustered field", "#d62728", "s")]

d = {}
for n in NS:
    d[n] = np.load(os.path.join(DATA, f"nscale_N{n}.npz"), allow_pickle=True)["res"].item()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.3))
for fld, lab, col, mk in FIELDS:
    slope = [d[n][fld]["swarm_slope"] for n in NS]
    harv = [d[n][fld]["swarm_harvest"] for n in NS]
    hsd = [d[n][fld]["swarm_harvest_sd"] for n in NS]
    ax1.plot(NS, slope, mk + "-", color=col, label=lab, markersize=6)
    ax2.errorbar(NS, harv, yerr=hsd, fmt=mk + "-", color=col, label=lab, markersize=6, capsize=3)

ax1.axhline(1.0, ls="--", color="gray", lw=1, zorder=0)
ax1.set_xscale("log"); ax1.set_xticks(NS); ax1.set_xticklabels(NS)
ax1.set_xlabel("foragers $N$ (system grown $\\propto\\sqrt{N}$)"); ax1.set_ylabel("ideal-free matching slope")
ax1.set_title("(a) allocation shape vs size"); ax1.set_ylim(0.6, 1.6); ax1.legend(frameon=False, fontsize=9)

ax2.set_xscale("log"); ax2.set_xticks(NS); ax2.set_xticklabels(NS)
ax2.set_xlabel("foragers $N$ (system grown $\\propto\\sqrt{N}$)"); ax2.set_ylabel("harvest (fraction of oracle)")
ax2.set_title("(b) harvest vs size"); ax2.set_ylim(0, 1); ax2.legend(frameon=False, fontsize=9)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
# text summary
for n in NS:
    r = d[n]
    print(f"N={n:4d} L={r['L']:.0f} C={r['C']:2d} K={r['K']:3d} | "
          f"PAPER slope {r['PAPER']['swarm_slope']:.3f} harv {r['PAPER']['swarm_harvest']:.3f} "
          f"expl2 {r['PAPER']['swarm_exploit'][2.0]:.3f} | "
          f"FAV slope {r['FAVORABLE']['swarm_slope']:.3f} harv {r['FAVORABLE']['swarm_harvest']:.3f} "
          f"expl2 {r['FAVORABLE']['swarm_exploit'][2.0]:.3f}")
