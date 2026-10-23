#!/usr/bin/env python3
"""Learner training curves for Paper 2 (v5): the fairness / convergence evidence (Decision A).

A fair "swarm vs trained learner" comparison requires the learner to have actually converged. This figure
shows the embodied learners' harvest fraction vs training ticks (150k), for both operating points, with the
learner-free swarm and the random baseline drawn as horizontal reference lines. Both learners plateau well
before 150k (the binding comparator, MLP-IPPO, is flat over the last half: slope ~2e-5 per 1k ticks), so the
learner had every opportunity to converge. Data: step3_gpu_long.npz (IPPO) + step3_gpu_mappo.npz (CTDE).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_RD = os.path.normpath(os.path.join(_HERE, "..", "data", "step3_gpu"))
DATA_DIR = _RD if os.path.isdir(_RD) else "/mnt/newssd/flocking_ltm_data/step3_gpu"
LONG = np.load(os.path.join(DATA_DIR, "step3_gpu_long.npz"), allow_pickle=True)["res"].item()
MAPPO = np.load(os.path.join(DATA_DIR, "step3_gpu_mappo.npz"), allow_pickle=True)["res"].item()
HERE = os.path.dirname(os.path.abspath(__file__))
POINTS = ["PAPER", "FAVORABLE"]
TITLE = {"PAPER": "(a) scarce", "FAVORABLE": "(b) clustered, resource-rich"}

# (label, source, curve-key, ct-key, colour)
CURVES = [
    ("MLP-IPPO",   LONG,  "mlp_ippo_curve_f", "mlp_ippo_ct", "#1f77b4"),
    ("MAPPO (CTDE)", MAPPO, "mappo_mlp_curve_f", "mappo_mlp_ct", "#9467bd"),
]

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=False)
for ax, pt in zip(axes, POINTS):
    for lab, src, ck, tk, c in CURVES:
        cf = np.asarray(src[pt][ck], float)       # (n_ckpt, n_worlds)
        ct = np.asarray(src[pt][tk], float)        # (n_ckpt,)
        mean = cf.mean(axis=1)
        sem = cf.std(axis=1, ddof=1) / np.sqrt(cf.shape[1])
        ax.plot(ct, mean, color=c, lw=2.0, label=lab, zorder=3)
        ax.fill_between(ct, mean - sem, mean + sem, color=c, alpha=0.18, linewidth=0, zorder=2)
    # swarm / random reference lines (learner-free; 0 training)
    sw = np.asarray(LONG[pt]["swarm"], float).mean()
    rnd = np.asarray(LONG[pt]["random"], float).mean()
    ax.axhline(sw, ls="--", color="#2ca02c", lw=1.6, zorder=1,
               label="swarm (0 train)")
    ax.axhline(rnd, ls=":", color="#9e9e9e", lw=1.4, zorder=1, label="random")
    ax.set_title(TITLE[pt], fontsize=11)
    ax.set_xlabel("training ticks")
    ax.set_xlim(0, ct.max())
    if pt == POINTS[0]:
        ax.set_ylabel("harvest fraction of oracle ceiling")
        ax.legend(fontsize=8.5, loc="lower right", frameon=False)
fig.tight_layout()
out = os.path.join(HERE, "fig_learning_curves.pdf")
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print("wrote", out)
