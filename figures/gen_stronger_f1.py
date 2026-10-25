#!/usr/bin/env python3
"""Rebuild fig_stronger for Paper 2 (v5) from the FAIR F1 embodied comparison (step3_gpu, 150k).

Both arms use the sensing-limited F1 primitives: approach steers only toward a source sensed within
radius S (or a recruited heading), identical for swarm and learner. Data: the 150k longer-training run
(step3_gpu_long.npz = swarm, random, MLP-IPPO, GNN-L1/L3/L5) merged with the centralized-critic steelman
(step3_gpu_mappo.npz = MAPPO-MLP, MAPPO-GNN-L3). 16 seeds x 16 eval replicas per operating point.

Three panels (two operating points each: PAPER=scarce, FAVORABLE=clustered):
 (a) harvest fraction of a validated oracle ceiling. The reward-free swarm (0 training) is competitive
     everywhere, statistically ties the best learner (MLP-IPPO) in the clustered regime, and beats the
     centralized-critic steelman (MAPPO-MLP) there; the learner out-harvests it only in the scarce regime.
 (b) ideal-free allocation slope (log committed foragers vs log richness). The reward-free swarm matches the
     resource gradient most faithfully (slope ~1) in both regimes -- it is UNBIASED w.r.t. the gradient,
     while the trained learners systematically under-concentrate (slope < 1). Dashed line = ideal slope 1.
 (c) exploitability = occupancy-weighted CV of per-capita intake (0 = equal-intake Nash). Here the trained
     learner is more PRECISE (lower CV): learning buys lower per-capita variance, not a better gradient match.
     Panels (b)+(c) are the honest bias-variance split: emergent allocation is gradient-unbiased but noisier;
     learned allocation is gradient-biased but tighter. Neither dominates on allocation.
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
NSEED = LONG["_meta"]["nseed"]

# GNN L=1 HARVEST re-run at matched parallel-environment count (R=24, 24 seeds); see
# step3_gpu/GNN_DONE_RIGHT_DESIGN.md. Override ONLY the L=1 harvest finals for panel (a);
# slope (b) and exploitability (c) keep the original R=10 campaign (gnn_L1_slope / _rich / _occ
# are not re-saved by the rerun).
_NEW = os.path.join(DATA_DIR, "step3_gpu_gnn_right.npz")
if os.path.exists(_NEW):
    _nr = np.load(_NEW, allow_pickle=True)["res"].item()
    for _op in ("PAPER", "FAVORABLE"):
        LONG[_op]["gnn_L1_finals"] = np.asarray(_nr[_op]["gnn_L1_finals"], float)

HERE = os.path.dirname(os.path.abspath(__file__))
POINTS = ["PAPER", "FAVORABLE"]
PRETTY = {"PAPER": "scarce", "FAVORABLE": "clustered,\nresource-rich"}


def per_seed(arr):
    """Collapse replicas within each seed -> (nseed,) per-seed means. Arrays that are already
    one-value-per-seed (e.g. the matched-environment GNN rerun, 24 seeds) pass through unchanged."""
    a = np.asarray(arr, float).ravel()
    if a.size % NSEED == 0:
        return a.reshape(NSEED, -1).mean(axis=1)
    return a


def ci(arr):
    """Mean and bootstrap 95% CI over per-seed means (the project rule: always bootstrap)."""
    x = per_seed(arr)
    rng = np.random.default_rng(0)
    bs = rng.choice(x, (4000, x.size), replace=True).mean(1)
    m = float(x.mean())
    return m, m - float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)) - m


def get(op, key):
    """Fetch a metric key from whichever npz holds it (mappo_* live in the MAPPO run)."""
    src = MAPPO if key.startswith("mappo") else LONG
    return src[op].get(key)


EXPLOIT_THR = 2.0   # occupancy threshold for exploitability (n_i >= THR committed foragers).
# The stored *_exploit arrays use n_i>=0.5; a CV of the per-capita ratio Q_i/n_i is dominated by
# small-denominator outliers from marginally-occupied clumps, which spuriously inflates the swarm's
# dispersion in the clustered regime (CV flips above random only via that tail). THR=2 removes that
# artifact and gives an ordering stable across regimes and across robust measures (CV at 2/4, Gini).


def exploit_cv(op, arm, thr=EXPLOIT_THR):
    """Occupancy-weighted CV of per-capita intake over clumps with n_i>=thr; mean+/-SEM over worlds."""
    src = MAPPO if arm.startswith("mappo") else LONG
    rich = np.asarray(src[op][f"{arm}_rich"], float)
    occ = np.asarray(src[op][f"{arm}_occ"], float)
    vals = []
    for w in range(rich.shape[0]):
        r, o = rich[w], occ[w]
        m = o >= thr
        if m.sum() < 2:
            continue
        v = r[m] / o[m]
        mu = np.average(v, weights=o[m])
        var = np.average((v - mu) ** 2, weights=o[m])
        vals.append(np.sqrt(var) / max(mu, 1e-9))
    vals = np.asarray(vals)
    return vals.mean(), (vals.std(ddof=1) / np.sqrt(vals.size) if vals.size > 1 else 0.0)


# arms: (label, finals-key, slope-key, exploit-key, colour)
ARMS = [
    ("random",          "random",            None,               "random_exploit",      "#9e9e9e"),
    ("swarm\n(0 train)", "swarm",            "swarm_slope",      "swarm_exploit",       "#2ca02c"),
    ("MLP-IPPO",        "mlp_ippo_finals",   "mlp_ippo_slope",   "mlp_ippo_exploit",    "#1f77b4"),
    ("GNN (comm.)",     "gnn_L1_finals",     "gnn_L1_slope",     "gnn_L1_exploit",      "#d62728"),
    ("MAPPO\n(CTDE)",   "mappo_mlp_finals",  "mappo_mlp_slope",  "mappo_mlp_exploit",   "#9467bd"),
]

fig, (axH, axS, axE) = plt.subplots(1, 3, figsize=(15.0, 4.3))
nA = len(ARMS)
gap = 1.3
xg = {pt: i * (nA + gap) for i, pt in enumerate(POINTS)}

# ---- (a) harvest ----
for pt in POINTS:
    base = xg[pt]
    for j, (lab, fk, sk, ek, c) in enumerate(ARMS):
        v = get(pt, fk)
        if v is None:
            continue
        m, lo, hi = ci(v)
        axH.bar(base + j, m, yerr=[[lo], [hi]], capsize=3, color=c, width=0.9,
                label=lab.replace("\n", " ") if pt == POINTS[0] else None)
        axH.text(base + j, m + max(hi, 0.01) + 0.015, f"{m:.2f}", ha="center", va="bottom", fontsize=8)
    sw = ci(get(pt, "swarm"))[0]
    axH.plot([base - 0.5, base + nA - 0.5], [sw, sw], color="#2ca02c", ls="--", lw=1.0, alpha=0.6)
axH.set_xticks([xg[pt] + (nA - 1) / 2 for pt in POINTS])
axH.set_xticklabels([PRETTY[pt] for pt in POINTS], fontsize=9)
axH.set_ylabel("fraction of oracle ceiling")
axH.set_ylim(0, 1.0)
axH.set_title("(a) harvest: reward-free swarm vs trained RL", fontsize=10)
axH.legend(fontsize=7.5, loc="upper left", frameon=False, ncol=1)

# ---- (b) allocation slope (gradient match; ideal = 1) ----
for pt in POINTS:
    base = xg[pt]
    for j, (lab, fk, sk, ek, c) in enumerate(ARMS):
        if sk is None:
            continue
        s = get(pt, sk)
        if s is None:
            continue
        s = float(np.mean(s))
        axS.bar(base + j, s, color=c, width=0.9)
        axS.text(base + j, s + 0.02, f"{s:.2f}", ha="center", va="bottom", fontsize=8)
axS.axhline(1.0, color="k", ls="--", lw=1.0, alpha=0.6)
axS.text(xg[POINTS[-1]] + nA - 0.6, 1.0, "ideal", ha="right", va="bottom", fontsize=8, alpha=0.7)
axS.set_xticks([xg[pt] + (nA - 1) / 2 for pt in POINTS])
axS.set_xticklabels([PRETTY[pt] for pt in POINTS], fontsize=9)
axS.set_ylabel("ideal-free allocation slope")
axS.set_ylim(0, 1.25)
axS.set_title("(b) gradient match: who is unbiased (slope $\\to$1)", fontsize=10)

# ---- (c) exploitability (per-capita intake CV, n_i>=2; lower = tighter; 0 = equal-intake Nash) ----
for pt in POINTS:
    base = xg[pt]
    for j, (lab, fk, sk, ek, c) in enumerate(ARMS):
        stem = fk[:-7] if fk.endswith("_finals") else fk
        m, se = exploit_cv(pt, stem)
        axE.bar(base + j, m, yerr=[[se], [se]], capsize=3, color=c, width=0.9)
        axE.text(base + j, m + se + 0.02, f"{m:.2f}", ha="center", va="bottom", fontsize=8)
axE.set_xticks([xg[pt] + (nA - 1) / 2 for pt in POINTS])
axE.set_xticklabels([PRETTY[pt] for pt in POINTS], fontsize=9)
axE.set_ylabel("exploitability (per-capita intake CV, $n_i\\geq2$)")
axE.set_title("(c) precision: who equalizes per-capita intake", fontsize=10)

fig.tight_layout()
out = os.path.join(HERE, "fig_stronger.pdf")
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print("wrote", out)
