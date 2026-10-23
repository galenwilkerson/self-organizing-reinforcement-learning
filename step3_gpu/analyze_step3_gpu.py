#!/usr/bin/env python3
"""Headline figure + verdict for the GPU step-3 swarm-vs-RL comparison. Reads step3_gpu_full.npz
(dict per operating point: mlp_ippo_finals, gnn_L{1,3,5}_finals/curves/ct, swarm, random, slopes) and makes:
 (a) per-operating-point bars: random < swarm vs learners (MLP-IPPO, GNN L1/L3/L5), fraction-of-oracle ± CI;
 (b) GNN learning curves vs the swarm (0-training) line.
Prints the verdict table: does ANY learner (incl. the communicating GNN) beat the reward-free swarm, and
where. All numbers are fraction-of-oracle (true intake / min(N,Σcap) per tick), seeds = unit of uncertainty.
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
_rd = os.path.normpath(os.path.join(HERE, "..", "data", "step3_gpu"))
DATA = _rd if os.path.isdir(_rd) else "/mnt/newssd/flocking_ltm_data/step3_gpu"
tag = sys.argv[1] if len(sys.argv) > 1 else "full"
d = np.load(os.path.join(DATA, f"step3_gpu_{tag}.npz"), allow_pickle=True)
res = d["res"].item() if "res" in d else {k: d[k] for k in d.files}
POINTS = [p for p in ("PAPER", "FAVORABLE") if p in res]
LS = [1, 3, 5]


def ci(x):
    x = np.asarray(x, float).ravel()
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(0); bs = rng.choice(x, (2000, x.size), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def arm_vals(r):
    arms = [("random", r.get("random")), ("swarm\n(0 train)", r.get("swarm")),
            ("MLP-IPPO", r.get("mlp_ippo_finals"))]
    for L in LS:
        arms.append((f"GNN L{L}", r.get(f"gnn_L{L}_finals")))
    return arms


fig, ax = plt.subplots(1, len(POINTS), figsize=(7.2 * len(POINTS), 4.6), squeeze=False)
cols = {"random": "#7f7f7f", "swarm\n(0 train)": "#2ca02c", "MLP-IPPO": "#1f77b4",
        "GNN L1": "#9ecae1", "GNN L3": "#4292c6", "GNN L5": "#084594"}
print("\n==== STEP-3 GPU verdict: fraction of oracle (mean [95% CI]) ====")
for pi, pt in enumerate(POINTS):
    r = res[pt]; a = ax[0, pi]; arms = arm_vals(r)
    labels, means, los, his = [], [], [], []
    print(f"\n-- {pt} --")
    for name, v in arms:
        if v is None:
            continue
        m, lo, hi = ci(v); labels.append(name); means.append(m); los.append(m - lo); his.append(hi - m)
        print(f"  {name.replace(chr(10),' '):16s} {m:.3f}  [{lo:.3f}, {hi:.3f}]")
    sw = ci(r.get("swarm"))[0] if r.get("swarm") is not None else np.nan
    x = np.arange(len(labels))
    a.bar(x, means, yerr=[los, his], capsize=4,
          color=[cols.get(l, "#333") for l in labels])
    if not np.isnan(sw):
        a.axhline(sw, color="#2ca02c", ls="--", lw=1, alpha=0.6)
    a.set_xticks(x); a.set_xticklabels([l for l in labels], fontsize=8)
    a.set_ylabel("fraction of oracle"); a.set_ylim(0, 1.0); a.set_title(pt, fontsize=11)
    for xi, m in zip(x, means):
        a.text(xi, m + 0.02, f"{m:.2f}", ha="center", fontsize=8)
    # verdict line
    best_learn = max([ci(r.get(k))[0] for k in (["mlp_ippo_finals"] + [f"gnn_L{L}_finals" for L in LS])
                      if r.get(k) is not None] + [np.nan])
    verdict = ("learner > swarm" if best_learn > sw + 0.02 else
               "learner ≈ swarm" if abs(best_learn - sw) <= 0.02 else "learner < swarm")
    print(f"  => best learner {best_learn:.3f} vs swarm {sw:.3f}: {verdict}")
fig.suptitle("Step 3 (GPU, F1 fair primitives): reward-free swarm vs trained RL (feedforward + communicating GNN)",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = os.path.join(HERE, f"step3_gpu_{tag}.png"); fig.savefig(out, dpi=140)
print(f"\nwrote {out}")
