#!/usr/bin/env python3
"""Robustness appendix for Paper 2 (v5): the swarm's near-ideal allocation is not fine-tuned.

From a 1024-point Sobol sweep over 7 dimensionless controls (data: dimensionless_sweep/sweep_v1.npz; proper
multi-hop model, B=64 replica worlds/point). We show the IFD matching slope (gradient faithfulness) against the
three controls the reader worries about: the cascade activation threshold phi (code name `theta`), the Levy
exponent alpha, and the resource patchiness. Mapping code->paper symbols: code `theta` = paper phi (threshold);
code `phi` = paper rho_s (coverage). The swarm's two OWN controls (phi, alpha) sit on broad plateaus (near-ideal
slope across wide ranges => not tuned); patchiness is a property of the RESOURCE FIELD, and it is the one axis
the gradient match depends on (honest limitation). Binned median + IQR; operating point marked.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
_RD = os.path.normpath(os.path.join(HERE, "..", "data", "dimensionless_sweep"))
SW = (os.path.join(_RD, "sweep_v1.npz") if os.path.isdir(_RD)
      else "/mnt/newssd/flocking_ltm_data/dimensionless_sweep/sweep_v1.npz")
d = np.load(SW, allow_pickle=True)
axes = list(d["axes"]); P = d["points"]; recs = d["records"]
ix = {a: i for i, a in enumerate(axes)}
def col(k): return np.array([r.get(k, np.nan) for r in recs], float)
slope = col("slope"); frac = col("frac_oracle")
usable = np.isfinite(slope) & np.isfinite(frac) & (frac >= 0) & (frac <= 1.2)

def ax(a): return P[:, ix[a]]
# favorable resource field (so we isolate the swarm's own controls); and reasonable swarm controls
field_fav = usable & (ax("patchiness") <= 0.09) & (ax("phi") >= 0.015) & (ax("clump_beta") <= 3.0)
swarm_ok = usable & (ax("alpha") <= 1.4) & (ax("theta") >= 0.08) & (ax("theta") <= 0.30) & (ax("phi") >= 0.015)

def binned(mask, axis_name, nb=7):
    x = ax(axis_name)[mask]; y = slope[mask]
    edges = np.linspace(x.min(), x.max(), nb + 1)
    cx, med, lo, hi, nct = [], [], [], [], []
    for b in range(nb):
        m = (x >= edges[b]) & (x <= edges[b + 1] if b == nb - 1 else x < edges[b + 1])
        if m.sum() < 3:
            continue
        cx.append(0.5 * (edges[b] + edges[b + 1]))
        med.append(np.median(y[m])); lo.append(np.percentile(y[m], 25)); hi.append(np.percentile(y[m], 75))
        nct.append(m.sum())
    return map(np.array, (cx, med, lo, hi, nct))

PAN = [
    ("theta",      field_fav, r"cascade threshold $\phi$",      0.18,  "swarm control: broad plateau"),
    ("alpha",      field_fav, "Lévy exponent $\\alpha$",   1.0,   "swarm control: broad plateau"),
    ("patchiness", swarm_ok,  "resource patchiness",            0.0375,"field property: slope depends on it"),
]
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.1))
for k, (name, mask, xlab, opv, sub) in enumerate(PAN):
    a = axs[k]
    cx, med, lo, hi, nct = binned(mask, name)
    a.fill_between(cx, lo, hi, color="#2ca02c", alpha=0.18, linewidth=0)
    a.plot(cx, med, "-o", color="#2ca02c", lw=2, ms=4, label="swarm (binned median, IQR)")
    a.axhline(1.0, color="k", ls="--", lw=1.0, alpha=0.6)
    a.text(cx[-1], 1.01, "ideal", ha="right", va="bottom", fontsize=8, alpha=0.7)
    a.axvline(opv, color="#d62728", ls=":", lw=1.4)
    a.text(opv, 0.03, " operating\n point", color="#d62728", fontsize=7.5, va="bottom", ha="left",
           transform=a.get_xaxis_transform())
    a.set_xlabel(xlab); a.set_ylim(0, 1.25)
    if k == 0: a.set_ylabel("ideal-free allocation slope")
    a.set_title(f"({'abc'[k]}) {sub}", fontsize=9.5)
fig.suptitle("The swarm's near-ideal allocation is not fine-tuned: its own controls "
             r"($\phi,\alpha$) sit on broad plateaus; only resource patchiness sets the gradient match",
             fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = os.path.join(HERE, "fig_robustness.pdf")
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print("wrote", out)
# report subset sizes
print("field_fav n=%d  swarm_ok n=%d  usable n=%d" % (field_fav.sum(), swarm_ok.sum(), usable.sum()))
