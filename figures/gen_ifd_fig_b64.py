#!/usr/bin/env python3
"""
End-state Ideal-Free-Distribution matching figure at B=64 -> fig_ifd.pdf (replaces the B=16 version).

Same experiment/config as Code/ifd_test.py on Code/mobile_cascade.py (coupling ON = converge=True,
OFF = converge=False; Pareto clump_beta; committed foragers = ACTIVE agents assigned to nearest clump;
richness Q_i = sources per clump). End-state method = the paper's ~400-tick tail average of the
committed-forager occupancy, then a log-log regression of committed foragers per clump on richness.

Reuses the per-tick occupancy cache already produced for the convergence figure (B=64, T=800, seed=0)
at /mnt/newssd/flocking_ltm_data/paper2_convergence/ ; no re-simulation. CPU only.

Usage: python3 gen_ifd_fig_b64.py
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11.5})

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/mnt/newssd/flocking_ltm_data/paper2_convergence"
CACHE = os.path.join(DATA, "convergence_B64_T800_seed0.npz")
TAIL_START = 400          # tail-average window start (ticks); matches ifd_test.py T//2 with T=800

d = np.load(CACHE)


def tail_occ(cond):
    """(B,C) committed-forager occupancy averaged over ticks [TAIL_START:], matching ifd_test's tail."""
    return d[f"occ_{cond}"][TAIL_START:].mean(axis=0)


def fit(rich, occ):
    """log10 scatter + linear fit, same masking as Code/gen_ifd_fig.py (rich>=1, occ>=0.5)."""
    r, o = rich.flatten(), occ.flatten()
    m = (r >= 1) & (o >= 0.5)
    x, y = np.log10(r[m]), np.log10(o[m])
    A = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - np.polyval(A, x)) ** 2) / np.sum((y - y.mean()) ** 2)
    return x, y, A, r2, int(m.sum())


fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
report = {}
for k, (cond, col, title) in enumerate([
        ("on", "#2ca02c", "recruiting (coupling on)"),
        ("off", "#c0392b", "ablated (coupling off)")]):
    rich = d[f"rich_{cond}"]
    occ = tail_occ(cond)
    x, y, A, r2, n = fit(rich, occ)
    ax[k].scatter(x, y, s=12, color=col, alpha=0.45, edgecolors="none")
    xs = np.linspace(x.min(), x.max(), 50)
    ax[k].plot(xs, np.polyval(A, xs), color=col, lw=2, label=f"slope {A[0]:.2f}")
    ax[k].plot(xs, xs + (y.mean() - x.mean()), "k--", lw=1, label="IFD (slope 1)")
    ax[k].set_title(f"({'ab'[k]})", loc="left", fontweight="bold", fontsize=13)
    ax[k].set_xlabel(r"$\log_{10}$ patch richness (sources/clump)")
    ax[k].legend(fontsize=11.5, loc="upper left", frameon=False)
    report[cond] = (A[0], r2, n)
    print(f"{title}: slope={A[0]:.3f}  R^2={r2:.3f}  (n={n} clump-replica points)")

ax[0].set_ylabel(r"$\log_{10}$ committed foragers at clump")
fig.tight_layout()
pdf = os.path.join(HERE, "fig_ifd.pdf")
png = os.path.join(HERE, "fig_ifd.png")
fig.savefig(pdf); fig.savefig(png, dpi=200)
print("wrote", pdf)
print("wrote", png)
print(f"\nB=64, tail average ticks [{TAIL_START}:800], seed=0")
print(f"ON  slope={report['on'][0]:.3f}  R^2={report['on'][1]:.3f}")
print(f"OFF slope={report['off'][0]:.3f}  R^2={report['off'][1]:.3f}")
