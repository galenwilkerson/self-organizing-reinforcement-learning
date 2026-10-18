#!/usr/bin/env python3
"""
Discrimination Exp 2 -- blocking ablation (DISCRIMINATION_EXPERIMENT_DESIGN.md Exp 2).

Question: is slope-1 (proportional / IFD) matching the specific signature of the congestion cap
(BLOCKING), or would generic aggregation give it anyway?

Conditions (identical fields; B=64 replicas; same config as Code/ifd_test.py):
  (a) FULL           recruitment + blocking      (converge=True, block_off=False)   expect slope ~0.94
  (b) BLOCKING OFF   recruitment ON, cap removed (converge=True, block_off=True)    expect over-aggregation

Surgical toggle: the ONLY change is the cascade activation rule. Canonical uses the degree-normalised
fraction rule  newly = (a/k >= theta)  -- the congestion cap (as a patch crowds, degree k rises, a/k
falls below threshold, recruitment self-limits). Blocking-OFF replaces it with an ABSOLUTE-count rule
newly = (a >= block_m)  so recruitment never shuts off as a patch crowds. Recruitment, motion, seeding,
fields are otherwise UNCHANGED. The canonical model file is not edited: we subclass and patch step() via
inspect.getsource. Sanity-checks the active fraction stays bounded (no simulation runaway).

Metric: IFD matching slope + distance-to-IFD (total variation) per condition, bootstrap CI over replicas.
CPU. Raw data -> /mnt/newssd/flocking_ltm_data/paper2_discrim/. Figure -> figures/fig_discrim_blocking.
"""
from __future__ import annotations
import os, sys, time, inspect, textwrap
import numpy as np

CODE = "/home/galen/Dropbox_2025/Dropbox/projects/flocking_LTM/Code"
sys.path.insert(0, CODE)
import math  # noqa: E402
import torch  # noqa: E402
from mobile_cascade import MobileCascade, MobileCascadeConfig  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/mnt/newssd/flocking_ltm_data/paper2_discrim"
os.makedirs(DATA, exist_ok=True)


def build_ablatable():
    """Subclass of MobileCascade whose step() has a surgical blocking toggle (the one activation line)."""
    src = inspect.getsource(MobileCascade.step)
    src = textwrap.dedent(src)
    old = "newly = (a * inv_deg >= self.theta_vec) & (~active)"
    new = ("newly = ((a >= self._block_m) if getattr(self, '_block_off', False) "
           "else (a * inv_deg >= self.theta_vec)) & (~active)")
    assert old in src, "canonical activation line not found -- model changed; re-check patch"
    src = src.replace(old, new)
    ns = {"torch": torch, "math": math}
    exec(src, ns)
    Ablatable = type("AblatableCascade", (MobileCascade,), {"step": ns["step"]})
    return Ablatable


Ablatable = build_ablatable()


def run(block_off, B=64, T=800, seed=0, block_m=1.0):
    cfg = MobileCascadeConfig(B=B, N=200, K=80, L=800.0, rho_r=2.1, phi=0.02, theta=0.18,
                              levy_alpha=1.0, clump_count=10, clump_beta=2.2, clump_radius=30.0,
                              capacity=40.0, consumption=1.0, step=20.0, pull=0.12,
                              converge=True, device="cpu", seed=seed)
    sim = Ablatable(cfg)
    sim._block_off = block_off
    sim._block_m = block_m
    C = sim._n_clumps
    idx = sim._clump_idx
    rich = torch.zeros(sim.B, C); rich.scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float32))
    occ = torch.zeros(sim.B, C); nmeas = 0
    actfrac = []
    for t in range(T):
        sim.step()
        actfrac.append(sim._active_state.float().mean().item())
        if t >= T // 2:
            active = sim._active_state
            d = sim._pair_dist(sim.pos, sim.clump_centers)
            near = d.argmin(dim=2)
            o = torch.zeros(sim.B, C); o.scatter_add_(1, near, active.to(torch.float32))
            occ += o; nmeas += 1
    occ /= max(nmeas, 1)
    return rich.numpy(), occ.numpy(), np.array(actfrac)


def slope_pooled(rich, occ, idx=None):
    r = (rich[idx] if idx is not None else rich).flatten()
    o = (occ[idx] if idx is not None else occ).flatten()
    m = (r >= 1) & (o >= 0.5)
    if m.sum() < 8: return np.nan
    return np.polyfit(np.log(r[m]), np.log(o[m]), 1)[0]


def tv_to_ifd(rich, occ):
    q = rich / rich.sum(1, keepdims=True).clip(min=1e-9)
    s = occ.sum(1, keepdims=True)
    p = np.where(s > 0, occ / s.clip(min=1e-9), 0.0)
    return 0.5 * np.abs(p - q).sum(1)


def main(B=64, T=800):
    cache = os.path.join(DATA, f"exp2_blocking_B{B}_T{T}.npz")
    if os.path.exists(cache) and os.environ.get("FORCE", "") != "1":
        print(f"loading {cache}"); z = np.load(cache)
        res = {k: z[k] for k in z.files}
    else:
        t0 = time.time(); res = {}
        for lab, bo in [("full", False), ("blockoff", True)]:
            r, o, af = run(bo, B=B, T=T, seed=0)
            res[f"rich_{lab}"] = r; res[f"occ_{lab}"] = o; res[f"actfrac_{lab}"] = af
            print(f"  {lab} done ({time.time()-t0:.0f}s), mean active frac tail="
                  f"{af[T//2:].mean():.3f} (max {af.max():.3f})")
        np.savez_compressed(cache, **res); print(f"wrote {cache}")

    rng = np.random.default_rng(1)
    draws = rng.integers(0, B, size=(2000, B))
    print("\n=== Exp 2 blocking ablation (B=%d) ===" % B)
    summary = {}
    for lab, name in [("full", "full (recruit+block)"), ("blockoff", "blocking OFF")]:
        rich, occ = res[f"rich_{lab}"], res[f"occ_{lab}"]
        s = slope_pooled(rich, occ)
        bs = np.array([slope_pooled(rich, occ, draws[b]) for b in range(0, 2000, 4)])
        bs = bs[np.isfinite(bs)]
        tv = tv_to_ifd(rich, occ); tvb = tv[draws].mean(1)
        af = res[f"actfrac_{lab}"]
        summary[lab] = dict(slope=s, slo=np.percentile(bs, 2.5), shi=np.percentile(bs, 97.5),
                            tv=tv.mean(), tvlo=np.percentile(tvb, 2.5), tvhi=np.percentile(tvb, 97.5),
                            af=af[T//2:].mean())
        print(f"  {name:22s}: slope={s:.3f} [{summary[lab]['slo']:.3f},{summary[lab]['shi']:.3f}]  "
              f"TV-to-IFD={tv.mean():.3f} [{np.percentile(tvb,2.5):.3f},{np.percentile(tvb,97.5):.3f}]  "
              f"active-frac={summary[lab]['af']:.3f}")
    make_figure(summary)
    return summary


def make_figure(summary):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "legend.fontsize": 10})
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.2))
    labs = ["full", "blockoff"]
    names = ["full\n(recruit+block)", "blocking OFF\n(recruit only)"]
    cols = ["#2ca02c", "#c0392b"]
    # slope
    s = [summary[l]["slope"] for l in labs]
    slo = [summary[l]["slope"] - summary[l]["slo"] for l in labs]
    shi = [summary[l]["shi"] - summary[l]["slope"] for l in labs]
    ax[0].bar(range(2), s, yerr=[slo, shi], capsize=4, color=cols)
    ax[0].axhline(1.0, ls="--", color="k", lw=1); ax[0].text(1.4, 1.01, "IFD (slope 1)", fontsize=9, ha="right")
    ax[0].set_xticks(range(2)); ax[0].set_xticklabels(names)
    ax[0].set_ylabel("IFD matching slope"); ax[0].set_title("(a) proportional matching", fontsize=11)
    # TV
    tv = [summary[l]["tv"] for l in labs]
    tlo = [summary[l]["tv"] - summary[l]["tvlo"] for l in labs]
    thi = [summary[l]["tvhi"] - summary[l]["tv"] for l in labs]
    ax[1].bar(range(2), tv, yerr=[tlo, thi], capsize=4, color=cols)
    ax[1].set_xticks(range(2)); ax[1].set_xticklabels(names)
    ax[1].set_ylabel("distance to IFD (total variation)"); ax[1].set_title("(b) distance from IFD", fontsize=11)
    af_full = summary["full"]["af"]; af_bo = summary["blockoff"]["af"]
    fig.suptitle("Exp 2 — proportional (IFD) matching SURVIVES removing the congestion cap "
                 f"(active frac {af_full:.2f}$\\to${af_bo:.2f}, no runaway)", fontsize=10.8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("pdf", "png"):
        p = os.path.join(HERE, f"fig_discrim_blocking.{ext}")
        fig.savefig(p, dpi=200 if ext == "png" else None); print("wrote", p)


if __name__ == "__main__":
    import sys as _s
    if len(_s.argv) > 1 and _s.argv[1] == "smoke":
        r, o, af = run(False, B=8, T=100); print("full smoke slope", slope_pooled(r, o), "af", af[-10:].mean())
        r, o, af = run(True, B=8, T=100); print("blockoff smoke slope", slope_pooled(r, o), "af", af[-10:].mean())
    else:
        main()
