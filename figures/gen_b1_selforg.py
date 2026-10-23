#!/usr/bin/env python3
"""
Exp B1 -- self-organization / cascade-window sweep (EXPERIMENT_DESIGN_B.md B1; referee M2).

Goal: show IN THIS PAPER that the cascade window exists and the swarm self-regulates across connectivity
conditions without per-condition tuning.

Two measurements (CPU, scratch driver, canonical Code/ untouched):
 (A) THE WINDOW -- single-seed Watts cascade on a uniform RGG as a function of mean degree z. One random
     seed, linear-threshold rule (activate if active-neighbour FRACTION >= theta), run to fixed point;
     final active fraction vs z. Classic inverted-U peaking between percolation onset z_c^RGG=4.51 and the
     blocking bound 2/theta (theta=0.18 -> 11.1). Reproduces Part I's window in Part II's own runs.
 (B) THE SWARM -- the full mobile model (Code/mobile_cascade.py) swept over connectivity rho_r at the
     paper's phi/theta. Per condition (steady state): operating mean degree z_global, branching ratio
     sigma, active fraction, cascade-size fraction, and the IFD matching slope.

HONEST framing (see RESULTS_B.md): on the mobile DEPLETING model the swarm AGGREGATES, so its bulk
operating degree is ABOVE the single-seed window (clustering inflates degree) and the multi-seed cascade
fraction is monotone, NOT inverted-U. The self-organization signature that DOES hold without tuning is the
density-invariant SUBCRITICAL branching ratio (sigma ~ 0.07-0.10 across a 60x density range) together with
near-ideal IFD matching -- i.e. criticality is LOCAL (per the paper's own framing), not bulk-degree-in-window.

B>=32 replicas, bootstrap CIs. Data -> /mnt/newssd/.../paper2_selforg/.
"""
from __future__ import annotations
import os, sys, time, math
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.normpath(os.path.join(_HERE, "..", "code"))
if not os.path.isdir(CODE):
    CODE = "/home/galen/Dropbox_2025/Dropbox/projects/flocking_LTM/Code"
sys.path.insert(0, CODE)
import torch  # noqa: E402
from mobile_cascade import MobileCascade, MobileCascadeConfig  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_RD = os.path.normpath(os.path.join(HERE, "..", "data", "paper2_selforg"))
DATA = _RD if os.path.isdir(_RD) else "/mnt/newssd/flocking_ltm_data/paper2_selforg"
os.makedirs(DATA, exist_ok=True)

N = 200
THETA = 0.18
Z_PERC = 4.51
Z_BLOCK = 2.0 / THETA            # 11.11


# ---------------- (A) single-seed Watts cascade window ----------------
@torch.no_grad()
def single_seed_window(zs, B=400, Nn=400, seed=0):
    """Final active fraction of a 1-seed linear-threshold cascade on a uniform RGG, vs mean degree z."""
    g = torch.Generator().manual_seed(seed)
    L = 1.0
    out = []
    for z in zs:
        r = math.sqrt(z / ((Nn - 1) * math.pi)) * L           # mean degree z = (N-1) pi r^2 / L^2
        pos = torch.rand(B, Nn, 2, generator=g)
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)
        diff = diff - torch.round(diff)                        # torus min-image (L=1)
        d = diff.norm(dim=-1)
        adj = (d <= r) & (~torch.eye(Nn, dtype=torch.bool).unsqueeze(0))
        deg = adj.sum(2).clamp(min=1).float()
        active = torch.zeros(B, Nn, dtype=torch.bool)
        s = torch.randint(0, Nn, (B,), generator=g)
        active[torch.arange(B), s] = True
        for _ in range(Nn):
            a = (adj & active.unsqueeze(1)).sum(2).float()
            newly = (a / deg >= THETA) & (~active)
            if not bool(newly.any()):
                break
            active = active | newly
        out.append(active.float().mean().item())
    return np.array(out)


# ---------------- (B) swarm connectivity sweep ----------------
@torch.no_grad()
def run_swarm(rho_r, B, T, seed=0):
    cfg = MobileCascadeConfig(B=B, N=N, K=80, L=800.0, rho_r=rho_r, phi=0.02, theta=THETA,
                              levy_alpha=1.0, clump_count=10, clump_beta=2.2, clump_radius=30.0,
                              capacity=40.0, consumption=1.0, step=20.0, pull=0.12,
                              converge=True, device="cpu", seed=seed)
    sim = MobileCascade(cfg)
    C = sim._n_clumps
    idx = sim._clump_idx
    rich = torch.zeros(sim.B, C); rich.scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float32))
    t0 = T // 2
    zg = []; sg = []; af = []; cf = []
    occ = torch.zeros(sim.B, C); nmeas = 0
    for t in range(T):
        sim.step()
        if t >= t0:
            zg.append(sim._z_global.clone()); sg.append(sim._sigma.clone())
            af.append(sim._active_count.clone()); cf.append(sim._cascade_size.clone())
            active = sim._active_state
            d = sim._pair_dist(sim.pos, sim.clump_centers); near = d.argmin(dim=2)
            o = torch.zeros(sim.B, C); o.scatter_add_(1, near, active.to(torch.float32))
            occ += o; nmeas += 1
    occ /= max(nmeas, 1)
    return dict(rich=rich.numpy(), occ=occ.numpy(),
                zg=torch.stack(zg, 1).mean(1).numpy(), sg=torch.stack(sg, 1).mean(1).numpy(),
                af=torch.stack(af, 1).mean(1).numpy() / N, cf=torch.stack(cf, 1).mean(1).numpy() / N)


def slope_pooled(rich, occ, idx=None):
    r = (rich[idx] if idx is not None else rich).flatten()
    o = (occ[idx] if idx is not None else occ).flatten()
    m = (r >= 1) & (o >= 0.5)
    return np.polyfit(np.log(r[m]), np.log(o[m]), 1)[0] if m.sum() >= 8 else np.nan


def main(B=48, T=600):
    rhos = np.array([0.3, 0.5, 0.8, 1.2, 1.7, 2.1, 2.8, 3.6, 5.0, 7.0, 10.0])
    cache = os.path.join(DATA, f"b1_B{B}_T{T}.npz")
    zs_win = np.geomspace(1.0, 25.0, 30)
    if os.path.exists(cache) and os.environ.get("FORCE", "") != "1":
        print(f"loading {cache}"); z = np.load(cache, allow_pickle=True)
        R = z["res"].item(); win = z["win"]
    else:
        t0 = time.time()
        print("single-seed window ...")
        win = single_seed_window(zs_win)
        R = {}
        for rr in rhos:
            R[float(rr)] = run_swarm(float(rr), B, T)
            d = R[float(rr)]
            print(f"  rho_r={rr:5.2f}: z_global={d['zg'].mean():6.2f} sigma={d['sg'].mean():.3f} "
                  f"active={d['af'].mean():.3f} casc={d['cf'].mean():.3f} "
                  f"slope={slope_pooled(d['rich'],d['occ']):.3f} ({time.time()-t0:.0f}s)")
        np.savez_compressed(cache, res=R, win=win, zs_win=zs_win); print(f"wrote {cache}")

    rng = np.random.default_rng(0); draws = rng.integers(0, B, size=(1500, B))
    rows = []
    for rr in rhos:
        d = R[float(rr)]
        s = slope_pooled(d["rich"], d["occ"])
        bs = np.array([slope_pooled(d["rich"], d["occ"], draws[b]) for b in range(0, 1500, 5)]); bs = bs[np.isfinite(bs)]
        zb = d["zg"][draws].mean(1); sb = d["sg"][draws].mean(1)
        rows.append(dict(rho=float(rr), z=d["zg"].mean(), zlo=np.percentile(zb, 2.5), zhi=np.percentile(zb, 97.5),
                         sig=d["sg"].mean(), siglo=np.percentile(sb, 2.5), sighi=np.percentile(sb, 97.5),
                         af=d["af"].mean(), cf=d["cf"].mean(),
                         slope=s, slo=np.percentile(bs, 2.5), shi=np.percentile(bs, 97.5)))
    peak_z = zs_win[np.argmax(win)]
    print("\n=== B1 verdict numbers ===")
    print(f"(A) single-seed window: cascade peaks at z={peak_z:.1f}; window [{Z_PERC},{Z_BLOCK:.1f}]; "
          f"peak frac={win.max():.2f}")
    op = [r for r in rows if abs(r['rho'] - 2.1) < 1e-9][0]
    print(f"(B) swarm operating z_global at paper rho_r=2.1 = {op['z']:.1f}  (ABOVE window top {Z_BLOCK:.1f}: "
          f"{'yes' if op['z']>Z_BLOCK else 'no'})")
    sigs = np.array([r['sig'] for r in rows]); slopes = np.array([r['slope'] for r in rows])
    print(f"    sigma across rho_r in [{rhos.min()},{rhos.max()}] (60x density): "
          f"{sigs.min():.3f}..{sigs.max():.3f} (flat, subcritical)")
    print(f"    IFD slope across sweep: {slopes.min():.3f}..{slopes.max():.3f}")
    make_figure(zs_win, win, peak_z, rows)
    return rows


def make_figure(zs_win, win, peak_z, rows):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11.5, "axes.labelsize": 11.5, "legend.fontsize": 9})
    rho = np.array([r["rho"] for r in rows]); z = np.array([r["z"] for r in rows])
    sig = np.array([r["sig"] for r in rows]); siglo = np.array([r["siglo"] for r in rows]); sighi = np.array([r["sighi"] for r in rows])
    sl = np.array([r["slope"] for r in rows]); slo = np.array([r["slo"] for r in rows]); shi = np.array([r["shi"] for r in rows])

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.1))
    # (a) single-seed window
    a0 = ax[0]
    a0.axvspan(Z_PERC, Z_BLOCK, color="gold", alpha=0.2, label="cascade window")
    a0.axvline(Z_PERC, color="goldenrod", ls=":", lw=1); a0.axvline(Z_BLOCK, color="goldenrod", ls=":", lw=1)
    a0.plot(zs_win, win, "-o", color="#2c7fb8", ms=3, lw=2)
    a0.scatter([peak_z], [win.max()], s=90, color="#d62728", marker="*", zorder=6, label=f"peak z={peak_z:.1f}")
    a0.set_xscale("log"); a0.set_xlabel("RGG mean degree  $z$")
    a0.set_ylabel("single-seed cascade fraction")
    a0.text(Z_PERC, -0.09, "$z_c^{RGG}$", fontsize=9, ha="center"); a0.text(Z_BLOCK, -0.09, "$2/\\theta$", fontsize=9, ha="center")
    a0.set_title("(a) window exists (single-seed Watts)", fontsize=10.8); a0.legend(fontsize=8.5)
    # (b) swarm operating degree vs connectivity, window band
    a1 = ax[1]
    a1.axhspan(Z_PERC, Z_BLOCK, color="gold", alpha=0.2, label="window $[z_c^{RGG},2/\\theta]$")
    a1.plot(rho, z, "-o", color="#8e44ad", lw=2, label="swarm operating $z_{global}$")
    a1.fill_between(rho, [r["zlo"] for r in rows], [r["zhi"] for r in rows], color="#8e44ad", alpha=0.2, lw=0)
    a1.axvline(2.1, color="#d62728", ls=":", lw=1.2); a1.text(2.15, z.max()*0.9, "paper\n$\\rho_r$=2.1", color="#d62728", fontsize=8.5)
    a1.set_xlabel("connectivity  $\\rho_r$"); a1.set_ylabel("operating mean degree $z_{global}$")
    a1.set_yscale("log")
    a1.set_title("(b) swarm AGGREGATES above the window\n(criticality is local, not bulk)", fontsize=10.2)
    a1.legend(fontsize=8.5, loc="lower right")
    # (c) sigma (flat subcritical) + IFD slope vs connectivity
    a2 = ax[2]
    a2.plot(rho, sig, "-o", color="#e67e22", lw=2, label="branching ratio $\\sigma$")
    a2.fill_between(rho, siglo, sighi, color="#e67e22", alpha=0.2, lw=0)
    a2.set_xlabel("connectivity  $\\rho_r$"); a2.set_ylabel("branching ratio $\\sigma$", color="#e67e22")
    a2.set_ylim(0, 0.2); a2.tick_params(axis="y", labelcolor="#e67e22")
    a2b = a2.twinx()
    a2b.plot(rho, sl, "-s", color="#2ca02c", lw=2, label="IFD slope")
    a2b.fill_between(rho, slo, shi, color="#2ca02c", alpha=0.18, lw=0)
    a2b.axhline(1.0, color="k", ls="--", lw=0.8); a2b.set_ylabel("IFD matching slope", color="#2ca02c")
    a2b.set_ylim(0, 1.15); a2b.tick_params(axis="y", labelcolor="#2ca02c")
    a2.set_title("(c) $\\sigma$ density-invariant & subcritical;\nmatching near-ideal (no tuning)", fontsize=10.2)
    a2.set_xscale("log")
    fig.suptitle("The cascade window (a); the mobile swarm self-regulates to a stable subcritical "
                 "operating regime across 60$\\times$ density (b,c)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("pdf", "png"):
        p = os.path.join(HERE, f"fig_b1_selforg.{ext}")
        fig.savefig(p, dpi=200 if ext == "png" else None); print("wrote", p)


if __name__ == "__main__":
    import sys as _s
    if len(_s.argv) > 1 and _s.argv[1] == "win":
        zs = np.geomspace(1, 25, 20); w = single_seed_window(zs, B=200, Nn=300)
        for zz, ww in zip(zs, w): print(f"z={zz:5.2f} cascade_frac={ww:.3f}")
    else:
        main()
