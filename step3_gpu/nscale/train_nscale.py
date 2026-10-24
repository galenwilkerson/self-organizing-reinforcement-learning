#!/usr/bin/env python3
"""TMLR strengthening: N-scaling of the learner-free swarm (TRUE finite-size sweep).

Addresses the reviewers' "N=200 is fixed" point by growing the system size while holding EVERY dimensionless
control fixed, so the sweep isolates finite-size scaling and does not secretly vary a dimensionless group.

The model has several length scales: arena L, interaction radius r, Levy step STEP, exploit step, clump radius,
source-sense radius S. A naive "vary N, shrink r to hold rho_r" sweep in a FIXED arena leaves STEP, clump_radius
fixed while r shrinks, so STEP/r and clump_radius/r drift ~sqrt(N) -- a hidden parameter sweep. Instead we grow
the ARENA and scale the counts with area, holding all microscopic lengths fixed:
  L = L0 sqrt(N/N0);  K = 0.4 N;  C = 0.05 N (round, >=3);  patchiness = patchiness0 sqrt(N0/N).
Consequences (all dimensionless groups fixed, only absolute size grows):
  r = R_NEIGH = L sqrt(rho_r/N) = L0 sqrt(rho_r/N0) = const  ->  rho_r fixed AND r fixed -> STEP/r fixed.
  clump_radius = patchiness*L = patchiness0*L0 = const        ->  clump_radius/r fixed.
  S = L sqrt(phi/K) = const                                   ->  S/r fixed, sensing group phi fixed.
  N/L^2, K/L^2, C/L^2 fixed (densities);  K/C = const (sources per clump -> richness granularity fixed).
If the ideal-free slope stays ~1 across this sweep, the result is genuinely scale-free; if it drifts, that is a
true finite-size effect (not a disguised parameter change).

One process per N (N,K,C,L baked at import via env vars). Metrics per field: harvest (frac of oracle), IFD slope,
exploitability at n_i>={0.5,2,4}. Prints length-ratio diagnostics to CONFIRM invariance. Output -> nscale_N<n>.npz.
"""
import os, sys, time, math, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)
sys.path.insert(0, HERE)
import torch_env as TE
from train_long import eval_sr, PAPER, FAV

DATA = os.environ.get("FLOCKING_DATA") or "/mnt/newssd/flocking_ltm_data/step3_gpu/nscale"
os.makedirs(DATA, exist_ok=True)
THRESHOLDS = (0.5, 2.0, 4.0)
N0 = 200  # reference scale (the paper's N); patchiness scaled by sqrt(N0/N) to hold clump_radius fixed


def exploit_thresh(rich, occ, thr):
    """Occupancy-weighted CV of per-capita intake over patches with n_i >= thr (0 at the equal-intake Nash)."""
    rich = np.asarray(rich, float); occ = np.asarray(occ, float)
    occupied = occ >= thr
    v = rich / np.clip(occ, 1e-9, None)
    w = occ * occupied
    wsum = w.sum(1)
    mu = (w * v).sum(1) / np.clip(wsum, 1e-9, None)
    var = (w * (v - mu[:, None]) ** 2).sum(1) / np.clip(wsum, 1e-9, None)
    cv = np.sqrt(np.clip(var, 0.0, None)) / np.clip(mu, 1e-9, None)
    return np.where(wsum > 0, cv, 0.0)


def eval_oom(controller, W, eval_ticks, window, op, seed, floor=64):
    while True:
        try:
            return eval_sr(controller, W, eval_ticks, window, op, seed), W
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if W <= floor:
                raise
            W = max(floor, W // 2)


def main(W, eval_ticks, window, base):
    n, k, c, L = TE.N, TE.K, TE.C, TE.L
    r = TE.R_NEIGH
    t0 = time.time()
    pscale = math.sqrt(N0 / n)  # patchiness *= pscale  ->  clump_radius = patchiness*L stays fixed
    cr_paper = PAPER["patchiness"] * pscale * L
    print(f"N-scaling (finite-size): N={n} K={k} C={c} L={L:.0f} r={r:.2f} | invariants: STEP/r={TE.STEP/r:.3f} "
          f"EXPL/r={TE.EXPLOIT_STEP/r:.3f} clump_r/r(PAPER)={cr_paper/r:.3f} K/C={k/c:.2f} N/L^2={n/L**2*1e4:.3f}e-4",
          flush=True)
    out = {"N": n, "K": k, "C": c, "L": float(L),
           "_meta": dict(W_req=W, eval_ticks=eval_ticks, window=window, rho_r=TE.RHO_R, R_NEIGH=float(r),
                         pscale=pscale, thresholds=list(THRESHOLDS), STEP_over_r=TE.STEP / r)}
    for name, op in (("PAPER", PAPER), ("FAVORABLE", FAV)):
        op2 = dict(op); op2["patchiness"] = op["patchiness"] * pscale   # hold clump_radius fixed as L grows
        (sf, ss, srich, socc), Wu = eval_oom("swarm", W, eval_ticks, window, op2, base + 555)
        (rf, rs, _, _), _ = eval_oom("random", Wu, eval_ticks, window, op2, base + 555)
        sx = {thr: float(np.mean(exploit_thresh(srich, socc, thr))) for thr in THRESHOLDS}
        out[name] = dict(swarm_harvest=float(np.mean(sf)), swarm_harvest_sd=float(np.std(sf)),
                         swarm_slope=float(ss), swarm_exploit=sx, random_harvest=float(np.mean(rf)),
                         random_slope=float(rs), W_used=Wu, patchiness=op2["patchiness"],
                         rich=srich.astype(np.float32), occ=socc.astype(np.float32))
        print(f"[N={n} {name}] W={Wu} swarm {np.mean(sf):.3f}+/-{np.std(sf):.3f} slope {ss:.3f} "
              f"exploit(.5/2/4) {sx[0.5]:.3f}/{sx[2.0]:.3f}/{sx[4.0]:.3f} | random {np.mean(rf):.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
    fp = os.path.join(DATA, f"nscale_N{n}.npz")
    np.savez_compressed(fp, res=out)
    print(f"wrote {fp} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=int, default=512)
    ap.add_argument("--eval_ticks", type=int, default=4000)
    ap.add_argument("--window", type=int, default=2000)
    ap.add_argument("--base", type=int, default=0)
    a = ap.parse_args()
    main(a.W, a.eval_ticks, a.window, a.base)
