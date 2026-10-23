#!/usr/bin/env python3
"""Learner-robustness sweep for the F1 sensing-limited comparison, NON-destructive.

The committee's main worry about the central result (swarm statistically equivalent to the best learner on
harvest, and the most gradient-faithful arm) is that the learner baseline might be under-powered: it was trained
at a single hidden width (64) and learning rate (3e-4). This sweep retrains the feedforward MLP-IPPO learner
across a grid of CAPACITIES (hidden width) and LEARNING RATES, at both operating points, to test whether the
~0.80 (clustered) / ~0.33 (scarce) oracle fraction is a real ceiling or left on the table by a weak/under-tuned
network. If no stronger/better-tuned learner meaningfully exceeds it, the equivalence/allocation claims hold
against stronger learners, not just one configuration.

Reuses train_gpu.train (the standard IPPO trainer -- it already takes h and lr), eval_policy, _with_oom_retry,
and train_long (PAPER/FAV at gamma=1.0, exploit_per_world, per_seed) UNCHANGED. torch_env.py / train_gpu.py /
train_long.py are NOT modified. Output -> step3_gpu_robust.npz (same op-point definitions as the main run ->
directly comparable to step3_gpu_long.npz's mlp_ippo_* keys).
Run:  python train_robust.py --ticks 80000 --nseed 8 --R 16
"""
import os, sys, time, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch_env as TE
from train_gpu import train, eval_policy, DEV, _with_oom_retry
from train_long import PAPER, FAV, exploit_per_world, per_seed

_rd = os.path.normpath(os.path.join(HERE, "..", "data", "step3_gpu"))
DATA = os.environ.get("FLOCKING_DATA") or (_rd if os.path.isdir(_rd) else "/mnt/newssd/flocking_ltm_data/step3_gpu")
os.makedirs(DATA, exist_ok=True)

# (tag, hidden width h, learning rate lr).  First entry = baseline reproduction check (should match the
# main run's MLP-IPPO ~0.80/0.33); the rest test more capacity and higher/lower learning rate.
CONFIGS = [
    ("h128_lr3em4", 128, 3e-4),   # 2x capacity
    ("h256_lr3em4", 256, 3e-4),   # 4x capacity
    ("h128_lr1em3", 128, 1e-3),   # tuning: higher learning rate
]
# (the h=64, lr=3e-4 baseline = the main run's MLP-IPPO, 0.80/0.33; reused, not re-run)


def main(ticks, nseed, R, R_eval, eval_ticks, base, out, configs):
    POINTS = [("PAPER", PAPER), ("FAVORABLE", FAV)]
    ops = [op for _, op in POINTS for _ in range(nseed)]          # M = 2*nseed, PAPER block then FAV block
    res = {name: {} for name, _ in POINTS}
    pslice = {"PAPER": slice(0, nseed), "FAVORABLE": slice(nseed, 2 * nseed)}
    wslice = lambda Rw: {"PAPER": slice(0, nseed * Rw), "FAVORABLE": slice(nseed * Rw, 2 * nseed * Rw)}
    t0 = time.time()

    for (tag, h, lr) in configs:
        (tr, Rused) = _with_oom_retry(
            lambda R, h=h, lr=lr: train(ops, R, ticks=ticks, seed=base, L=None, h=h, lr=lr, sample_gpu=True), R)
        ev, rich, occ = eval_policy(tr["pol"], ops, R_eval, eval_ticks, seed=base + 10000, L=None)
        ex = exploit_per_world(rich, occ)
        print(f"[{tag}] h={h} lr={lr:g} R={Rused} ticks={ticks} eval frac: "
              f"PAPER {ev[pslice['PAPER']].mean():.3f}  FAV {ev[pslice['FAVORABLE']].mean():.3f}  "
              f"util={tr['gpu_util']}% vram={tr['gpu_mem']}MiB ({time.time()-t0:.0f}s)", flush=True)
        for name in res:
            res[name][f"{tag}_finals"] = ev[pslice[name]]
            ws = wslice(R_eval)[name]
            res[name][f"{tag}_slope"] = TE.slope_pooled(rich[ws], occ[ws])
            res[name][f"{tag}_exploit"] = per_seed(ex[ws], nseed, R_eval)
            res[name][f"{tag}_hw"] = h; res[name][f"{tag}_lr"] = lr
        if DEV == "cuda":
            torch.cuda.empty_cache()

    res["_meta"] = dict(ticks=ticks, nseed=nseed, R=R, R_eval=R_eval, eval_ticks=eval_ticks,
                        configs=[c[0] for c in configs], grid=[(c[1], c[2]) for c in configs])
    outp = os.path.join(DATA, out)
    np.savez_compressed(outp, res=res)
    print(f"wrote {outp}  ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=80000)
    ap.add_argument("--nseed", type=int, default=8)
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--R_eval", type=int, default=16)
    ap.add_argument("--eval_ticks", type=int, default=4000)
    ap.add_argument("--base", type=int, default=0)
    ap.add_argument("--out", type=str, default="step3_gpu_robust.npz")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU-sized config to validate shapes")
    a = ap.parse_args()
    if a.smoke:
        print(f"[SMOKE] device={DEV}")
        main(ticks=256, nseed=1, R=2, R_eval=2, eval_ticks=200, base=0,
             out="step3_gpu_robust_smoke.npz", configs=[("h64_lr3em4", 64, 3e-4), ("h128_lr1em3", 128, 1e-3)])
    else:
        main(ticks=a.ticks, nseed=a.nseed, R=a.R, R_eval=a.R_eval, eval_ticks=a.eval_ticks,
             base=a.base, out=a.out, configs=CONFIGS)
