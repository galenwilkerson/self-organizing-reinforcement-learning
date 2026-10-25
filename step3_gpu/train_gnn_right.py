#!/usr/bin/env python3
"""GNN-done-right: test whether the communication-equipped learner's clustered shortfall is an
optimization artifact (noisy gradients from the small world-batch the dense N*N graph forces) rather
than a representational ceiling. See GNN_DONE_RIGHT_DESIGN.md.

The fix is VARIANCE REDUCTION = more parallel worlds. The long run used R_gnn=10 (capped by VRAM on a
24 GB card); this raises it to R_gnn=24 to match the feedforward learner, which needs ~30 GB -> an A100.
Nothing in the training/eval code changes (train_gpu.train / eval_policy, torch_env all UNCHANGED): only
R, lr, and the number of seeds. Reward shaping, entropy, clip, width and the eval protocol are held
identical to the feedforward baseline so the comparison cannot be said to favour the GNN.

Seeds are run in CHUNKS (M = 2*seed_chunk policies per call) so that R stays pinned at 24 and memory is
controlled by the chunk size, NOT by backing R down (backing R down would defeat the experiment). Chunks
are numerically independent (grouped per-policy weights, no cross-policy gradient), distinct worlds/inits
per chunk via a per-chunk seed offset. Results are saved incrementally so a walltime kill keeps finished work.

Modes:
  screen : GNN-L1 at R_gnn, SHORT ticks, few seeds, for each lr in --lrs; prints clustered harvest per lr.
  full   : GNN-L1 at --lr, full ticks + seeds, both fields; optionally MLP-IPPO reference + swarm/random.

Examples:
  # local smoke (tiny, correctness only):
  python train_gnn_right.py --mode full --R_gnn 4 --ticks 2000 --nseed 2 --seed_chunk 2 --eval_ticks 500 \
      --no_mlp --no_baseline --out gnn_right_smoke.npz
  # Anvil lr screen then full:
  python train_gnn_right.py --mode screen --R_gnn 24 --ticks 60000 --nseed 8 --seed_chunk 8 --lrs 1e-4,3e-4,5e-4
  python train_gnn_right.py --mode full   --R_gnn 24 --ticks 150000 --nseed 24 --seed_chunk 8 --lr 3e-4
"""
import os, sys, time, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch_env as TE
from torch_env import N
import train_gpu as TG
from train_gpu import train, eval_policy, _with_oom_retry, DEV
from train_long import exploit_per_world, per_seed, eval_sr, PAPER, FAV, DATA

POINTS = [("PAPER", PAPER), ("FAVORABLE", FAV)]


def run_gnn_chunked(nseed, seed_chunk, R_gnn, ticks, lr, eval_ticks, R_eval, base, tag, partial_path=None):
    """GNN-L1 over nseed seeds/field, run in chunks of seed_chunk. Returns per-field dicts with
    finals (nseed,), slope, exploit (nseed,), and the seed-averaged learning curve (t, frac).
    If partial_path is set, a running npz of the seeds completed so far is written after each chunk,
    so a walltime kill keeps finished seeds."""
    acc = {name: dict(finals=[], exploit=[], curves=[]) for name, _ in POINTS}
    curve_t = None
    gpu_util = gpu_mem = 0
    done = 0
    _t_chunk0 = time.time()
    while done < nseed:
        c = min(seed_chunk, nseed - done)
        ops = [op for _, op in POINTS for _ in range(c)]            # M = 2*c, PAPER block then FAV block
        pslice = {"PAPER": slice(0, c), "FAVORABLE": slice(c, 2 * c)}
        wsl = {"PAPER": slice(0, c * R_eval), "FAVORABLE": slice(c * R_eval, 2 * c * R_eval)}
        sd = base + 1000 * (done // max(seed_chunk, 1))
        (tr, R) = _with_oom_retry(
            lambda R: train(ops, R, ticks=ticks, seed=sd, L=1, lr=lr, sample_gpu=True), R_gnn)
        if R != R_gnn:
            print(f"  [WARN] {tag}: R backed off {R_gnn}->{R} (OOM). Lower --seed_chunk to keep R pinned.",
                  flush=True)
        ev, rich, occ = eval_policy(tr["pol"], ops, R_eval, eval_ticks, seed=sd + 10000, L=1)
        ex = exploit_per_world(rich, occ)
        for name in acc:
            acc[name]["finals"].append(ev[pslice[name]])
            acc[name]["exploit"].append(per_seed(ex[wsl[name]], c, R_eval))
            acc[name]["curves"].append(tr["curve_f"][:, pslice[name]].mean(1))  # (n_rec,)
        curve_t = tr["curve_t"]
        gpu_util = max(gpu_util, tr["gpu_util"]); gpu_mem = max(gpu_mem, tr["gpu_mem"])
        done += c
        torch.cuda.empty_cache()
        if partial_path is not None:                                # running save: seeds done so far
            part = {name: dict(finals=np.concatenate(acc[name]["finals"]),
                               exploit=np.concatenate(acc[name]["exploit"]),
                               seeds_done=done) for name, _ in POINTS}
            np.savez_compressed(partial_path, res=part)
            print(f"  [{tag}] {done}/{nseed} seeds done, partial saved ({time.time()-_t_chunk0:.0f}s)",
                  flush=True)
    out = {}
    for name, _ in POINTS:
        fin = np.concatenate(acc[name]["finals"]); exn = np.concatenate(acc[name]["exploit"])
        cur = np.stack(acc[name]["curves"]).mean(0)                 # avg curve over chunks
        out[name] = dict(finals=fin, exploit=exn, curve_t=np.asarray(curve_t), curve_f=cur)
    out["_gpu"] = (gpu_util, gpu_mem)
    return out


def main(a):
    t0 = time.time()
    print(f"device={DEV}  N={N}  mode={a.mode}  R_gnn={a.R_gnn}  ticks={a.ticks}  nseed={a.nseed} "
          f"seed_chunk={a.seed_chunk}", flush=True)
    os.makedirs(DATA, exist_ok=True)
    outp = os.path.join(DATA, a.out)

    if a.mode == "screen":
        lrs = [float(x) for x in a.lrs.split(",")]
        res = {"_meta": dict(mode="screen", R_gnn=a.R_gnn, ticks=a.ticks, nseed=a.nseed, lrs=lrs)}
        for lr in lrs:
            r = run_gnn_chunked(a.nseed, a.seed_chunk, a.R_gnn, a.ticks, lr, a.eval_ticks,
                                a.R_eval, a.base, f"GNN-L1 lr={lr:.0e}")
            fav = r["FAVORABLE"]["finals"]; pap = r["PAPER"]["finals"]
            res[f"lr_{lr:.0e}"] = {k: r[k] for k in ("PAPER", "FAVORABLE")}
            print(f"[screen lr={lr:.0e}] FAV {fav.mean():.3f}+-{fav.std():.3f}  PAPER {pap.mean():.3f}"
                  f"  util={r['_gpu'][0]}% vram={r['_gpu'][1]}MiB  ({time.time()-t0:.0f}s)", flush=True)
            np.savez_compressed(outp, res=res)                      # incremental
        print(f"wrote {outp}  ({time.time()-t0:.0f}s)", flush=True)
        return

    # full
    res = {name: {} for name, _ in POINTS}
    res["_meta"] = dict(mode="full", R_gnn=a.R_gnn, ticks=a.ticks, nseed=a.nseed, lr=a.lr,
                        R_eval=a.R_eval, eval_ticks=a.eval_ticks)
    g = run_gnn_chunked(a.nseed, a.seed_chunk, a.R_gnn, a.ticks, a.lr, a.eval_ticks,
                        a.R_eval, a.base, f"GNN-L1 lr={a.lr:.0e}", partial_path=outp + ".gnn_partial.npz")
    for name, _ in POINTS:
        res[name]["gnn_L1_finals"] = g[name]["finals"]
        res[name]["gnn_L1_exploit"] = g[name]["exploit"]
        res[name]["gnn_L1_curve_t"] = g[name]["curve_t"]; res[name]["gnn_L1_curve_f"] = g[name]["curve_f"]
    print(f"[GNN-L1 R={a.R_gnn}] FAV {res['FAVORABLE']['gnn_L1_finals'].mean():.3f}"
          f"+-{res['FAVORABLE']['gnn_L1_finals'].std():.3f}  PAPER {res['PAPER']['gnn_L1_finals'].mean():.3f}"
          f"  util={g['_gpu'][0]}% vram={g['_gpu'][1]}MiB  ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(outp, res=res)                              # incremental

    if not a.no_mlp:
        ops = [op for _, op in POINTS for _ in range(a.nseed)]
        pslice = {"PAPER": slice(0, a.nseed), "FAVORABLE": slice(a.nseed, 2 * a.nseed)}
        (trm, R) = _with_oom_retry(
            lambda R: train(ops, R, ticks=a.ticks, seed=a.base, L=None, lr=3e-4, sample_gpu=True), a.R_mlp)
        evm, richm, occm = eval_policy(trm["pol"], ops, a.R_eval, a.eval_ticks, seed=a.base + 10000, L=None)
        for name in ("PAPER", "FAVORABLE"):
            res[name]["mlp_ippo_finals"] = evm[pslice[name]]
        print(f"[MLP-IPPO R={R}] FAV {res['FAVORABLE']['mlp_ippo_finals'].mean():.3f}  "
              f"PAPER {res['PAPER']['mlp_ippo_finals'].mean():.3f}  ({time.time()-t0:.0f}s)", flush=True)
        np.savez_compressed(outp, res=res)

    if not a.no_baseline:
        for name, op in POINTS:
            sf, ss, srich, socc = eval_sr("swarm", 256, a.eval_ticks, 2000, op, a.base + 555)
            res[name]["swarm"] = sf; res[name]["swarm_slope"] = ss
            res[name]["swarm_exploit"] = exploit_per_world(srich, socc)
            print(f"[swarm {name}] {sf.mean():.3f} (slope {ss:.2f})  ({time.time()-t0:.0f}s)", flush=True)
        np.savez_compressed(outp, res=res)

    print(f"wrote {outp}  ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["screen", "full"], default="full")
    ap.add_argument("--R_gnn", type=int, default=24)
    ap.add_argument("--R_mlp", type=int, default=24)
    ap.add_argument("--R_eval", type=int, default=16)
    ap.add_argument("--ticks", type=int, default=150000)
    ap.add_argument("--nseed", type=int, default=24)
    ap.add_argument("--seed_chunk", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lrs", type=str, default="1e-4,3e-4,5e-4")
    ap.add_argument("--eval_ticks", type=int, default=4000)
    ap.add_argument("--base", type=int, default=0)
    ap.add_argument("--no_mlp", action="store_true")
    ap.add_argument("--no_baseline", action="store_true")
    ap.add_argument("--out", type=str, default="step3_gpu_gnn_right.npz")
    main(ap.parse_args())
