# GNN-done-right: design + pre-registration (2026-10-24)

Addresses TMLR round-2 Objection 3 (RL expert): the clustered-field harvest tie between the
reward-free swarm and the learners may reflect **communication vs no-communication**, not
learner-free vs learner. The feedforward IPPO baseline has no comms; recruitment is a hard-wired
coordination channel. The communication-equipped learner (GNN) was said to be *under-trained*, so
its shortfall cannot bound what a well-trained comms learner could do.

## Falsifiable question
Is the GNN's clustered-field shortfall (harvest 0.755 vs swarm 0.801, MLP-IPPO 0.798; long run,
150k env-steps, 16 seeds) a **genuine ceiling** or a **training/optimization artifact**?

- **H0 (our prior):** a properly-optimized comms learner still does NOT exceed the swarm in the
  clustered field → communication is not the lever; Objection 3 dies.
- **H1 (reviewer's worry):** a properly-optimized comms learner clearly exceeds the swarm → the tie
  was a comms-architecture artifact; the "competitive even with communication" claim must retreat.

## Evidence already in hand (zero new compute) — narrows the question sharply
From `step3_gpu_long.npz` (the run the paper cites):

1. **The GNN-L1 training curve is converged, not step-starved.** Clustered harvest plateaus at 0.76
   by ~66k ticks; the final 84k ticks are dead flat (final three records 0.760/0.761/0.761). Scarce
   is flat at 0.308 from ~66k. **⇒ more training steps cannot help. Route "A" (more ticks on the
   local 3090) is refuted and will not be run.**
2. **The clustered shortfall is largely seed variance, not a representational ceiling.** GNN-L1
   clustered per-seed sd = 0.040 (3× the MLP's 0.012); **best seed = 0.795 ≈ the 0.80 ceiling**,
   worst = 0.608, pulling the mean to 0.755. So some seeds already reach the ceiling; the mean is
   dragged down by high run-to-run variance.
3. **Scarce field is a small, tight, genuine shortfall:** GNN sd = 0.007, max 0.319 < MLP 0.329.
   Not variance-driven; a modest real gap there.

**Diagnosis:** the GNN's deficit is high-variance optimization, caused by the dense `N×N` adjacency
forcing a small world-batch. The MLP trains on `R=24` parallel worlds per update; the GNN on
`R=10` (≈2.4× fewer) because the graph's memory caps `R` at 10 on the 24 GB 3090 (GNN-L1 already
uses 12.4 GB at R=10). Fewer worlds ⇒ noisier PPO advantage estimates for the larger net ⇒ some
seeds converge to worse optima.

## The ONE informative experiment: variance reduction (more worlds) + light HP
The only lever consistent with the evidence is **lower-variance gradients**, i.e. **more parallel
worlds**, plus a small learning-rate check for the deeper net. Concretely:

- **Primary:** GNN-L1 with **R_gnn = 24** (match the MLP's worlds), ticks = 150k (already converged),
  16 seeds × 2 fields. This directly tests whether variance reduction lifts the mean toward the
  best-seed 0.795.
- **HP check (fairness-bounded):** lr ∈ {1e-4, 3e-4, 5e-4}, pick by clustered harvest at a 60k-tick
  screen, confirm the winner at full budget. Keep everything else identical to the MLP
  (ent=0.01, grad-clip 0.5, shape=0.05, h=64) so no reviewer can say we tuned the GNN preferentially.
- **Optional:** one extra `R_gnn = 48` point on an 80 GB card to show the mean has plateaued in R
  (so "still too few worlds" cannot be re-raised). L=3 skipped (already monotonically worse).
- **Seeds for a clean tie claim:** 16 seeds gave GNN CI ≈ ±0.03; a TOST equivalence at ±0.02 wants
  tighter. Run **24–32 seeds** in the clustered field so an equivalence (tie) verdict is powered.

### Where to run it (Tesla is feasible via seed-chunking; Anvil chosen for throughput)
R=24 OOMs the 24 GB 3090 only if all seeds are folded into one batch (M=48 ≈ 30 GB). With
**seed-chunking** (run `seed_chunk` seeds at a time, M = 2·seed_chunk, R pinned at 24) VRAM scales
with M·R: at seed_chunk=8, M=16, R=24 ≈ 15 GB, which **fits the 3090**. So the experiment runs on
either machine; R is never backed off, only the chunk size shrinks.

- **Tesla (3090):** full campaign ≈ 12–15 h (lean, screen skipped ≈ 9 h); GPU is free now, no queue.
- **Anvil (A100):** compute ≈ 6–9 h (faster card, bigger chunks) but pay the allocation queue
  (≈ up to 2 days last time). Chosen here to keep the Tesla free and because the A100 fits larger
  chunks (fewer re-inits, cleaner saturation). Scratch `/anvil/scratch/x-gwilkerson/rl_gnn/`,
  allocation `cis240536-gpu`, env `ulmv5`.

## Pre-registered decision rule (clustered field, the contested one)
Let g = best-HP GNN mean clustered harvest, s = swarm 0.801.
- **g ≤ s and within TOST ±0.02 (≈ 0.78–0.82):** report "a properly-optimized, communication-equipped
  learner still does not exceed the reward-free swarm" → **strengthen** the claim; Objection 3 closed.
- **g > 0.84 (clearly beats swarm beyond noise):** report honestly → **soften** abstract/claims to
  "competitive with feedforward learners; a well-trained graph policy can exceed the swarm's harvest
  in clustered fields, at the cost of N env-steps of training" — the allocation-shape headline is
  untouched regardless.
- **0.82 < g ≤ 0.84:** report as weak edge; keep scoped "competitive" language with the number stated.

Scarce field: report whatever it gives; the swarm already *trails* on scarce harvest, so no risk there.

## Fairness guardrails (so the result survives adversarial reading)
- Identical reward shaping, entropy, clip, width, eval protocol (eval_policy, R_eval=16,
  eval_ticks=4000) and exploitability metric (n_i ≥ 2) across arms — drop-in for Table in §compare.
- Match the GNN's *worlds* to the MLP (R=24), do not merely exceed the MLP's samples wildly.
- Report the GNN learning curve + per-seed spread in the appendix so neither "too few steps" nor
  "unlucky seed" can be re-raised.
- At most the same per-arm HP budget any baseline would get; log every HP.

## Compute plan
1. **Local smoke (3090, minutes):** tiny R/ticks/seeds run to verify the seed-chunked GNN path,
   incremental + per-chunk saves, env determinism, and that curve/eval plumbing emits drop-in Table
   numbers. No science.
2. **Anvil full:** A100; primary GNN-L1 R=24, lr 3e-4, 24 seeds, both fields, + MLP-IPPO reference +
   swarm/random baselines (`train_gnn_right.py --mode full`). Single job, `--time=12:00:00`, incremental
   saves so a walltime kill keeps finished seeds. Optional separate lr-screen job as insurance.
3. **Fallback:** the Tesla can run the identical job (seed-chunked, R pinned at 24) in ≈ 9–15 h if the
   Anvil queue stalls — same code, same result; only the host differs.

## Risks / honest downside
- H1 could hold (GNN beats swarm with more worlds). Then we retreat one clause; the allocation-shape
  result and the learner-free/reward-free framing survive. Low prior: the simpler, better-fed MLP
  tops out at 0.798 ≈ swarm, so ~0.80 looks like an embodiment ceiling for every architecture.
- Anvil env/port friction + queue (previously cancelled a job for a 2-day queue). Mitigation: the
  code is single-file torch, already A100-ready; submit early, treat as background.

## Zero-compute paper additions available NOW (independent of the run)
Two sentences the existing data already support, strengthening the "weak lower bound" caveat:
- "The graph policy's training curve is flat over its final 84k env-steps (clustered plateau 0.76 by
  step 66k), so its shortfall is a converged optimum at this configuration, not insufficient training."
- "Its clustered shortfall is largely run-to-run variance (per-seed sd 0.04 vs the feedforward
  learner's 0.01; its best seeds reach 0.795, at the 0.80 ceiling), i.e. optimization noise from the
  small world-batch the dense graph forces, not a representational ceiling."
