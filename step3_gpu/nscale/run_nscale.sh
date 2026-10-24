#!/bin/bash
# TRUE finite-size N-scaling, run CONCURRENTLY to saturate the GPU. Grow the ARENA with N and scale counts with
# area, holding every microscopic length (r, STEP, clump_radius, S) and every density/ratio fixed:
#   L = round(800*sqrt(N/200));  K = 0.4N;  C = round(0.05N) (>=3);  patchiness scaled in the driver.
# Each N-point is a latency-bound sequential sim (~57% util alone); running all at once overlaps the gaps -> ~100%.
export FLOCKING_DATA=/mnt/newssd/flocking_ltm_data/step3_gpu/nscale
D=.
NS=(50 100 200 400 800)
KS=(20 40 80 160 320)
CS=(3 5 10 20 40)
LS=(400 566 800 1131 1600)
WS=(2048 1024 512 256 128)
echo "=== N-scaling FINITE-SIZE CONCURRENT START $(date) FLOCKING_DATA=$FLOCKING_DATA ==="
pids=()
for i in 0 1 2 3 4; do
  n=${NS[$i]}; k=${KS[$i]}; c=${CS[$i]}; l=${LS[$i]}; w=${WS[$i]}
  FLOCKING_N=$n FLOCKING_K=$k FLOCKING_C=$c FLOCKING_L=$l \
    nohup python3 "$D/train_nscale.py" --W "$w" --eval_ticks 4000 --window 2000 \
    > "/tmp/nscale_N$n.log" 2>&1 &
  pids+=($!)
  echo "launched N=$n K=$k C=$c L=$l W=$w pid=$! log=/tmp/nscale_N$n.log"
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "=== N-scaling run DONE $(date) fail=$fail ==="
if [ $fail -eq 0 ]; then touch "$FLOCKING_DATA/NSCALE_DONE"; else echo "SOME N FAILED - no sentinel"; fi
