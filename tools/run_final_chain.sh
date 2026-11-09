#!/bin/bash
# Waits for the v2 cube download, then runs the final evaluation (CPU) and seed-340 neural training (GPU) in parallel.
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE/../pipelines"
LOG=../logs
until grep -q "ALL NDTI CUBES FINISHED" $LOG/ndti_v2_run.log 2>/dev/null; do sleep 60; done
echo "download finished $(date)" >> $LOG/final_chain.log
grep FAILED $LOG/ndti_v2_run.log >> $LOG/final_chain.log
( python final_eval.py > $LOG/final_eval.log 2>&1 && python final_analyze.py > $LOG/final_analyze.log 2>&1; echo "eval done rc=$? $(date)" >> $LOG/final_chain.log ) &
( python final_neural.py --models convlstm earthformer --seeds 340 --epochs 12 --resume --skip-done --max-hours 7 > $LOG/final_neural_s340.log 2>&1; echo "neural s340 done rc=$? $(date)" >> $LOG/final_chain.log ) &
wait
echo "CHAIN FINISHED $(date)" >> $LOG/final_chain.log
