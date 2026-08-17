#!/bin/bash
cd /home/zhengkai/r2dreamer
eval "$(/home/zhengkai/miniconda3/bin/conda shell.bash hook)" && conda activate r2dreamer

TASKS=("dmc_cartpole_swingup" "dmc_cheetah_run" "dmc_walker_walk")
SEEDS=(42 123 456)

for TASK in "${TASKS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    # Skip cartpole s42 (already complete) and s123 (just launched)
    if [ "$TASK" = "dmc_cartpole_swingup" ] && [ "$SEED" = "42" ]; then continue; fi
    if [ "$TASK" = "dmc_cartpole_swingup" ] && [ "$SEED" = "123" ]; then continue; fi
    
    LOG="/home/zhengkai/r2dreamer/logs/ext_tt_${TASK}_s${SEED}.log"
    echo "[$(date)] Starting TT $TASK seed=$SEED on GPU 1"
    CUDA_VISIBLE_DEVICES=1 python3 train.py \
      model=size12M_tt \
      env=dmc_proprio env.task=$TASK \
      env.steps=100000 env.env_num=1 \
      seed=$SEED \
      > "$LOG" 2>&1
    echo "[$(date)] Finished TT $TASK seed=$SEED, exit=$?"
  done
done
echo "[$(date)] ALL TT RUNS COMPLETE"
