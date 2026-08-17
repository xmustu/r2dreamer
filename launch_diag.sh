#!/bin/bash
set -euo pipefail
eval "$(/home/zhengkai/miniconda3/bin/conda shell.bash hook)"
conda activate r2dreamer
export MUJOCO_GL=egl
export DAVIS_PATH=$HOME/datasets/DAVIS/JPEGImages/480p
cd /home/zhengkai/r2dreamer

echo "=== Launching Dreamer Diagnostic Study ==="
echo "Date: $(date)"

# DMC walker_walk - baseline on GPU 2
LOGDIR="/home/zhengkai/logdir/diag_study/baseline_walker_walk"
rm -rf "$LOGDIR"
mkdir -p "$LOGDIR"
echo "[$(date +%H:%M)] Starting walker_walk on GPU 2"
CUDA_VISIBLE_DEVICES=2 nohup python train.py \
    env=dmc_vision env.task=dmc_walker_walk env.steps=500000 \
    trainer.eval_every=20000 seed=42 \
    logdir="$LOGDIR" device=cuda:0 \
    > "$LOGDIR/console.log" 2>&1 &
echo "  PID: $!"

sleep 2

# DMC cheetah_run - baseline on GPU 3
LOGDIR="/home/zhengkai/logdir/diag_study/baseline_cheetah_run"
rm -rf "$LOGDIR"
mkdir -p "$LOGDIR"
echo "[$(date +%H:%M)] Starting cheetah_run on GPU 3"
CUDA_VISIBLE_DEVICES=3 nohup python train.py \
    env=dmc_vision env.task=dmc_cheetah_run env.steps=500000 \
    trainer.eval_every=20000 seed=42 \
    logdir="$LOGDIR" device=cuda:0 \
    > "$LOGDIR/console.log" 2>&1 &
echo "  PID: $!"

sleep 5

echo ""
echo "Checking progress..."
for d in baseline_walker_walk baseline_cheetah_run; do
    lines=$(wc -l < "/home/zhengkai/logdir/diag_study/$d/console.log" 2>/dev/null || echo 0)
    echo "  $d: $lines lines"
done

nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo "Launcher done."
