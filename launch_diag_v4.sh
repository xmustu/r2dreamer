#!/bin/bash
set -euo pipefail
eval "$(/home/zhengkai/miniconda3/bin/conda shell.bash hook)"
conda activate r2dreamer
export MUJOCO_GL=egl
export DAVIS_PATH=$HOME/datasets/DAVIS/JPEGImages/480p
cd /home/zhengkai/r2dreamer

echo "=== Diagnostic Study — DCS Vision ==="
echo "Date: $(date)"

run_exp() {
    local name=$1 gpu=$2 task=$3 steps=$4
    local logdir="/home/zhengkai/logdir/diag_study/${name}"
    rm -rf "$logdir"
    mkdir -p "$logdir"
    echo "[$(date +%H:%M)] $name on GPU $gpu: $task ($steps steps)"
    CUDA_VISIBLE_DEVICES=$gpu nohup python train.py \
        env=dcs_vision env.task="$task" env.steps="$steps" \
        trainer.eval_every=10000 seed=42 model=size12M device=cuda:0 \
        batch_size=4 batch_length=16 buffer.max_size=100000 \
        buffer.storage_device=cpu env.env_num=2 env.eval_episode_num=1 \
        logdir="$logdir" > "$logdir/console.log" 2>&1 &
    echo "  PID: $!"
    sleep 3
}

# GPU 2: DreamerV3 baseline on cheetah_clean + walker_clean
run_exp "dv3_cheetah_clean" 2 "dcs_cheetah_run_clean" 200000

# GPU 3: DreamerV3 baseline on another task
run_exp "dv3_walker_clean" 3 "dcs_walker_walk_clean" 200000

sleep 15

echo ""
echo "=== Progress Check ==="
for name in dv3_cheetah_clean dv3_walker_clean; do
    l=$(wc -l < "/home/zhengkai/logdir/diag_study/${name}/console.log" 2>/dev/null || echo 0)
    prog=$(grep -E 'Simulat|Compil|Evaluat' "/home/zhengkai/logdir/diag_study/${name}/console.log" 2>/dev/null | tail -1)
    echo "  $name: $l lines - $prog"
done
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo "Launcher done."
