#!/bin/bash
set -euo pipefail
eval "$(/home/zhengkai/miniconda3/bin/conda shell.bash hook)"
conda activate r2dreamer
export MUJOCO_GL=egl
cd /home/zhengkai/r2dreamer

echo "=== Final Diagnostic Study Launch ==="
echo "Date: $(date)"

launch() {
    local name=$1 gpu=$2 task=$3 steps=$4 seed=$5
    local logdir="/home/zhengkai/logdir/diag_study/${name}_s${seed}"
    rm -rf "$logdir" 2>/dev/null
    mkdir -p "$logdir"
    echo "[$(date +%H:%M)] GPU $gpu: $task ($steps steps, seed=$seed)"
    CUDA_VISIBLE_DEVICES=$gpu nohup python train.py \
        env=dmc_vision env.task="$task" env.steps="$steps" \
        trainer.eval_every=10000 seed="$seed" model=size12M device=cuda:0 \
        batch_size=4 batch_length=16 buffer.max_size=100000 \
        buffer.storage_device=cpu env.env_num=2 env.eval_episode_num=1 \
        logdir="$logdir" > "$logdir/console.log" 2>&1 &
    echo "  PID: $!"
}

# GPU 2: walker_walk baseline 
launch "walker_baseline" 2 "dmc_walker_walk" 200000 42
sleep 3

# GPU 3: cheetah_run baseline
launch "cheetah_baseline" 3 "dmc_cheetah_run" 200000 42
sleep 3

# Wait for env creation
echo "Waiting for env creation..."
for i in $(seq 1 12); do
    sleep 5
    for name in walker_baseline_s42 cheetah_baseline_s42; do
        line=$(grep -E 'Simulat|Evaluat|Compil' "/home/zhengkai/logdir/diag_study/${name}/console.log" 2>/dev/null | tail -1)
        if [ -n "$line" ]; then echo "  $name: $line"; fi
    done
done

echo ""
echo "=== Status ==="
for name in walker_baseline_s42 cheetah_baseline_s42; do
    last=$(grep -E 'Simulat|Evaluat|Compil|episode' "/home/zhengkai/logdir/diag_study/${name}/console.log" 2>/dev/null | tail -1)
    echo "  $name: ${last:-still init}"
done
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo "Launcher done."
