#!/bin/bash
set -euo pipefail
eval "$(/home/zhengkai/miniconda3/bin/conda shell.bash hook)"
conda activate r2dreamer
export MUJOCO_GL=egl
cd /home/zhengkai/r2dreamer

echo "=== Diagnostic Study (Fixed) ==="
echo "Date: $(date)"

launch() {
    local name=$1 gpu=$2 task=$3 steps=$4 seed=$5
    local logdir="/home/zhengkai/logdir/diag_study/${name}_s${seed}"
    rm -rf "$logdir" 2>/dev/null
    mkdir -p "$logdir"
    echo "[$(date +%H:%M)] GPU $gpu: $task ($steps steps, seed=$seed)"
    CUDA_VISIBLE_DEVICES=$gpu nohup python -c "
import numpy as np
_orig_reshape = np.reshape
def _patched_reshape(a, *args, **kwargs):
    if 'newshape' in kwargs:
        kwargs = dict(kwargs)
        args = (kwargs.pop('newshape'),) + args
    return _orig_reshape(a, *args, **kwargs)
np.reshape = _patched_reshape

# Now run training
import sys
sys.argv = ['train.py', 'env=dmc_vision', 'env.task=$task', 'env.steps=$steps',
    'trainer.eval_every=10000', 'seed=$seed', 'model=size12M', 'device=cuda:0',
    'batch_size=4', 'batch_length=16', 'buffer.max_size=100000',
    'buffer.storage_device=cpu', 'env.env_num=2', 'env.eval_episode_num=1',
    'logdir=$logdir']
exec(open('train.py').read())
" > "$logdir/console.log" 2>&1 &
    echo "  PID: $!"
}

# GPU 2: walker_walk
launch "walker_diag" 2 "dmc_walker_walk" 200000 42
sleep 5

# GPU 3: cheetah_run
launch "cheetah_diag" 3 "dmc_cheetah_run" 200000 42
sleep 5

echo "Waiting for init..."
for i in $(seq 1 12); do
    sleep 5
    for name in walker_diag_s42 cheetah_diag_s42; do
        prog=$(grep -E 'Simulat|Evaluat|Compil|step.*episode' "/home/zhengkai/logdir/diag_study/${name}/console.log" 2>/dev/null | tail -1)
        if [ -n "$prog" ]; then echo "  $name: $prog"; fi
    done
done
echo ""
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
