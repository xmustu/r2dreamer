#!/bin/bash
#
# v2 Experiment Orchestration Script
# ===================================
# Manages multi-task, multi-seed training and OOD evaluation across 4 GPUs.
#
# Run Order (from EXPERIMENT_PLAN.md):
#   M0 (Sanity): cartpole-swingup, 1 seed, DV3 + SF-RSSM → verify training
#   M1 (B1-ID):  5 tasks × 3 seeds × 2 systems = 30 training runs
#   M2 (B2-OOD): Mechanism-aligned OOD eval on all trained models
#   M3 (B3-OOD): Encoder-corrupting OOD eval on all trained models
#
# Usage:
#   chmod +x run_experiments_v2.sh
#   ./run_experiments_v2.sh sanity     # Run sanity check only
#   ./run_experiments_v2.sh train      # Run full B1 training
#   ./run_experiments_v2.sh ood        # Run OOD evaluation
#   ./run_experiments_v2.sh all        # Run everything
#
# Environment:
#   - 4x A6000 GPUs
#   - conda env: r2dreamer
#   - MUJOCO_GL=egl (set in conda activate)

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda
eval "$(/home/zhengkai/miniconda3/bin/conda shell.bash hook)"
conda activate r2dreamer

# Clear stale pycache
find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

# Experiment root
LOGDIR="logdir/v2"
mkdir -p "$LOGDIR"

# 5 DMC tasks for v2
TASKS=(
    "dmc_walker_walk"
    "dmc_cheetah_run"
    "dmc_cartpole_swingup"
    "dmc_finger_spin"
    "dmc_hopper_hop"
)

# Systems to train
SYSTEMS=(
    "size12M_dv3"    # DreamerV3 baseline
    "size12M_sf"     # SF-RSSM (our method)
)

# Seeds
SEEDS=(42 200 201)

# Training config
TRAIN_STEPS=1010000  # 1.01M steps

# OOD perturbations (comma-separated)
OOD_MECH="M1_high_friction,M1_low_friction,M2_high_mass,M2_low_mass,M3_camera_shift,M4_camera_rotation"
OOD_ENC="E1_texture_noise,E2_color_bg,E3_blur"
OOD_ALL="${OOD_MECH},${OOD_ENC}"

# Number of OOD eval episodes
OOD_EPISODES=10

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

wait_for_gpu_slot() {
    # Wait until fewer than MAX_JOBS GPU processes are running
    local max_jobs=${1:-4}
    local check_cmd="ps aux | grep -E 'python.*train\.py' | grep -v grep | wc -l"
    while true; do
        local running
        running=$(eval "$check_cmd" 2>/dev/null || echo 0)
        if [ "$running" -lt "$max_jobs" ]; then
            break
        fi
        log "Waiting for GPU slot... ($running/$max_jobs running)"
        sleep 60
    done
}

get_gpu_id() {
    # Return the GPU ID with the most free memory
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null |
        sort -t',' -k2 -rn | head -1 | cut -d',' -f1 | tr -d ' '
}

run_training() {
    local system=$1
    local task=$2
    local seed=$3
    local gpu=$4

    local task_short="${task#dmc_}"
    local run_name="${system}_${task_short}_s${seed}"
    local run_logdir="${LOGDIR}/${run_name}"

    log "Starting: ${run_name} on GPU ${gpu}"

    CUDA_VISIBLE_DEVICES="${gpu}" \
        nohup python train.py \
        "env.task=${task}" \
        "model=${system}" \
        "seed=${seed}" \
        "device=cuda:0" \
        "logdir=${run_logdir}" \
        > "${LOGDIR}/${run_name}.log" 2>&1 &

    echo $!  # Return PID
}

run_sanity() {
    # M0: Sanity check — cartpole-swingup, 1 seed, DV3 + SF-RSSM
    log "========== M0: Sanity Check =========="

    local task="dmc_cartpole_swingup"
    local seed=42

    for system in "${SYSTEMS[@]}"; do
        local gpu
        gpu=$(get_gpu_id)
        log "Sanity: ${system} ${task} seed=${seed} → GPU ${gpu}"
        local pid
        pid=$(run_training "$system" "$task" "$seed" "$gpu")
        log "  PID=${pid}"
        sleep 5
    done

    log "Sanity runs launched. Monitor with:"
    log "  tail -f ${LOGDIR}/size12M_dv3_cartpole_swingup_s42.log"
    log "  tail -f ${LOGDIR}/size12M_sf_cartpole_swingup_s42.log"
    log ""
    log "After both complete (~20h each), run: ./run_experiments_v2.sh train"
}

run_full_training() {
    # M1 (B1): Full in-distribution training
    log "========== M1 (B1): In-Distribution Training =========="
    log "Tasks: ${#TASKS[@]} | Systems: ${#SYSTEMS[@]} | Seeds: ${#SEEDS[@]}"
    log "Total runs: $((${#TASKS[@]} * ${#SYSTEMS[@]} * ${#SEEDS[@]}))"
    log ""

    local total=0
    local pids=()

    for task in "${TASKS[@]}"; do
        for system in "${SYSTEMS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                wait_for_gpu_slot 4
                local gpu
                gpu=$(get_gpu_id)
                local pid
                pid=$(run_training "$system" "$task" "$seed" "$gpu")
                pids+=("$pid:$system:$task:$seed:$gpu")
                total=$((total + 1))
                log "Launched run ${total}/30: ${system} ${task} s=${seed} GPU=${gpu} PID=${pid}"
                sleep 10  # Stagger launches to avoid race conditions
            done
        done
    done

    log ""
    log "========== All ${total} training runs launched =========="
    log "Monitor all: watch -n 5 'ps aux | grep train.py | grep -v grep | wc -l'"
    log "Check progress: tail -n 3 ${LOGDIR}/*.log | grep -E '(Steps|episode_return|loss)'"
}

run_ood_evaluation() {
    # M2 + M3: OOD evaluation on all trained models
    log "========== M2/M3: OOD Evaluation =========="

    local results_dir="${LOGDIR}/ood_results"
    mkdir -p "$results_dir"

    for task in "${TASKS[@]}"; do
        local task_short="${task#dmc_}"
        for system in "${SYSTEMS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                local run_name="${system}_${task_short}_s${seed}"
                local ckpt="${LOGDIR}/${run_name}/latest.pt"
                local config="${LOGDIR}/${run_name}/.hydra/config.yaml"
                local output="${results_dir}/${run_name}_ood.json"

                if [ ! -f "$ckpt" ]; then
                    log "WARNING: Checkpoint not found: $ckpt — skipping"
                    continue
                fi

                wait_for_gpu_slot 3  # Use at most 3 GPUs for OOD (leave 1 free)
                local gpu
                gpu=$(get_gpu_id)

                log "OOD eval: ${run_name} → GPU ${gpu}"

                CUDA_VISIBLE_DEVICES="${gpu}" \
                    nohup python eval_ood_v2.py \
                    "$ckpt" \
                    "$config" \
                    "cuda:0" \
                    "$output" \
                    --task "$task_short" \
                    --episodes "$OOD_EPISODES" \
                    --perturbations "$OOD_ALL" \
                    --seed 42 \
                    > "${LOGDIR}/${run_name}_ood.log" 2>&1 &

                sleep 5
            done
        done
    done

    log "OOD evaluation launched for all trained models."
    log "Results will be in: ${results_dir}/"
}

run_aggregate_results() {
    # Aggregate all OOD results into a single summary JSON
    log "========== Aggregating Results =========="

    local results_dir="${LOGDIR}/ood_results"
    local aggregate="${LOGDIR}/aggregate_results.json"

    python3 -c "
import json, os, glob
from pathlib import Path

results = {}
for f in sorted(glob.glob('${results_dir}/*_ood.json')):
    name = Path(f).stem.replace('_ood', '')
    try:
        with open(f) as fh:
            data = json.load(fh)
        results[name] = {
            'task': data.get('task', '?'),
            'sf_enabled': data.get('sf_enabled', False),
            'clean_mean': data['results']['clean']['mean'],
            'mechanism_retention': {},
            'encoder_retention': {},
        }
        for k, v in data['results'].items():
            if k == 'clean':
                continue
            cat = v.get('category', '?')
            ret = v.get('retention', None)
            if ret is not None:
                results[name][f'{cat}_retention'][k] = round(ret, 4)
    except Exception as e:
        results[name] = {'error': str(e)}

# Compute summary statistics
summary = {
    'dv3_mech_retention': [],
    'sf_mech_retention': [],
    'dv3_enc_retention': [],
    'sf_enc_retention': [],
}
for name, data in results.items():
    if 'error' in data:
        continue
    is_sf = data.get('sf_enabled', False)
    mech_vals = list(data.get('mechanism_retention', {}).values())
    enc_vals = list(data.get('encoder_retention', {}).values())
    if is_sf:
        if mech_vals:
            summary['sf_mech_retention'].extend(mech_vals)
        if enc_vals:
            summary['sf_enc_retention'].extend(enc_vals)
    else:
        if mech_vals:
            summary['dv3_mech_retention'].extend(mech_vals)
        if enc_vals:
            summary['dv3_enc_retention'].extend(enc_vals)

import numpy as np
for key in summary:
    if summary[key]:
        summary[f'{key}_mean'] = float(np.mean(summary[key]))
        summary[f'{key}_std'] = float(np.std(summary[key]))

output = {'per_run': results, 'summary': summary}
with open('${aggregate}', 'w') as fh:
    json.dump(output, fh, indent=2)

print(json.dumps(summary, indent=2))
"
    log "Aggregate results saved to: ${aggregate}"
}

show_status() {
    log "========== Experiment Status =========="
    log ""

    local running
    running=$(ps aux | grep -E 'python.*train\.py' | grep -v grep | wc -l)
    log "Training runs active: ${running}"

    local ood_running
    ood_running=$(ps aux | grep -E 'python.*eval_ood_v2' | grep -v grep | wc -l)
    log "OOD evals active: ${ood_running}"

    log ""
    log "GPU usage:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null

    log ""
    log "Completed training runs:"
    for task in "${TASKS[@]}"; do
        local task_short="${task#dmc_}"
        for system in "${SYSTEMS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                local run_name="${system}_${task_short}_s${seed}"
                local ckpt="${LOGDIR}/${run_name}/latest.pt"
                if [ -f "$ckpt" ]; then
                    local ckpt_time
                    ckpt_time=$(stat -c %y "$ckpt" 2>/dev/null | cut -d'.' -f1)
                    echo "  ✅ ${run_name} — ${ckpt_time}"
                fi
            done
        done
    done

    log ""
    log "OOD results available:"
    ls -1 "${LOGDIR}/ood_results/"*_ood.json 2>/dev/null | wc -l
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

case "${1:-help}" in
    sanity)
        run_sanity
        ;;
    train)
        run_full_training
        ;;
    ood)
        run_ood_evaluation
        ;;
    aggregate)
        run_aggregate_results
        ;;
    status)
        show_status
        ;;
    all)
        log "Running full pipeline: sanity → train → ood → aggregate"
        log ""
        run_sanity
        log ""
        log "⚠️  Sanity runs launched. After they complete, re-run with 'train' then 'ood'."
        log "   Or use a single screen/tmux session and call each phase sequentially."
        ;;
    help|*)
        echo "Usage: $0 {sanity|train|ood|aggregate|status|all}"
        echo ""
        echo "  sanity    — Run sanity check (cartpole, 1 seed, DV3 + SF-RSSM)"
        echo "  train     — Run full B1 training (5 tasks × 3 seeds × 2 systems)"
        echo "  ood       — Run OOD evaluation on all trained models"
        echo "  aggregate — Aggregate all OOD results into summary JSON"
        echo "  status    — Show experiment progress"
        echo "  all       — Start with sanity (manual phase transitions)"
        exit 0
        ;;
esac
