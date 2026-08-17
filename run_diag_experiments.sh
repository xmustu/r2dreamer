#!/bin/bash
# Full diagnostic experiment launcher for r2dreamer
# Usage: bash run_diag_experiments.sh

set -e
export PATH=/home/zhengkai/miniconda3/bin:$PATH
source activate r2dreamer
cd /home/zhengkai/r2dreamer

TIMESTAMP=$(date +%Y-%m-%d/%H-%M-%S)
BASE_LOGDIR="/home/zhengkai/logdir/diag_study/${TIMESTAMP}"

echo "=== Dreamer Diagnostic Study ==="
echo "Logdir: ${BASE_LOGDIR}"
echo "GPUs: 4x A6000"
echo ""

# ============================================================
# MILESTONE 1: Baselines (standard DreamerV3 on multiple envs)
# ============================================================
declare -A BASELINES
BASELINES=(
  ["dmc_walker_walk"]="500000"
  ["dmc_cheetah_run"]="500000"
  ["dmc_cartpole_swingup"]="500000"
  ["dmc_reacher_easy"]="500000"
  ["dmc_finger_spin"]="500000"
)

echo "[M1] Launching baseline experiments..."
GPU=0
for TASK in "${!BASELINES[@]}"; do
  STEPS=${BASELINES[$TASK]}
  LOGDIR="${BASE_LOGDIR}/baseline_${TASK}"
  mkdir -p "${LOGDIR}"

  echo "  GPU ${GPU}: ${TASK} (${STEPS} steps) -> ${LOGDIR}"
  
  nohup python3 train.py \
    env.task=${TASK} \
    env.steps=${STEPS} \
    trainer.eval_every=20000 \
    trainer.update_log_every=5000 \
    logdir=${LOGDIR} \
    seed=42 \
    device=cuda:${GPU} \
    > ${LOGDIR}/console.log 2>&1 &

  GPU=$(( (GPU + 1) % 4 ))
  sleep 5  # stagger launches
done

echo ""
echo "All baseline experiments launched."
echo "Monitor: tail -f ${BASE_LOGDIR}/baseline_*/console.log"
echo ""
echo "After completion, run: python3 analyze_all_checkpoints.py --base-logdir ${BASE_LOGDIR}"
