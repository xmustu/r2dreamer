#!/bin/bash
eval "$(/home/zhengkai/miniconda3/bin/conda shell.bash hook)"
conda activate r2dreamer
cd ~/r2dreamer
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

C="env=dmc_proprio_domain_rand env.task=dmc_cartpole_swingup env.steps=200000 env.env_num=1 env.eval_episode_num=10 model.compile=True"

# GPU 0: DV3 clean baseline (R001)
nohup python train.py model=size12M_dv3 env=dmc_proprio env.task=dmc_cartpole_swingup env.steps=200000 env.env_num=1 env.eval_episode_num=10 model.compile=True buffer.storage_device=cuda:0 seed=42 device=cuda:0 > logs/r001_dv3_clean.log 2>&1 &
echo "R001: $!"

# GPU 1: NoDistill 70/30 (R002)
nohup python train.py model=size12M_domain_mixed $C env.nominal_prob=0.7 buffer.storage_device=cuda:1 seed=42 device=cuda:1 > logs/r002_nodistill.log 2>&1 &
echo "R002: $!"

# GPU 2: Homogeneous (R009)
nohup python train.py model=size12M_domain_homogeneous $C env.nominal_prob=0.7 buffer.storage_device=cuda:2 seed=42 device=cuda:2 > logs/r009_homogeneous.log 2>&1 &
echo "R009: $!"

# GPU 3: Balanced (R010)
nohup python train.py model=size12M_domain_balanced $C env.nominal_prob=0.7 buffer.storage_device=cuda:3 seed=42 device=cuda:3 > logs/r010_balanced.log 2>&1 &
echo "R010: $!"

echo "All 4 launched"
wait
