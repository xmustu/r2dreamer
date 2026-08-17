# Change Log: Ensemble RSSM Prior

## 2026-06-08: Initial implementation of Ensemble RSSM Prior

### New Files
- `ensemble_rssm.py` — EnsembleRSSM class with K prior heads, JS divergence disagreement metric

### Modified Files
- `dreamer.py` — Added `ensemble_enabled` flag, ensemble prior computation in `_cal_grad`,
  ensemble disagreement tracking in `_imagine`, actor disagreement penalty

### New Config
- `configs/model/size12M_ensemble.yaml` — Config for K=3 ensemble with decoder reconstruction

### Architecture
DreamerV3 (unchanged):  encoder → GRU → posterior → decoder / reward / cont
                                         → prior_head_0 (NEW)
                                         → prior_head_1 (NEW)
                                         → prior_head_2 (NEW)
Only the prior is modified: K=3 independent MLP heads on shared deter state.
Total added params: ~2%.

### Training
- All K heads trained with same KL(posterior || prior_k) loss
- Average logits across heads used as the main prior for decoding/value
- Frozen random prior functions (0.1 scale) added for head diversity

### Imagination (Actor-Critic)
- Each imagined step: all K heads predict → disagreement = JS divergence across heads
- Actor advantage: adv - lambda * disagreement
- Lambda = fixed 0.1 for M0 (will add linear ramp schedule in full version)

### Key Differences from Prior Work
- vs cRSSM (2024): No ground-truth context needed
- vs DALI (2025): No separate context encoder — disagreement IS the detection signal
- vs Plan2Explore (2020): Disagreement penalizes actor exploitation, not exploration reward
- vs DreamerV3-XP (2025): Prior-head-only ensemble (2% params), not full model (5x params)

### TODO for Full Version
- [ ] Bootstrap masks for per-head KL (Bernoulli 0.7 per sequence)
- [ ] Linear ramp schedule for lambda (0 → lambda_max over 10-30% of training)
- [ ] EMA quantile normalization of disagreement
- [ ] Detection AUROC analysis on OOD evaluation
- [ ] K=5 ensemble (vs K=3 for M0)

---

## 2026-06-10: Nominal-Anchored Dynamics Randomization (NaS)

### New Files
- `envs/randomized_dynamics.py` — RandomizedDynamicsWrapper: perturbs physics params on reset, tracks dynamics_mode

### Modified Files
- `envs/__init__.py` — Added RandomizedDynamicsWrapper to DMC env creation chain
- `dreamer.py` — Added teacher loading (_load_teacher), distillation loss in _cal_grad
- `configs/env/dmc_vision.yaml` — Added randomize_dynamics_prob config field

### New Config
- `configs/model/size12M_nas.yaml` — NaS config with distill_policy/value loss scales

### Usage
```bash
# Pre-train teacher (if needed):
python train.py env.task=dmc_cheetah_run model=size12M_dv3 seed=42

# Train NaS student:
python train.py model=size12M_nas env.randomize_dynamics_prob=0.7 \
    teacher_checkpoint=logdir/v2/size12M_dv3_cheetah_run_s42/latest.pt
```
