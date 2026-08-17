"""Analyze existing DreamerV3 experiments for diagnostic correlations."""
import json, os, sys, numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr

LOG_BASE = '/home/zhengkai/r2dreamer/logdir/v2'

EXPERIMENTS = {
    'dv3_walker': f'{LOG_BASE}/size12M_dv3_walker_walk_s42',
    'dv3_cheetah': f'{LOG_BASE}/size12M_dv3_cheetah_run_s42',
    'dv3_cartpole': f'{LOG_BASE}/size12M_dv3_cartpole_swingup_s42',
    'sf_walker': f'{LOG_BASE}/size12M_sf_walker_walk_s42',
    'sf_cheetah': f'{LOG_BASE}/size12M_sf_cheetah_run_s42',
    'sf_center_walker': f'{LOG_BASE}/size12M_sf_center_1M_walker_s42',
    'sf_center_cheetah': f'{LOG_BASE}/size12M_sf_center_1M_cheetah_s42',
    'sf_center_cartpole': f'{LOG_BASE}/size12M_sf_center_1M_cartpole_s42',
}

# Metrics to correlate with eval returns
LOSS_METRICS = [
    'train/loss/dyn', 'train/loss/rep', 'train/loss/image',
    'train/loss/rew', 'train/loss/value', 'train/loss/policy',
]
ENTROPY_METRICS = ['train/dyn_entropy', 'train/rep_entropy']
VALUE_METRICS = ['train/ret', 'train/val', 'train/adv', 'train/adv_std']
ALL_METRICS = LOSS_METRICS + ENTROPY_METRICS + VALUE_METRICS

def collect_data(logdir):
    metrics_path = Path(logdir) / 'metrics.jsonl'
    if not metrics_path.exists():
        return None
    
    records = []
    with open(metrics_path) as f:
        for line in f:
            records.append(json.loads(line))
    
    # Align eval scores with training metrics
    eval_steps = {}  # step -> score
    for r in records:
        if 'episode/eval_score' in r:
            eval_steps[r['step']] = r['episode/eval_score']
    
    # For each eval step, find the closest preceding training metrics
    train_by_step = {}
    for r in records:
        if 'train/opt/updates' in r:
            train_by_step[r['step']] = r
    
    # Build aligned data: for each eval, collect preceding training metrics
    aligned = []
    for eval_step, eval_score in sorted(eval_steps.items()):
        # Find closest training step before this eval
        prev_steps = [s for s in train_by_step if s < eval_step]
        if not prev_steps:
            continue
        closest = max(prev_steps)
        train_data = train_by_step[closest]
        
        row = {'eval_step': eval_step, 'eval_score': eval_score}
        for m in ALL_METRICS:
            if m in train_data:
                row[m] = train_data[m]
        aligned.append(row)
    
    return aligned

def compute_correlations(aligned_data):
    if len(aligned_data) < 5:
        return []
    
    results = []
    eval_scores = np.array([d['eval_score'] for d in aligned_data])
    
    for metric in ALL_METRICS:
        values = np.array([d.get(metric, np.nan) for d in aligned_data])
        valid = np.isfinite(values)
        if np.sum(valid) < 5 or np.std(values[valid]) < 1e-10:
            continue
        
        rho, p = spearmanr(values[valid], eval_scores[valid])
        results.append({
            'metric': metric.replace('train/', ''),
            'rho': float(rho), 'p': float(p),
            'sig': p < 0.05,
        })
    
    results.sort(key=lambda x: abs(x['rho']), reverse=True)
    return results

# Main analysis
print("=" * 80)
print("WORLD MODEL DIAGNOSTIC CORRELATION ANALYSIS")
print("Existing DreamerV3 Experiments")
print("=" * 80)

all_results = {}
for name, logdir in EXPERIMENTS.items():
    data = collect_data(logdir)
    if not data:
        print(f"\n{name}: NO DATA")
        continue
    
    corrs = compute_correlations(data)
    all_results[name] = corrs
    
    scores = [d['eval_score'] for d in data]
    print(f"\n{name}: {len(data)} aligned eval points")
    print(f"  Score range: {min(scores):.1f} → {max(scores):.1f}")
    print(f"  {'Metric':<20s} {'ρ':>7s} {'p':>8s} {'Sig':>5s}")
    print(f"  {'-'*40}")
    for c in corrs[:8]:
        sig = '**' if c['sig'] else ''
        print(f"  {c['metric']:<20s} {c['rho']:>+7.3f} {c['p']:>8.4f} {sig:>5s}")

# Cross-experiment summary
print(f"\n{'='*80}")
print("CROSS-EXPERIMENT SUMMARY")
print(f"{'='*80}")

# Which metric is best predictor in most experiments?
best_count = defaultdict(int)
for name, corrs in all_results.items():
    if corrs:
        best_count[corrs[0]['metric']] += 1

print("\nBest predictor counts:")
for metric, count in sorted(best_count.items(), key=lambda x: -x[1]):
    print(f"  {metric}: {count} experiments")

# Loss vs entropy comparison
loss_win = entropy_win = 0
for name, corrs in all_results.items():
    if not corrs: continue
    loss_rhos = [abs(c['rho']) for c in corrs if c['metric'].startswith('loss/')]
    entropy_rhos = [abs(c['rho']) for c in corrs if 'entropy' in c['metric']]
    best_loss = max(loss_rhos) if loss_rhos else 0
    best_entropy = max(entropy_rhos) if entropy_rhos else 0
    if best_entropy > best_loss:
        entropy_win += 1
    else:
        loss_win += 1

print(f"\nEntropy beats loss: {entropy_win}/{entropy_win + loss_win} experiments")
print(f"Loss beats entropy: {loss_win}/{entropy_win + loss_win} experiments")
