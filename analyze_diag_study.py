#!/usr/bin/env python3
"""
Post-experiment analysis: compute drift metrics from checkpoints
and correlate with eval returns across all experiment runs.
"""
import json, os, sys, glob, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/zhengkai/r2dreamer')

def collect_all_metrics(base_dir='/home/zhengkai/logdir/diag_study'):
    """Collect eval scores and training metrics from all experiments."""
    results = {}
    for exp_dir in sorted(Path(base_dir).iterdir()):
        if not exp_dir.is_dir(): continue
        metrics_file = exp_dir / 'metrics.jsonl'
        if not metrics_file.exists(): continue
        
        env_name = exp_dir.name
        eval_scores = []
        train_losses = defaultdict(list)
        
        with open(metrics_file) as f:
            for line in f:
                data = json.loads(line)
                if 'episode/eval_score' in data:
                    eval_scores.append({
                        'step': data.get('step', 0),
                        'score': data['episode/eval_score'],
                    })
                for key in ['loss/dyn', 'loss/rep', 'loss/rec', 'loss/rew',
                           'dyn_ent', 'rep_ent']:
                    if key in data:
                        train_losses[key].append({
                            'step': data.get('step', 0),
                            'value': data[key],
                        })
        
        if eval_scores:
            results[env_name] = {
                'eval_scores': eval_scores,
                'best_score': max(s['score'] for s in eval_scores),
                'final_score': eval_scores[-1]['score'],
                'n_evals': len(eval_scores),
                'train_losses': dict(train_losses),
            }
    
    return results

def main():
    print('=== Dreamer Diagnostic Study Results ===')
    results = collect_all_metrics()
    
    if not results:
        print('No results found yet.')
        return
    
    print(f'\nFound {len(results)} experiments:\n')
    print(f'{"Experiment":<35s} {"Evals":>5s} {"Best":>10s} {"Final":>10s}')
    print('-' * 65)
    for name, data in sorted(results.items()):
        print(f'{name:<35s} {data["n_evals"]:>5d} {data["best_score"]:>10.2f} {data["final_score"]:>10.2f}')
    
    # Save summary
    out_path = '/home/zhengkai/logdir/diag_study/study_summary.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nSummary saved to {out_path}')

if __name__ == '__main__':
    main()
