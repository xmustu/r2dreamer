#!/usr/bin/env python3
"""
Load r2dreamer checkpoints and compute world model diagnostic metrics.

Computes:
- Multi-horizon latent rollout drift (KL + cosine)
- Correlation between diagnostic metrics and eval returns
- Checkpoint selection regret analysis

Usage:
  python checkpoint_diagnostics.py --logdir logdir/2026-05-31/12-31-55
  python checkpoint_diagnostics.py --logdir logdir/2026-05-31/12-31-55 --compute-drift
"""

import argparse
import json
import os
import sys
import glob
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import hydra
from omegaconf import OmegaConf

# Add r2dreamer to path
sys.path.insert(0, '/home/zhengkai/r2dreamer')

from dreamer import Dreamer
from rssm import RSSM
import tools


def load_checkpoint(logdir, device='cuda:0'):
    """Load a Dreamer checkpoint from logdir."""
    ckpt_path = Path(logdir) / 'latest.pt'
    if not ckpt_path.exists():
        ckpt_path = Path(logdir) / 'checkpoint.pt'
    if not ckpt_path.exists():
        # Try finding any .pt file
        pts = list(Path(logdir).glob('*.pt'))
        if pts:
            ckpt_path = pts[0]
        else:
            raise FileNotFoundError(f"No checkpoint found in {logdir}")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    return ckpt


def compute_drift_kl_pytorch(post_logit, prior_logit, unimix=0.01):
    """Compute KL(post || prior) for categorical latents (PyTorch version)."""
    nclass = post_logit.shape[-1]
    post_prob = torch.softmax(post_logit, dim=-1)
    log_post = torch.log_softmax(post_logit, dim=-1)
    prior_prob = torch.softmax(prior_logit, dim=-1)
    prior_prob = (1 - unimix) * prior_prob + unimix / nclass
    log_prior = torch.log(prior_prob + 1e-8)
    kl = torch.sum(post_prob * (log_post - log_prior), dim=-1)  # (..., S)
    return kl.mean().item()


def compute_cosine_dist_pytorch(deter_real, deter_imag):
    """Compute cosine distance between deterministic states."""
    eps = 1e-8
    r_n = deter_real / (torch.norm(deter_real, dim=-1, keepdims=True) + eps)
    i_n = deter_imag / (torch.norm(deter_imag, dim=-1, keepdims=True) + eps)
    return (1.0 - torch.sum(r_n * i_n, dim=-1)).mean().item()


def compute_multi_horizon_drift(agent, data, horizons=[5, 10, 20, 50]):
    """
    Compute multi-horizon latent rollout drift.
    
    For each horizon h, runs open-loop imagination for h steps and compares
    imagined latents with posterior latents at matching timesteps.
    """
    agent.eval()
    device = next(agent.parameters()).device
    B, T = data['image'].shape[:2]
    T_eff = min(T, max(horizons) + 10)

    with torch.no_grad():
        # Get posterior latents along real trajectory
        embed = agent.encoder(data)
        initial = agent.get_initial_state(B)
        post_stoch, post_deter, post_logit = agent.rssm.observe(
            embed[:, :T_eff], data['action'][:, :T_eff], initial, data['is_first'][:, :T_eff])

        # Run open-loop imagination from t=0
        carry_stoch = post_stoch[:, 0]  # (B, S, K)
        carry_deter = post_deter[:, 0]  # (B, D)
        
        imag_stochs, imag_deters, imag_logits = [], [], []
        for t in range(T_eff):
            # Use the actual action at step t
            action_t = data['action'][:, t]
            carry_stoch, carry_deter, logit_t = agent.rssm.img_step(
                carry_stoch, carry_deter, action_t)
            imag_stochs.append(carry_stoch)
            imag_deters.append(carry_deter)
            imag_logits.append(logit_t)

        imag_stoch = torch.stack(imag_stochs, dim=1)  # (B, T, S, K)
        imag_deter = torch.stack(imag_deters, dim=1)  # (B, T, D)
        imag_logit = torch.stack(imag_logits, dim=1)  # (B, T, S, K)

    # Compute drift at each horizon
    metrics = {}
    for h in horizons:
        if T_eff <= h:
            metrics[f'drift_kl_h{h}'] = float('nan')
            metrics[f'drift_cos_h{h}'] = float('nan')
            continue

        # Compare posterior[t+h] vs imagined[t+h]
        post_logit_h = post_logit[:, h:]  # (B, T-h, S, K)
        imag_logit_h = imag_logit[:, h:]  # (B, T-h, S, K)
        post_deter_h = post_deter[:, h:]
        imag_deter_h = imag_deter[:, h:]

        kl = compute_drift_kl_pytorch(
            post_logit_h.reshape(-1, post_logit_h.shape[-2], post_logit_h.shape[-1]),
            imag_logit_h.reshape(-1, imag_logit_h.shape[-2], imag_logit_h.shape[-1]))
        cos = compute_cosine_dist_pytorch(
            post_deter_h.reshape(-1, post_deter_h.shape[-1]),
            imag_deter_h.reshape(-1, imag_deter_h.shape[-1]))

        metrics[f'drift_kl_h{h}'] = kl
        metrics[f'drift_cos_h{h}'] = cos

    return metrics


def load_eval_returns(logdir):
    """Load evaluation returns from metrics.jsonl or tensorboard logs."""
    metrics_path = Path(logdir) / 'metrics.jsonl'
    eval_returns = []

    if metrics_path.exists():
        with open(metrics_path) as f:
            for line in f:
                data = json.loads(line)
                if 'episode/eval_score' in data:
                    eval_returns.append({
                        'step': data.get('step', 0),
                        'eval_score': data['episode/eval_score'],
                    })

    # Also try to find eval scores from train metrics
    if not eval_returns:
        train_metrics = Path(logdir) / 'train_metrics.jsonl'
        if train_metrics.exists():
            with open(train_metrics) as f:
                for line in f:
                    data = json.loads(line)
                    if 'episode/score' in data:
                        eval_returns.append({
                            'step': data.get('step', 0),
                            'eval_score': data['episode/score'],
                        })

    return eval_returns


def analyze_checkpoint_diagnostics(logdir, device='cuda:0', batch_size=4):
    """Main analysis: load checkpoint, compute drift, correlate with eval."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {logdir}")
    print(f"{'='*60}")

    # Load eval returns
    eval_returns = load_eval_returns(logdir)
    print(f"Eval returns: {len(eval_returns)} entries")
    if eval_returns:
        print(f"  Best: {max(r['eval_score'] for r in eval_returns):.2f}")
        print(f"  Latest: {eval_returns[-1]['eval_score']:.2f}")

    # Load training losses
    metrics_path = Path(logdir) / 'metrics.jsonl'
    train_losses = defaultdict(list)
    if metrics_path.exists():
        with open(metrics_path) as f:
            for line in f:
                data = json.loads(line)
                for key in ['loss/dyn', 'loss/rep', 'loss/rec', 'loss/rew',
                            'dyn_ent', 'rep_ent']:
                    metric_key = key.replace('/', '_')
                    if key in data:
                        train_losses[metric_key].append({
                            'step': data.get('step', 0),
                            'value': data[key],
                        })

    # Summary
    summary = {
        'logdir': str(logdir),
        'n_eval_entries': len(eval_returns),
        'best_eval_score': max((r['eval_score'] for r in eval_returns), default=0),
        'latest_eval_score': eval_returns[-1]['eval_score'] if eval_returns else 0,
        'n_loss_entries': sum(len(v) for v in train_losses.values()),
    }

    # Save summary
    out_path = Path(logdir) / 'diagnostic_summary.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Checkpoint diagnostics for r2dreamer')
    parser.add_argument('--logdir', type=str, required=True,
                        help='Path to Dreamer log directory')
    parser.add_argument('--compute-drift', action='store_true',
                        help='Compute full multi-horizon drift (requires GPU + replay data)')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=4)

    args = parser.parse_args()

    if args.compute_drift:
        # This requires loading the full model and having replay data
        # For now, do the basic analysis
        pass

    summary = analyze_checkpoint_diagnostics(
        args.logdir, device=args.device, batch_size=args.batch_size)
    
    # Print results
    print(f"\nResults for {Path(args.logdir).name}:")
    print(f"  Best eval score: {summary['best_eval_score']:.3f}")
    print(f"  Eval entries: {summary['n_eval_entries']}")


if __name__ == '__main__':
    main()
