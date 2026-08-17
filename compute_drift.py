#!/usr/bin/env python3
"""
Compute multi-horizon latent rollout drift for r2dreamer checkpoints.

Loads a saved Dreamer model, samples replay data, runs posterior inference
and open-loop imagination, then measures drift at multiple horizons.

Usage:
  python compute_drift.py --ckpt logdir/baseline_dmc_walker_walk/latest.pt \
      --config logdir/baseline_dmc_walker_walk/.hydra/config.yaml \
      --output drift_results.json
"""

import argparse, json, sys, os, warnings
from pathlib import Path
import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, '/home/zhengkai/r2dreamer')

from dreamer import Dreamer
from envs import make_envs
from buffer import Buffer
import tools
from omegaconf import OmegaConf


def load_model(config_path, ckpt_path, device='cuda:0'):
    """Load a trained Dreamer model from config + checkpoint."""
    cfg = OmegaConf.load(config_path)
    model_cfg = cfg.model
    
    _, _, obs_space, act_space = make_envs(cfg.env)
    
    agent = Dreamer(model_cfg, obs_space, act_space).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    agent.load_state_dict(ckpt['agent_state_dict'])
    agent.eval()
    
    return agent, cfg


@torch.no_grad()
def compute_drift_metrics(agent, data, horizons=[5, 10, 20, 50, 100], unimix=0.01):
    """
    Compute multi-horizon latent rollout drift.
    
    For each horizon h, compares posterior latents with imagined latents
    rolled forward h steps in open-loop mode using the actual action sequence.
    
    Returns dict of {metric_name: float}
    """
    device = next(agent.parameters()).device
    B, T = data['image'].shape[:2]
    T = min(T, max(horizons) + 20)  # Don't need full sequence
    
    # Get posterior latents
    embed = agent.encoder(data)
    initial = agent.get_initial_state(B)
    post_stoch, post_deter, post_logit = agent.rssm.observe(
        embed[:, :T], data['action'][:, :T], initial, data['is_first'][:, :T])
    
    # Open-loop imagination from t=0 with actual actions
    carry_stoch = post_stoch[:, 0].clone()
    carry_deter = post_deter[:, 0].clone()
    
    imag_logits, imag_deters = [], []
    for t in range(T):
        action_t = data['action'][:, t]
        carry_stoch, carry_deter, logit_t = agent.rssm.img_step(
            carry_stoch, carry_deter, action_t)
        imag_logits.append(logit_t)
        imag_deters.append(carry_deter)
    
    imag_logit = torch.stack(imag_logits, dim=1)
    imag_deter = torch.stack(imag_deters, dim=1)
    
    # Compute KL and cosine drift at each horizon
    metrics = {}
    nclass = post_logit.shape[-1]
    
    for h in horizons:
        if T <= h:
            metrics[f'drift_kl_h{h}'] = float('nan')
            metrics[f'drift_cos_h{h}'] = float('nan')
            continue
        
        # Compare posterior[t>=h] vs imagined[t>=h]
        post_h = post_logit[:, h:]  # (B, T-h, S, K)
        imag_h = imag_logit[:, h:]  # (B, T-h, S, K)
        
        # KL drift
        post_prob = torch.softmax(post_h, dim=-1)
        log_post = torch.log_softmax(post_h, dim=-1)
        prior_prob = torch.softmax(imag_h, dim=-1)
        prior_prob = (1 - unimix) * prior_prob + unimix / nclass
        log_prior = torch.log(prior_prob + 1e-8)
        kl = torch.sum(post_prob * (log_post - log_prior), dim=-1).mean().item()
        metrics[f'drift_kl_h{h}'] = kl
        
        # Cosine drift in deter space
        det_r = post_deter[:, h:]
        det_i = imag_deter[:, h:]
        eps = 1e-8
        r_n = det_r / (torch.norm(det_r, dim=-1, keepdims=True) + eps)
        i_n = det_i / (torch.norm(det_i, dim=-1, keepdims=True) + eps)
        cos_dist = (1.0 - torch.sum(r_n * i_n, dim=-1)).mean().item()
        metrics[f'drift_cos_h{h}'] = cos_dist
    
    # Also compute standard losses for comparison
    _, prior_logit = agent.rssm.prior(post_deter)
    dyn_kl, rep_kl = agent.rssm.kl_loss(post_logit, prior_logit, agent.kl_free)
    metrics['loss_dyn'] = dyn_kl.mean().item()
    metrics['loss_rep'] = rep_kl.mean().item()
    
    return metrics


def load_replay_data(replay_buffer, batch_size=16, batch_length=64, device='cuda:0'):
    """Sample a batch of data from the replay buffer."""
    try:
        batch = replay_buffer.sample(batch_size, batch_length)
        return batch
    except Exception as e:
        print(f"  Could not sample from buffer: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--replay', type=str, default=None,
                        help='Path to saved replay buffer (optional)')
    parser.add_argument('--output', type=str, default='drift_results.json')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--n-batches', type=int, default=5,
                        help='Number of replay batches to average over')
    
    args = parser.parse_args()
    
    print(f"Loading model: {args.ckpt}")
    agent, cfg = load_model(args.config, args.ckpt, args.device)
    
    # Create replay buffer and try to collect some data
    print("Setting up replay buffer...")
    replay_buffer = Buffer(cfg.buffer)
    
    # If we have a saved buffer, load it
    if args.replay and os.path.exists(args.replay):
        print(f"Loading replay buffer: {args.replay}")
        replay_buffer.load(args.replay)
    
    # Collect some fresh data if buffer is empty
    if replay_buffer.count() < args.batch_size * 2:
        print("Collecting fresh environment data...")
        train_envs, _, _, _ = make_envs(cfg.env)
        tools.set_seed_everywhere(cfg.seed)
        
        done = torch.ones(cfg.env.env_num, dtype=torch.bool, device=args.device)
        agent_state = agent.get_initial_state(cfg.env.env_num)
        act = agent_state['prev_action'].clone()
        
        for step in range(5000):  # Collect 5k steps
            act_cpu = act.to('cpu')
            done_cpu = done.to('cpu')
            trans_cpu, done_cpu = train_envs.step(act_cpu, done_cpu)
            trans = trans_cpu.to(args.device, non_blocking=True)
            done = done_cpu.to(args.device)
            trans['action'] = act
            replay_buffer.add(trans, done)
            act, agent_state = agent.act(trans, agent_state, eval=False)
            
            if step % 1000 == 0 and step > 0:
                print(f"  Collected {replay_buffer.count()} transitions...")
        
        train_envs.close()
    
    # Reset agent to eval mode
    agent.eval()
    
    # Compute drift over multiple batches
    print(f"\nComputing drift metrics ({args.n_batches} batches)...")
    all_metrics = []
    
    for i in range(args.n_batches):
        if replay_buffer.count() < args.batch_size * args.batch_length:
            print(f"  Not enough data in buffer ({replay_buffer.count()} transitions)")
            break
        
        data = replay_buffer.sample(args.batch_size, args.batch_length)
        metrics = compute_drift_metrics(agent, data)
        all_metrics.append(metrics)
        print(f"  Batch {i+1}/{args.n_batches}: "
              f"KL5={metrics.get('drift_kl_h5', float('nan')):.4f} "
              f"KL20={metrics.get('drift_kl_h20', float('nan')):.4f} "
              f"KL50={metrics.get('drift_kl_h50', float('nan')):.4f}")
    
    # Average over batches
    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0]:
            vals = [m[key] for m in all_metrics if key in m and not np.isnan(m[key])]
            avg_metrics[key] = float(np.mean(vals)) if vals else float('nan')
        
        # Save
        with open(args.output, 'w') as f:
            json.dump({
                'ckpt': args.ckpt,
                'n_batches': len(all_metrics),
                'buffer_size': replay_buffer.count(),
                'metrics': avg_metrics,
            }, f, indent=2)
        
        print(f"\nAveraged drift metrics saved to {args.output}")
        for k, v in sorted(avg_metrics.items()):
            print(f"  {k}: {v:.4f}")
    else:
        print("No metrics computed (empty buffer)")


if __name__ == '__main__':
    main()
