#!/usr/bin/env python3
"""
BYOL-RSSM: EMA Posterior-Prior Consistency for Decoder-Free World Models.
Core module: EMA teacher, L_TS loss, collapse diagnostics.
Drop-in for DreamerV3 on remote server (r2dreamer codebase).
"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from copy import deepcopy
from collections import defaultdict

class BYOLRSSMLoss(nn.Module):
    """L_TS = (1/N) Σ_t [1 − cos_sim(pred_t, target_t)]"""
    def __init__(self, beta_ts=0.05):
        super().__init__()
        self.beta_ts = beta_ts
    
    def forward(self, prior_logits, teacher_logits, online_proj, ema_proj, return_diag=False):
        B, T, S, K = prior_logits.shape
        N = B * T
        prior_flat = prior_logits.reshape(N, S * K)
        teacher_flat = teacher_logits.reshape(N, S * K)
        pred = online_proj(prior_flat)
        with torch.no_grad():
            target = ema_proj(teacher_flat)
        cos_sim = F.cosine_similarity(pred, target, dim=-1)
        loss = self.beta_ts * (1.0 - cos_sim).mean()
        if return_diag:
            return loss, self._diagnostics(pred, target, prior_logits, teacher_logits)
        return loss
    
    def _diagnostics(self, pred, target, prior_logits, teacher_logits):
        with torch.no_grad():
            teacher_std = target.std(dim=0).mean().item()
            _, S, _ = torch.linalg.svd(pred.float(), full_matrices=False)
            S_norm = S / (S.sum() + 1e-10)
            eff_rank = torch.exp(-(S_norm * torch.log(S_norm + 1e-10)).sum()).item()
            prior_ent = -(F.softmax(prior_logits, dim=-1) * torch.log(F.softmax(prior_logits, dim=-1) + 1e-10)).sum(dim=-1).mean().item()
            teacher_ent = -(F.softmax(teacher_logits, dim=-1) * torch.log(F.softmax(teacher_logits, dim=-1) + 1e-10)).sum(dim=-1).mean().item()
            cos_mean = F.cosine_similarity(pred, target, dim=-1).mean().item()
            return {'teacher_std': teacher_std, 'effective_rank': eff_rank,
                    'prior_entropy': prior_ent, 'teacher_entropy': teacher_ent,
                    'cos_sim_mean': cos_mean,
                    'healthy': teacher_std > 0.05 and eff_rank > 1.0 and prior_ent > 0.5}

class BYOLRSSMTrainer:
    """Manages EMA teacher + L_TS loss. Zero new trainable parameters."""
    
    def __init__(self, agent, beta_ts=0.05, tau_start=0.99, tau_end=0.999, device='cuda:0'):
        self.agent = agent
        self.device = device
        self.tau = tau_start
        self.tau_start = tau_start
        self.tau_end = tau_end
        # Deep copy encoder, key RSSM params, projector
        self.ema_encoder = deepcopy(agent.encoder)
        self.ema_encoder.requires_grad_(False)
        # Store ema posterior-related params from RSSM
        self.ema_post_params = {}
        for name, p in agent.rssm.named_parameters():
            if 'obs' in name or '_img_' in name or 'post' in name.lower():
                self.ema_post_params[name] = p.data.clone()
        # EMA projector
        self.ema_projector = deepcopy(agent.projector) if hasattr(agent, 'projector') else nn.Identity()
        self.ema_projector.requires_grad_(False)
        self.loss_fn = BYOLRSSMLoss(beta_ts=beta_ts)
        self.step = 0
    
    def update_tau(self, step, total_steps):
        self.step = step
        progress = min(step / max(total_steps, 1), 1.0)
        self.tau = self.tau_end - (self.tau_end - self.tau_start) * (1 + np.cos(np.pi * progress)) / 2
    
    def update_ema(self):
        """θ_ema ← τ·θ_ema + (1-τ)·θ"""
        with torch.no_grad():
            for (n1, p1), (n2, p2) in zip(self.ema_encoder.named_parameters(), self.agent.encoder.named_parameters()):
                p1.data.copy_(self.tau * p1.data + (1 - self.tau) * p2.data)
            for name, p in self.agent.rssm.named_parameters():
                if name in self.ema_post_params:
                    self.ema_post_params[name].copy_(self.tau * self.ema_post_params[name] + (1 - self.tau) * p.data)
            for (n1, p1), (n2, p2) in zip(self.ema_projector.named_parameters(), self.agent.projector.named_parameters()):
                p1.data.copy_(self.tau * p1.data + (1 - self.tau) * p2.data)
    
    @torch.no_grad()
    def get_teacher_logits(self, obs, h_t_detached):
        """Compute EMA teacher logits from observations. Full stop-grad."""
        B, T = obs['image'].shape[:2]
        obs_flat = obs['image'].reshape(B * T, *obs['image'].shape[2:])
        embed_dict = self.ema_encoder({'image': obs_flat})
        embed = embed_dict['image'].reshape(B, T, -1)
        
        S = self.agent.rssm.stoch
        K = self.agent.rssm.classes
        logits_all = torch.zeros(B, T, S, K, device=embed.device)
        
        for t in range(T):
            _, _, post_logits = self.agent.rssm.obs_step(
                torch.zeros(B, S, K, device=embed.device),
                h_t_detached[:, t],
                torch.zeros(B, self.agent.act_dim, device=embed.device),
                embed[:, t],
                torch.ones(B, dtype=torch.bool, device=embed.device) if t == 0 else torch.zeros(B, dtype=torch.bool, device=embed.device)
            )
            logits_all[:, t] = post_logits
        
        return logits_all  # already under no_grad
    
    def compute_loss(self, prior_logits, teacher_logits):
        """L_TS + diagnostics. prior_logits from online prior, teacher_logits from EMA posterior."""
        return self.loss_fn(prior_logits, teacher_logits, self.agent.projector, self.ema_projector, return_diag=True)

def compute_effective_rank(matrix):
    with torch.no_grad():
        _, S, _ = torch.linalg.svd(matrix.float(), full_matrices=False)
        S_norm = S / (S.sum() + 1e-10)
        return torch.exp(-(S_norm * torch.log(S_norm + 1e-10)).sum()).item()

def log_byol_diagnostics(logger, diag, tau, step):
    for k, v in diag.items():
        if isinstance(v, (int, float, bool)):
            logger.scalar(f'byol/{k}', float(v) if isinstance(v, bool) else v)
    logger.scalar('byol/tau', tau)
    logger.scalar('byol/healthy', 1.0 if diag.get('healthy', False) else 0.0)

print("BYOL-RSSM module loaded. Zero new trainable parameters.")
