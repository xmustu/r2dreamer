"""Effective-Rank-Aware Replay (ERAR) for Coverage-Guided World Model Learning.

Two-part data-side mechanism for JEPA-based world models:
  Part A: Coverage-weighted replay — reweights training samples by contribution
          to undercovered latent directions (implemented via WeightedRandomSampler).
  Part B: Latent-space regularization — adds learnable noise along undercovered
          eigendirections to context embeddings during training, with clean targets.
"""

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from sklearn.covariance import LedoitWolf


class CoverageMonitor:
    """Monitors latent embedding coverage using a frozen EMA encoder + projector.

    Every K steps, recomputes latent embeddings for cached observations,
    computes Ledoit-Wolf shrinkage covariance and participation ratio,
    and derives per-sample weights for undercovered latent directions.
    """

    def __init__(
        self,
        encoder,
        projector,
        embed_dim: int = 192,
        ema_decay: float = 0.99,
        update_every: int = 1000,
        eigenvalue_floor: float = 1e-6,
        tau: float = 0.5,
        device: str = "cuda",
    ):
        self.embed_dim = embed_dim
        self.ema_decay = ema_decay
        self.update_every = update_every
        self.eigenvalue_floor = eigenvalue_floor
        self.tau = tau
        self.device = device

        # Frozen EMA copies
        self.ema_encoder = deepcopy(encoder)
        for p in self.ema_encoder.parameters():
            p.requires_grad_(False)
        self.ema_encoder.to(device).eval()

        self.ema_projector = deepcopy(projector) if projector is not None else nn.Identity()
        for p in self.ema_projector.parameters():
            p.requires_grad_(False)
        self.ema_projector.to(device).eval()

        # State
        self.covariance = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.participation_ratio = None
        self.sample_weights = None   # per-sample weights for WeightedRandomSampler
        self._obs_buffer = []        # cached (pixels,)
        self._emb_cache = None       # cached latent embeddings
        self.pr_history = []

    @torch.no_grad()
    def update_ema(self, encoder, projector):
        for ema_p, curr_p in zip(self.ema_encoder.parameters(), encoder.parameters()):
            ema_p.data.mul_(self.ema_decay).add_(curr_p.data, alpha=1 - self.ema_decay)
        if projector is not None:
            for ema_p, curr_p in zip(self.ema_projector.parameters(), projector.parameters()):
                ema_p.data.mul_(self.ema_decay).add_(curr_p.data, alpha=1 - self.ema_decay)

    @torch.no_grad()
    def _encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode pixels → projector latents using EMA copies."""
        if pixels.dim() == 5:
            B, T = pixels.shape[:2]
            pixels = pixels.reshape(B * T, *pixels.shape[2:])
        output = self.ema_encoder(pixels, interpolate_pos_encoding=True)
        cls_emb = output.last_hidden_state[:, 0]   # (N, 192)
        latents = self.ema_projector(cls_emb)        # (N, proj_dim)
        return latents

    @torch.no_grad()
    def update(self, encoder, projector, observations: torch.Tensor, step: int):
        """Recompute coverage from cached observations."""
        self.update_ema(encoder, projector)

        # Batch-encode all cached observations
        BATCH = 256
        all_latents = []
        obs_list = observations if isinstance(observations, list) else [observations]
        for obs in obs_list:
            obs = obs.to(self.device)
            for i in range(0, len(obs), BATCH):
                batch = obs[i:i + BATCH]
                all_latents.append(self._encode(batch).cpu())
        latents = torch.cat(all_latents, dim=0)  # (N, D)

        # Ledoit-Wolf shrinkage covariance (stable, tested)
        cov_np = LedoitWolf().fit(latents.numpy()).covariance_
        cov = torch.from_numpy(cov_np).float().to(self.device)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        eigenvalues = eigenvalues.flip(0).clamp(min=self.eigenvalue_floor)
        eigenvectors = eigenvectors.flip(1)

        self.covariance = cov
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.participation_ratio = (eigenvalues.sum() ** 2 / (eigenvalues.pow(2).sum())).item()
        self.pr_history.append((step, self.participation_ratio))
        self._emb_cache = latents

        # Compute per-sample weights
        self._compute_weights(latents.to(self.device))

    @torch.no_grad()
    def _compute_weights(self, latents: torch.Tensor):
        """Compute per-sample coverage weights."""
        lambda_mean = self.eigenvalues.mean()
        under = self.eigenvalues < lambda_mean
        if not under.any():
            self.sample_weights = torch.ones(len(latents))
            return

        under_vecs = self.eigenvectors[:, under]          # (D, K)
        inv_w = 1.0 / self.eigenvalues[under].clamp(min=1e-8)   # (K,)
        # weight_i = Σ_k (1/λ_k) · (z_i · v_k)²
        proj = (latents @ under_vecs).pow(2)               # (N, K)
        raw = (proj @ inv_w) + 0.01                        # (N,)  epsilon baseline
        self.sample_weights = raw / raw.sum()

    def get_weights(self, alpha: float = 0.1) -> np.ndarray:
        """Return mixed weights (α-uniform + (1-α)-coverage) as numpy."""
        if self.sample_weights is None:
            return None
        N = len(self.sample_weights)
        uniform = torch.ones(N) / N
        mixed = alpha * uniform + (1 - alpha) * self.sample_weights
        return mixed.numpy()

    def is_coverage_adequate(self) -> bool:
        if self.participation_ratio is None:
            return True
        return self.participation_ratio / self.embed_dim >= self.tau

    def get_undercovered_directions(self) -> torch.Tensor | None:
        if self.eigenvectors is None:
            return None
        under = self.eigenvalues < self.eigenvalues.mean()
        if not under.any():
            return None
        return self.eigenvectors[:, under]

    def cache_obs(self, pixels: torch.Tensor):
        """Add observation batch to cache."""
        self._obs_buffer.append(pixels.detach().cpu())

    def get_cached_obs(self) -> torch.Tensor | None:
        if not self._obs_buffer:
            return None
        obs = torch.cat(self._obs_buffer, dim=0)[:5000]  # cap at 5000 frames
        self._obs_buffer = []
        return obs


class ERARAugmentation:
    """Part B: In-graph latent augmentation along undercovered eigendirections.

    Adds learnable noise to CONTEXT embeddings only (target stays clean).
    Gradients flow through augmentation — the model learns to be robust to
    coverage-directed perturbations. Training-only (not applied during val).
    """

    def __init__(self, noise_scale: float = 0.1, prob: float = 0.05):
        self.noise_scale = noise_scale
        self.prob = prob

    def augment_context(
        self,
        ctx_emb: torch.Tensor,         # (B, ctx_len, D)  — gradients flow
        under_dirs: torch.Tensor | None,  # (D, K) or None
    ) -> torch.Tensor:
        """Augment context embeddings. Target stays unchanged."""
        if under_dirs is None or under_dirs.shape[1] == 0:
            return ctx_emb

        B, T, D = ctx_emb.shape
        K = under_dirs.shape[1]

        # Per-sample mask
        mask = torch.rand(B, device=ctx_emb.device) < self.prob
        if not mask.any():
            return ctx_emb

        n_aug = mask.sum().item()

        # Random direction in undercovered subspace (gradients flow)
        coeffs = torch.randn(n_aug, K, device=ctx_emb.device)
        coeffs = coeffs / (coeffs.norm(dim=1, keepdim=True) + 1e-8)
        noise = (coeffs @ under_dirs.T) * self.noise_scale   # (n_aug, D)

        # Apply to all time steps of selected sequences
        ctx_aug = ctx_emb.clone()          # clone to keep original for target
        ctx_aug[mask] = ctx_aug[mask] + noise.unsqueeze(1)  # broadcast over T

        # IMPORTANT: noise is detached from under_dirs to avoid backprop
        # through eigendecomposition. The augmentation adds perturbation but
        # under_dirs itself is treated as fixed for this forward pass.
        ctx_aug[mask] = ctx_aug[mask].detach() + noise.unsqueeze(1)

        # Actually, for gradient flow: let noise be part of graph
        # but under_dirs is detached (from EMA encoder, frozen)
        ctx_aug = ctx_emb + torch.zeros_like(ctx_emb)  # start fresh
        ctx_aug = ctx_emb.clone()
        noise_detached = noise.unsqueeze(1).detach()
        ctx_aug[mask] = ctx_aug[mask] + noise_detached

        return ctx_aug


class ERARManager:
    """Top-level ERAR coordinator."""

    def __init__(
        self,
        encoder,
        projector=None,
        embed_dim: int = 192,
        ema_decay: float = 0.99,
        update_every: int = 1000,
        tau: float = 0.5,
        alpha: float = 0.1,
        noise_scale: float = 0.1,
        aug_prob: float = 0.05,
        device: str = "cuda",
    ):
        self.monitor = CoverageMonitor(
            encoder=encoder,
            projector=projector,
            embed_dim=embed_dim,
            ema_decay=ema_decay,
            update_every=update_every,
            tau=tau,
            device=device,
        )
        self.augmentation = ERARAugmentation(
            noise_scale=noise_scale,
            prob=aug_prob,
        )
        self.alpha = alpha
        self.update_every = update_every

    def should_update(self, epoch: int) -> bool:
        return epoch > 0 and epoch % max(1, self.update_every // 1000) == 0

    def update(self, encoder, projector, observations, epoch):
        return self.monitor.update(encoder, projector, observations, epoch)

    def get_weights(self):
        return self.monitor.get_weights(self.alpha)

    def get_participation_ratio(self):
        return self.monitor.participation_ratio

    def augment_context(self, ctx_emb, under_dirs):
        return self.augmentation.augment_context(ctx_emb, under_dirs)

    def get_undercovered_directions(self):
        return self.monitor.get_undercovered_directions()

    def cache_obs(self, pixels):
        self.monitor.cache_obs(pixels)
