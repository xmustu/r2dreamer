"""
SPER: Spatial Perception Enhancement for RSSM

Lightweight auxiliary prediction heads attached to RSSM latent state.
- Motion Head: predicts temporal change between consecutive frames
- Depth Head: predicts spatial depth structure

Heads are TRAIN-ONLY -- removed at inference (zero overhead).

Architecture: RSSM latent (feat) -> 2-layer MLP -> prediction target
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialPredictionHead(nn.Module):
    """Single spatial prediction head: feat -> hidden -> hidden -> target_dim."""

    def __init__(self, feat_size, hidden_size, target_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_size, hidden_size, bias=True),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, target_dim, bias=True),
        )

    def forward(self, feat):
        return self.net(feat)


class SPER(nn.Module):
    """SPatial perception Enhancement for RSSM.

    Attaches to RSSM latent state (post-stoch + post-deter concatenated).
    Motion head: predicts consecutive-frame embedding differences.
    Depth head: predicts spatial structure from encoder features.

    All heads use detached features by default for stability.
    Set detach_feat=False to allow gradient flow into RSSM.
    """

    def __init__(self, feat_size, embed_size, hidden_size=256, detach_feat=True):
        super().__init__()
        self.feat_size = feat_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.detach_feat = detach_feat

        # Motion head: predict motion signal from RSSM latent
        self.motion_target_dim = 64
        self.motion_head = SpatialPredictionHead(
            feat_size, hidden_size, self.motion_target_dim
        )

        # Depth head: predict spatial structure from RSSM latent
        self.depth_target_dim = 64
        self.depth_head = SpatialPredictionHead(
            feat_size, hidden_size, self.depth_target_dim
        )

        # Fixed random projections for target generation.
        # Use a dedicated Generator to avoid consuming the global RNG —
        # otherwise SPER's initialization would silently change the random
        # sequence of the rest of the model (breaking baseline comparability).
        gen = torch.Generator()
        gen.manual_seed(0)
        self.register_buffer(
            "_motion_proj",
            torch.randn(embed_size, self.motion_target_dim, generator=gen)
            / (embed_size ** 0.5),
        )
        self.register_buffer(
            "_depth_proj",
            torch.randn(embed_size, self.depth_target_dim, generator=gen)
            / (embed_size ** 0.5),
        )

    def forward(self, feat):
        """Compute prediction outputs from RSSM features."""
        if self.detach_feat:
            feat = feat.detach()
        B, T, F = feat.shape
        feat_flat = feat.reshape(B * T, F)
        motion_pred = self.motion_head(feat_flat).reshape(B, T, self.motion_target_dim)
        depth_pred = self.depth_head(feat_flat).reshape(B, T, self.depth_target_dim)
        return {"motion_pred": motion_pred, "depth_pred": depth_pred}

    def compute_motion_targets(self, embed):
        """Motion targets from encoder embedding differences (t -> t+1)."""
        B, T, E = embed.shape
        diffs = torch.zeros_like(embed)
        diffs[:, :-1] = embed[:, 1:] - embed[:, :-1]
        diffs_norm = F.normalize(diffs.reshape(B * T, E).float(), dim=-1)
        target = torch.mm(diffs_norm, self._motion_proj.to(embed.device))
        return target.reshape(B, T, self.motion_target_dim)

    def compute_depth_targets(self, embed):
        """Depth targets from encoder embeddings (spatial pooling proxy).

        Uses the embedding magnitude as a crude depth/spatial proxy.
        Different spatial configurations produce different embedding patterns.
        This is a weak but safe signal for M0 validation.
        """
        B, T, E = embed.shape
        # Pooling: mean + std of embedding as crude spatial descriptor
        emb_flat = embed.reshape(B * T, E).float()
        # Use the embedding pattern itself as depth proxy
        emb_norm = F.normalize(emb_flat, dim=-1)
        target = torch.mm(emb_norm, self._depth_proj.to(embed.device))
        return target.reshape(B, T, self.depth_target_dim)

    def compute_losses(self, feat, embed, data, lambda_motion=0.01, lambda_depth=0.01):
        """Compute SPER auxiliary losses.

        Gradient flows through the prediction heads. If detach_feat=False,
        gradient also flows into the RSSM latent (regularization mode).

        Targets are ALWAYS detached — they are fixed supervision signals
        derived from the encoder, never trainable.
        """
        losses = {}

        # Predictions: gradient enabled (through heads, optionally into RSSM)
        preds = self.forward(feat)

        # Targets: no gradient (fixed supervision)
        with torch.no_grad():
            motion_target = self.compute_motion_targets(embed)
        losses["sper_motion"] = lambda_motion * F.mse_loss(
            preds["motion_pred"], motion_target
        )

        # Depth prediction
        if lambda_depth > 0:
            with torch.no_grad():
                depth_target = self.compute_depth_targets(embed)
            losses["sper_depth"] = lambda_depth * F.mse_loss(
                preds["depth_pred"], depth_target
            )

        return losses
