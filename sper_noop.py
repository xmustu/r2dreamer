"""
SPER: Spatial Perception Enhancement for RSSM (NO-OP for M0 debugging)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialPredictionHead(nn.Module):
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
    def __init__(self, feat_size, embed_size, hidden_size=256, detach_feat=True):
        super().__init__()
        self.detach_feat = detach_feat
        self.motion_target_dim = 64
        self.depth_target_dim = 64
        self.motion_head = SpatialPredictionHead(feat_size, hidden_size, self.motion_target_dim)
        self.depth_head = SpatialPredictionHead(feat_size, hidden_size, self.depth_target_dim)

    def forward(self, feat):
        B, T, F = feat.shape
        f = feat.detach().reshape(B * T, F)
        return {"motion_pred": self.motion_head(f).reshape(B, T, self.motion_target_dim),
                "depth_pred": self.depth_head(f).reshape(B, T, self.depth_target_dim)}

    def compute_losses(self, feat, embed, data, lambda_motion=0.01, lambda_depth=0.01):
        # NO-OP for M0: just return empty losses
        return {}
