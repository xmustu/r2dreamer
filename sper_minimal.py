"""
Minimal SPER: predict a constant zero target from RSSM features.
Used to isolate the CUDA assert bug.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SPER(nn.Module):
    def __init__(self, feat_size, embed_size, hidden_size=256, detach_feat=True):
        super().__init__()
        self.detach_feat = detach_feat
        self.target_dim = 32

        # Single minimal head: predict zero target
        self.head = nn.Sequential(
            nn.Linear(feat_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, self.target_dim),
        )

    def forward(self, feat):
        if self.detach_feat:
            feat = feat.detach()
        B, T, F = feat.shape
        return self.head(feat.reshape(B * T, F)).reshape(B, T, self.target_dim)

    def compute_losses(self, feat, embed, data, lambda_motion=0.01, lambda_depth=0.01):
        pred = self.forward(feat)
        target = torch.zeros_like(pred)
        return {"sper_test": 0.01 * F.mse_loss(pred, target)}
