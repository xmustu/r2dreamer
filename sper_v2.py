"""
SPER v2: Spatial Perception Enhancement for RSSM — ORIGINAL DESIGN (rev3)

三模态空间感知 + 结构化门控融合（SGFM）：
- MotionHead  → 二值运动掩码（帧间差分阈值化）→ BCEWithLogitsLoss
- DepthHead   → 鲁棒归一化深度图              → L1 Loss
- ContactHead → 二值接触掩码（MuJoCo data.contact 真值）→ BCEWithLogitsLoss
- SGFM        → 各头 hidden 特征独立编码，决策阶段门控融合，
                 残差注入 actor 输入（编码纯净，决策耦合）

rev3 变更（2026-08-18）：
- 新增 ContactHead（nbody=8 二值接触目标，data["contact"]）
- 新增 SGFM 类（structured gating fusion mechanism，C4 消融用）
- SpatialHead 暴露 hidden() 供 SGFM 取各模态编码特征

独立新模块，与 sper.py 并存。集成见 SPER_V2_INTEGRATION.md。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _downsample_image(x, spatial_size):
    """Downsample (B, T, H, W) to (B, T, S, S) via adaptive average pooling."""
    B, T, H, W = x.shape
    return F.adaptive_avg_pool2d(x.reshape(B * T, 1, H, W), (spatial_size, spatial_size)).reshape(
        B, T, spatial_size, spatial_size
    )


def _downsample_occupancy(mask, spatial_size):
    """Downsample a binary (B, T, H, W) mask to (B, T, S, S) via max pooling."""
    B, T, H, W = mask.shape
    return F.adaptive_max_pool2d(mask.reshape(B * T, 1, H, W), (spatial_size, spatial_size)).reshape(
        B, T, spatial_size, spatial_size
    )


def compute_motion_mask_target(images, spatial_size=8, threshold=0.02):
    """Binary motion mask from consecutive RGB frames (preprocessed [0,1])."""
    images = images.float()
    B, T = images.shape[:2]
    diff = (images[:, 1:] - images[:, :-1]).abs().mean(dim=-1)
    mask = (diff > threshold).float()
    padded = torch.zeros(B, T, mask.shape[-2], mask.shape[-1], device=images.device, dtype=mask.dtype)
    padded[:, :-1] = mask
    return _downsample_occupancy(padded, spatial_size)


def compute_depth_target(depth, spatial_size=8, max_depth=10.0, eps=1e-6):
    """Robust per-frame depth normalization → S×S [0,1]."""
    depth = depth.float()
    depth = torch.nan_to_num(depth, nan=0.0, posinf=max_depth, neginf=0.0)
    depth = depth.clamp(0.0, max_depth)
    B, T = depth.shape[:2]
    flat = depth.reshape(B, T, -1)
    q_lo = torch.quantile(flat, 0.02, dim=-1, keepdim=True)
    q_hi = torch.quantile(flat, 0.98, dim=-1, keepdim=True)
    normalized = ((depth - q_lo.unsqueeze(-1)) / (q_hi.unsqueeze(-1) - q_lo.unsqueeze(-1) + eps)).clamp(0.0, 1.0)
    return _downsample_image(normalized, spatial_size)


def compute_contact_target(contact):
    """Contact target: (B, T, nbody) binary — 由 env wrapper 提供（MuJoCo data.contact 真值）."""
    return contact.float()


class SpatialHead(nn.Module):
    """Flat latent → S×S spatial map。2 层 MLP + LN，暴露 hidden 供 SGFM 融合。"""

    def __init__(self, feat_size, hidden_size, spatial_size, output_activation=None):
        super().__init__()
        self.spatial_size = spatial_size
        self.output_activation = output_activation
        self.first = nn.Sequential(
            nn.Linear(feat_size, hidden_size, bias=True),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.last = nn.Linear(hidden_size, spatial_size * spatial_size, bias=True)

    def _reshape_in(self, feat):
        was2d = feat.dim() == 2
        return (feat.unsqueeze(1) if was2d else feat), was2d

    def hidden(self, feat):
        """模态编码特征（决策阶段融合用）：(B, T, hidden) 或 (B, hidden)"""
        f, was2d = self._reshape_in(feat)
        B, T, F = f.shape
        h = self.first(f.reshape(B * T, F)).reshape(B, T, -1)
        return h[:, 0] if was2d else h

    def forward(self, feat):
        f, was2d = self._reshape_in(feat)
        B, T, F = f.shape
        h = self.first(f.reshape(B * T, F))
        out = self.last(h)
        if self.output_activation is not None:
            out = self.output_activation(out)
        out = out.reshape(B, T, self.spatial_size, self.spatial_size)
        return out[:, 0] if was2d else out


class FlatHead(nn.Module):
    """Flat latent → nbody 接触 logits。2 层 MLP + LN，暴露 hidden。"""

    def __init__(self, feat_size, hidden_size, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.first = nn.Sequential(
            nn.Linear(feat_size, hidden_size, bias=True),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.last = nn.Linear(hidden_size, out_dim, bias=True)

    def hidden(self, feat):
        was2d = feat.dim() == 2
        f = feat.unsqueeze(1) if was2d else feat
        B, T, F = f.shape
        h = self.first(f.reshape(B * T, F)).reshape(B, T, -1)
        return h[:, 0] if was2d else h

    def forward(self, feat):
        was2d = feat.dim() == 2
        f = feat.unsqueeze(1) if was2d else feat
        B, T, F = f.shape
        out = self.last(self.first(f.reshape(B * T, F))).reshape(B, T, self.out_dim)
        return out[:, 0] if was2d else out


class SGFM(nn.Module):
    """Structured Gating Fusion Mechanism — 结构化门控融合（C4 消融）。

    各模态 head 的 hidden 特征在编码阶段保持独立（各自经独立 MLP 提取）；
    SGFM 在决策阶段学习门控权重 α ∈ R³，加权融合后经输出投影生成与
    feat 同维的残差，注入 actor 输入：actor_input = feat + sgfm(feat)。
    """

    def __init__(self, feat_size, hidden_size, n_modalities=3, fusion_hidden=128):
        super().__init__()
        self.n_modalities = n_modalities
        # 门控网络：从各模态激活摘要（(B,T,n_mod)）学模态权重
        self.gate = nn.Sequential(
            nn.Linear(n_modalities, fusion_hidden, bias=True),
            nn.SiLU(),
            nn.Linear(fusion_hidden, n_modalities, bias=True),
        )
        # 融合特征投影回 feat_size（残差注入）
        self.out = nn.Linear(hidden_size, feat_size, bias=True)

    def forward(self, modal_hiddens):
        """modal_hiddens: list of (B, T, hidden) 或 (B, hidden) 各模态编码特征"""
        was2d = modal_hiddens[0].dim() == 2
        if was2d:
            modal_hiddens = [m.unsqueeze(1) for m in modal_hiddens]
        B, T, H = modal_hiddens[0].shape
        stacked = torch.stack(modal_hiddens, dim=-1)  # (B, T, hidden, n_mod)
        # 门控输入：各模态特征的均值池化（维度无关的摘要）
        feat_summary = stacked.mean(dim=2)  # (B, T, n_mod)
        logits = self.gate(feat_summary.detach())  # (B, T, n_mod)
        alpha = torch.softmax(logits, dim=-1)  # (B, T, n_mod)
        fused = (stacked * alpha.unsqueeze(2)).sum(dim=-1)  # (B, T, hidden)
        out = self.out(fused)  # (B, T, feat_size)
        return out[:, 0] if was2d else out


class SPERv2(nn.Module):
    """原设计：运动掩码 (BCE) + 深度 (L1) + 接触掩码 (BCE) + SGFM 融合。

    detach_feat=False（默认）时梯度流入 RSSM 隐状态（正则化模式）。
    目标（targets）始终 detached。
    """

    def __init__(
        self,
        feat_size,
        hidden_size=64,
        detach_feat=False,
        spatial_size=8,
        motion_threshold=0.02,
        max_depth=10.0,
        motion_pos_weight=None,
        contact_dim=8,
        contact_enabled=True,
        sgfm_enabled=True,
    ):
        super().__init__()
        self.detach_feat = detach_feat
        self.spatial_size = spatial_size
        self.motion_threshold = motion_threshold
        self.max_depth = max_depth
        self.motion_pos_weight = motion_pos_weight
        self.contact_dim = contact_dim
        self.contact_enabled = contact_enabled
        self.sgfm_enabled = sgfm_enabled
        self._warned_no_depth = False
        self._warned_no_contact = False

        self.motion_head = SpatialHead(feat_size, hidden_size, spatial_size)
        self.depth_head = SpatialHead(feat_size, hidden_size, spatial_size, output_activation=torch.sigmoid)
        # noContact 消融：contact_enabled=False 时不构建接触头，SGFM 退化为 2 模态
        self.contact_head = FlatHead(feat_size, hidden_size, contact_dim) if contact_enabled else None

        # SGFM：决策阶段融合（C4 消融）。模态数 = 运动/深度(/接触)
        n_modalities = 3 if contact_enabled else 2
        self.sgfm = SGFM(feat_size, hidden_size, n_modalities=n_modalities) if sgfm_enabled else None

    def forward(self, feat):
        if self.detach_feat:
            feat = feat.detach()
        preds = {
            "motion_logits": self.motion_head(feat),
            "depth_pred": self.depth_head(feat),
        }
        if self.contact_head is not None:
            preds["contact_logits"] = self.contact_head(feat)
        return preds

    def fused_residual(self, feat):
        """SGFM 残差：actor_input = feat + fused_residual(feat)。推断/训练共用。"""
        if self.sgfm is None:
            return None
        hiddens = [
            self.motion_head.hidden(feat),
            self.depth_head.hidden(feat),
        ]
        if self.contact_head is not None:
            hiddens.append(self.contact_head.hidden(feat))
        return self.sgfm(hiddens)

    def compute_losses(self, feat, data):
        """损失：motion BCE + depth L1 + contact BCE（**原始损失，未缩放**）。

        权重统一由 Dreamer 的 `loss_scales`（sper_motion/sper_depth/sper_contact
        各 0.01）在 total_loss 处一次性缩放——2026-08-26 修复双重缩放问题
        （此前模块内乘 λ=0.01 又被 loss_scales 乘 0.01，motion/depth 有效权重
        仅 1e-4，与 contact 的 0.01 不一致）。

        Args:
            feat: (B, T, F) RSSM latent
            data: 批次数据，需含 "image"；可选 "depth" / "contact"
        """
        B, T, _F = feat.shape
        img = data["image"]
        assert img.ndim == 5 and img.shape[-1] in (1, 3), f"image shape {img.shape}"
        assert img.shape[:2] == (B, T), f"image (B,T) {img.shape[:2]} vs feat {(B, T)}"
        assert img.device == feat.device, "image/feat device mismatch"
        assert T > 1, "BCE motion target needs T > 1 (batch_length)"

        losses = {}
        preds = self.forward(feat)

        # --- Motion: BCE（掩末帧）---
        with torch.no_grad():
            motion_target = compute_motion_mask_target(
                img, self.spatial_size, self.motion_threshold
            )
        if self.motion_pos_weight is None:
            pos = motion_target[:, :-1].sum()
            neg = motion_target[:, :-1].numel() - pos
            pos_weight = (neg / (pos + 1e-6)).clamp(min=1.0, max=20.0)
        else:
            pos_weight = self.motion_pos_weight
        losses["sper_motion"] = F.binary_cross_entropy_with_logits(
            preds["motion_logits"][:, :-1], motion_target[:, :-1], pos_weight=pos_weight
        )

        # --- Depth: L1 ---
        if "depth" in data:
            assert data["depth"].shape[:2] == (B, T) and data["depth"].device == feat.device
            with torch.no_grad():
                depth_target = compute_depth_target(data["depth"], self.spatial_size, self.max_depth)
            losses["sper_depth"] = F.l1_loss(
                preds["depth_pred"], depth_target  # 深度逐帧独立，末帧有效
            )
        elif not self._warned_no_depth:
            self._warned_no_depth = True
            print("[SPERv2] data has no 'depth' key — depth loss skipped.")

        # --- Contact: BCE（稀疏，pos_weight）---
        if self.contact_enabled and "contact" in data:
            assert data["contact"].shape[:2] == (B, T) and data["contact"].device == feat.device
            with torch.no_grad():
                contact_target = compute_contact_target(data["contact"])
            pos = contact_target.sum()
            neg = contact_target.numel() - pos
            cw = (neg / (pos + 1e-6)).clamp(min=1.0, max=20.0)
            losses["sper_contact"] = F.binary_cross_entropy_with_logits(
                preds["contact_logits"], contact_target, pos_weight=cw
            )
        elif self.contact_enabled and not self._warned_no_contact:
            self._warned_no_contact = True
            print("[SPERv2] data has no 'contact' key — contact loss skipped.")

        return losses
