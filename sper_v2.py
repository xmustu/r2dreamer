"""
SPER v2: Spatial Perception Enhancement for RSSM — ORIGINAL DESIGN (rev2, post-Codex-review)

与当前生产版 sper.py（M2 验证用 MSE + embedding 派生目标）不同，本模块实现
FINAL_PROPOSAL.md 的原设计：

- MotionHead  → 二值运动掩码（帧间差分阈值化）→ BCEWithLogitsLoss
- DepthHead   → 鲁棒归一化深度图              → L1 Loss

Codex 评审修复记录（2026-08-15）：
- [CRITICAL→FIXED] depth_obs._find_dmc 遍历逻辑
- [MAJOR→FIXED] 深度目标 NaN/Inf 与远平面值破坏归一化 → nan_to_num + clamp + 分位数归一化
- [MAJOR→FIXED] 末帧零目标污染 BCE → 掩掉最后一帧
- [MAJOR→FIXED] detach_feat 默认值改为 False（与计划的"正则化模式"一致）
- [MAJOR→FIXED] obs dict 原地修改 → dict(obs) 拷贝
- [MINOR→FIXED] 阈值化移到池化之前（max-pool 占用率，小运动目标不丢失）
- [MINOR→FIXED] BCE 类别不平衡 → 批次估计 pos_weight（可配置，clamp 上限）
- [MINOR→FIXED] 深度输出 sigmoid 约束到 [0,1]
- [INFO→FIXED] 结构改为计划所述 2 层 MLP（hidden=64，~170K 参数/头，计划粗估 50K）
- [INFO→FIXED] 集成断言（形状/设备/dtype 检查）

独立新模块，与 sper.py 并存；当前 M2 验证不受影响。集成步骤见 SPER_V2_INTEGRATION.md。
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
    """Downsample a binary (B, T, H, W) mask to (B, T, S, S) via max pooling.

    使用 max-pool：单元格内任意像素有运动 → 单元格记为运动（小运动目标不丢失）。
    """
    B, T, H, W = mask.shape
    return F.adaptive_max_pool2d(mask.reshape(B * T, 1, H, W), (spatial_size, spatial_size)).reshape(
        B, T, spatial_size, spatial_size
    )


def compute_motion_mask_target(images, spatial_size=8, threshold=0.02):
    """Compute binary motion mask target from consecutive RGB frames.

    Args:
        images: (B, T, H, W, C) float [0,1] — 注意：_cal_grad 中的 image 已经过
            Dreamer.preprocess（uint8 / 255.0），非原始 uint8
        spatial_size: 目标空间分辨率
        threshold: 亮度差阈值（[0,1] 尺度，0.02 ≈ 5/255）。像素级差分（池化之前）阈值化

    Returns:
        target: (B, T, S, S) float, {0,1} 二值掩码。最后一帧（无后继帧）目标为 0，
            由调用方在损失中掩掉。
    """
    images = images.float()  # 必须在减法前 cast，避免 uint8 回绕
    B, T = images.shape[:2]
    # 像素级差分 → 阈值化（池化之前）→ (B, T-1, H, W)
    diff = (images[:, 1:] - images[:, :-1]).abs().mean(dim=-1)
    mask = (diff > threshold).float()
    # 最后一帧无后继 → 目标 0
    padded = torch.zeros(B, T, mask.shape[-2], mask.shape[-1], device=images.device, dtype=mask.dtype)
    padded[:, :-1] = mask
    return _downsample_occupancy(padded, spatial_size)  # (B, T, S, S)


def compute_depth_target(depth, spatial_size=8, max_depth=10.0, eps=1e-6):
    """Robust per-frame depth normalization and downsample to S×S.

    Args:
        depth: (B, T, H, W) float — MuJoCo depth render（米），envs/depth_obs.py 提供
        spatial_size: 目标空间分辨率
        max_depth: 深度上限（米）。MuJoCo 远平面/天空可产生巨大值，先 clamp

    Returns:
        target: (B, T, S, S) float，每帧 2%/98% 分位数归一化到 [0,1]
    """
    depth = depth.float()
    depth = torch.nan_to_num(depth, nan=0.0, posinf=max_depth, neginf=0.0)
    depth = depth.clamp(0.0, max_depth)
    B, T = depth.shape[:2]
    # 分位数归一化：远平面少量极大值不会压扁物体几何
    flat = depth.reshape(B, T, -1)
    q_lo = torch.quantile(flat, 0.02, dim=-1, keepdim=True)  # (B, T, 1)
    q_hi = torch.quantile(flat, 0.98, dim=-1, keepdim=True)
    normalized = ((depth - q_lo.unsqueeze(-1)) / (q_hi.unsqueeze(-1) - q_lo.unsqueeze(-1) + eps)).clamp(0.0, 1.0)
    return _downsample_image(normalized, spatial_size)


class SpatialHead(nn.Module):
    """Flat latent → S×S spatial map。2 层 MLP（计划结构）：Linear→LN→SiLU→Linear。

    motion 头输出 logits（无激活）；depth 头输出经 sigmoid（目标 [0,1]）。
    """

    def __init__(self, feat_size, hidden_size, spatial_size, output_activation=None):
        super().__init__()
        self.spatial_size = spatial_size
        self.output_activation = output_activation
        self.net = nn.Sequential(
            nn.Linear(feat_size, hidden_size, bias=True),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, spatial_size * spatial_size, bias=True),
        )

    def forward(self, feat):
        B, T, F = feat.shape
        out = self.net(feat.reshape(B * T, F))
        if self.output_activation is not None:
            out = self.output_activation(out)
        return out.reshape(B, T, self.spatial_size, self.spatial_size)


class SPERv2(nn.Module):
    """原设计：运动掩码 (BCE) + 深度 (L1) 空间感知预测。

    挂载点与 sper.SPER 相同：feat = RSSM get_feat(post_stoch, post_deter)。
    detach_feat=False（默认，正则化模式）时梯度流入 RSSM 隐状态。
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
    ):
        super().__init__()
        self.detach_feat = detach_feat
        self.spatial_size = spatial_size
        self.motion_threshold = motion_threshold
        self.max_depth = max_depth
        self.motion_pos_weight = motion_pos_weight
        self._warned_no_depth = False

        self.motion_head = SpatialHead(feat_size, hidden_size, spatial_size)  # logits
        self.depth_head = SpatialHead(feat_size, hidden_size, spatial_size, output_activation=torch.sigmoid)

    def forward(self, feat):
        if self.detach_feat:
            feat = feat.detach()
        return {"motion_logits": self.motion_head(feat), "depth_pred": self.depth_head(feat)}

    def compute_losses(self, feat, data, lambda_motion=0.01, lambda_depth=0.01):
        """原设计损失：motion BCE（掩末帧）+ depth L1。

        Args:
            feat: (B, T, F) RSSM latent
            data: 批次数据（TensorDict），需含 "image"；含 "depth" 时启用深度损失
            lambda_motion / lambda_depth: 损失权重

        Returns:
            dict: {"sper_motion": BCE loss, "sper_depth": L1 loss（若 depth 可用）}
        """
        # --- 集成断言（Codex 建议 #17）：批量级，开销可忽略 ---
        B, T, _F = feat.shape
        img = data["image"]
        assert img.ndim == 5 and img.shape[-1] in (1, 3), f"image shape {img.shape}"
        assert img.shape[:2] == (B, T), f"image (B,T) {img.shape[:2]} vs feat {(B, T)}"
        assert img.device == feat.device, "image/feat device mismatch"
        assert T > 1, "BCE motion target needs T > 1 (batch_length)"

        losses = {}
        preds = self.forward(feat)

        # --- Motion: BCE with binary mask target（末帧掩掉）---
        with torch.no_grad():
            motion_target = compute_motion_mask_target(
                img, self.spatial_size, self.motion_threshold
            )  # (B, T, S, S)，最后一帧全 0
        # 类别不平衡：批次估计 pos_weight（clamp 上限防爆）
        if self.motion_pos_weight is None:
            pos = motion_target[:, :-1].sum()
            neg = motion_target[:, :-1].numel() - pos
            pos_weight = (neg / (pos + 1e-6)).clamp(min=1.0, max=20.0)
        else:
            pos_weight = self.motion_pos_weight
        losses["sper_motion"] = lambda_motion * F.binary_cross_entropy_with_logits(
            preds["motion_logits"][:, :-1], motion_target[:, :-1], pos_weight=pos_weight
        )

        # --- Depth: L1 vs robust-normalized depth（需 data["depth"]）---
        if lambda_depth > 0 and "depth" in data:
            assert data["depth"].shape[:2] == (B, T) and data["depth"].device == feat.device
            with torch.no_grad():
                depth_target = compute_depth_target(data["depth"], self.spatial_size, self.max_depth)
            losses["sper_depth"] = lambda_depth * F.l1_loss(
                preds["depth_pred"], depth_target  # 深度逐帧独立，末帧有效，不掩
            )
        elif lambda_depth > 0 and not self._warned_no_depth:
            # 集成阶段：depth 观测尚未接入时跳过深度损失并提示一次
            self._warned_no_depth = True
            print("[SPERv2] data has no 'depth' key — depth loss skipped. "
                  "Integrate envs/depth_obs.py to enable it.")

        return losses
