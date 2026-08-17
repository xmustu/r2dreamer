"""
SPER v2 单元验证 rev2（CPU only — 不影响服务器上运行的 M2 实验）。

覆盖 Codex 评审修复项：
1. BCE 运动掩码：二值、检测运动、静止全零；末帧在损失中被掩掉
2. 深度：分位数归一化 + 对 inf/远平面值鲁棒
3. 深度输出 sigmoid ∈ [0,1]
4. detach_feat=False（默认）→ 梯度流入 feat；True → 不流入
5. motion-only 时 feat 梯度在 t=T-1 为零（末帧掩码生效）
6. 无 depth 数据时优雅跳过

运行：python test_sper_v2.py（CPU，无 CUDA 依赖）
"""

import sys
import torch

sys.path.insert(0, ".")
import sper_v2


def main():
    torch.manual_seed(0)
    B, T, H, W, C = 4, 8, 64, 64, 3
    feat_size, hidden = 2560, 64

    # 合成数据：[0,1] 浮点（_cal_grad 中 image 已由 preprocess /255）。
    # 所有帧 = 帧0（完全相同）；batch0 帧3 加亮斑 → 该处运动=1
    images = torch.randint(0, 256, (B, T, H, W, C), dtype=torch.uint8).float() / 255.0
    images[:, 1:] = images[:, :1]
    images[0, 3, 10:40, 20:50] = 200 / 255.0

    depth = torch.rand(B, T, H, W) * 5.0 + 1.0  # 米，模拟 MuJoCo depth

    # --- 目标函数检查 ---
    mask = sper_v2.compute_motion_mask_target(images, spatial_size=8, threshold=0.02)
    assert mask.shape == (B, T, 8, 8), f"mask shape {mask.shape}"
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}, "mask must be binary"
    assert mask[0, 3].max() > 0.5, "moving patch should light up the mask"
    assert mask[1].sum() == 0, "identical frames should give all-zero mask"
    assert mask[0, -1].sum() == 0, "last frame must be zero (no successor)"
    print("[OK] motion mask target: binary, detects motion, static=0, last frame=0")

    dt = sper_v2.compute_depth_target(depth, spatial_size=8)
    assert dt.shape == (B, T, 8, 8), f"depth target shape {dt.shape}"
    assert (dt.min() >= 0.0) and (dt.max() <= 1.0), "depth must be normalized to [0,1]"
    # 鲁棒性：inf / 巨大远平面值不应产生 NaN 或破坏归一化
    depth_bad = depth.clone()
    depth_bad[0, 0, :, :] = float("inf")
    depth_bad[1, 1, 10, 10] = 1e6
    dt_bad = sper_v2.compute_depth_target(depth_bad, spatial_size=8)
    assert torch.isfinite(dt_bad).all(), "inf/huge depth must not produce NaN"
    print("[OK] depth target: normalized, robust to inf/huge values")

    # --- detach 模式梯度通路（默认 False）---
    model_default = sper_v2.SPERv2(feat_size, hidden, spatial_size=8)
    assert model_default.detach_feat is False, "default detach_feat must be False (regularization)"
    print("[OK] detach_feat default = False (matches plan)")

    for detach, expect_grad in [(True, False), (False, True)]:
        model = sper_v2.SPERv2(feat_size, hidden, detach_feat=detach, spatial_size=8)
        feat = torch.randn(B, T, feat_size, requires_grad=True)
        data = {"image": images, "depth": depth}
        losses = model.compute_losses(feat, data, lambda_motion=1.0, lambda_depth=1.0)
        total = sum(losses.values())
        assert torch.isfinite(total), f"loss NaN with detach={detach}"
        total.backward()
        grad_flows = feat.grad is not None and feat.grad.abs().sum() > 0
        assert grad_flows == expect_grad, (
            f"detach={detach}: expected grad_flow={expect_grad}, got {grad_flows}"
        )
        print(f"[OK] detach_feat={detach}: grad into feat = {grad_flows} "
              f"(motion={losses['sper_motion'].item():.4f}, depth={losses['sper_depth'].item():.4f})")

    # --- 末帧掩码：motion-only 时 feat 梯度在 t=T-1 应为零 ---
    model = sper_v2.SPERv2(feat_size, hidden, detach_feat=False, spatial_size=8)
    feat = torch.randn(B, T, feat_size, requires_grad=True)
    losses = model.compute_losses(feat, {"image": images}, lambda_motion=1.0, lambda_depth=0.0)
    losses["sper_motion"].backward()
    last_frame_grad = feat.grad[:, -1].abs().sum().item()
    assert last_frame_grad == 0.0, f"last-frame grad must be 0 (masked), got {last_frame_grad}"
    print("[OK] BCE masks out last timestep (no gradient at t=T-1)")

    # --- 深度输出 sigmoid ∈ [0,1] ---
    model = sper_v2.SPERv2(feat_size, hidden, detach_feat=True, spatial_size=8)
    preds = model.forward(torch.randn(B, T, feat_size))
    assert (preds["depth_pred"].min() >= 0.0) and (preds["depth_pred"].max() <= 1.0)
    print("[OK] depth head output is sigmoid-bounded to [0,1]")

    # --- 无 depth 数据时跳过深度损失 ---
    model = sper_v2.SPERv2(feat_size, hidden, detach_feat=True)
    losses = model.compute_losses(torch.randn(B, T, feat_size), {"image": images},
                                  lambda_motion=1.0, lambda_depth=1.0)
    assert "sper_depth" not in losses and "sper_motion" in losses
    print("[OK] missing depth: depth loss skipped, motion loss present")

    # --- 集成断言：坏输入应报错 ---
    model = sper_v2.SPERv2(feat_size, hidden, detach_feat=True)
    try:
        model.compute_losses(torch.randn(B, T, feat_size), {"image": images[:, :, :, :, 0]},
                             lambda_motion=1.0, lambda_depth=0.0)
        raise AssertionError("expected assertion failure for 4-D image")
    except AssertionError as e:
        assert "image shape" in str(e), f"unexpected assert msg: {e}"
    print("[OK] integration asserts catch malformed inputs")

    print("\nAll SPER v2 checks passed (rev2).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
