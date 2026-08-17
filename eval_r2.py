"""Post-hoc R² evaluation: linear probe from projector latents to ground-truth factors.

Uses train/test split for honest R², probes the correct latent (projector output,
not raw encoder CLS). Computes participation ratio on projector latents.
"""

import json
from pathlib import Path

import hydra
import numpy as np
import torch
import stable_worldmodel as swm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from omegaconf import OmegaConf
from torchvision.transforms import v2 as transforms
import stable_pretraining as spt


def get_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=img_size),
    ])


@torch.no_grad()
def compute_projector_latents(model, dataset, max_samples=5000, device="cuda"):
    """Compute PROJECTOR latents (LeWM training latents) + ground-truth states.

    Uses the full encode path: encoder CLS → projector → latent.
    """
    all_latents = []
    all_states = []
    transform_fn = get_transform(224)
    model = model.to(device).eval()

    indices = np.arange(len(dataset))
    if len(indices) > max_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(indices, size=max_samples, replace=False)
    indices = sorted(indices)

    for idx in indices:
        row = dataset.get_row_data([idx])
        pixels = row.get('pixels')
        state = row.get('state')
        if pixels is None or state is None:
            continue

        # pixels: (T, H, W, C) or (H, W, C) — take first frame
        if pixels.ndim >= 3:
            img = pixels[0] if pixels.shape[0] > 0 else pixels
        else:
            continue

        img = torch.from_numpy(img).float()
        if img.max() > 1.0:
            img = img / 255.0
        if img.ndim == 2:
            img = img.unsqueeze(-1)
        if img.shape[-1] not in (1, 3):
            img = img.permute(2, 0, 1)
        else:
            img = img.permute(2, 0, 1)

        img = transform_fn(img).unsqueeze(0).to(device)

        # Full encode: encoder → projector
        enc_out = model.encoder(img, interpolate_pos_encoding=True)
        cls_emb = enc_out.last_hidden_state[:, 0]
        latent = model.projector(cls_emb)  # ← correct latent

        all_latents.append(latent.cpu().numpy().squeeze(0))

        gt = state[0] if state.ndim >= 2 else state
        all_states.append(gt)

    return np.stack(all_latents), np.stack(all_states)


def compute_r2(latents, states, test_frac=0.3):
    """Honest R²: train/test split, linear probe, R² on held-out test set."""
    N = len(latents)
    n_test = int(N * test_frac)
    if n_test < 10:
        n_test = min(N // 3, N - 5)

    rng = np.random.default_rng(42)
    perm = rng.permutation(N)
    train_idx = perm[n_test:]
    test_idx = perm[:n_test]

    X_train, X_test = latents[train_idx], latents[test_idx]
    y_train, y_test = states[train_idx], states[test_idx]

    sx = StandardScaler().fit(X_train)
    X_tr = sx.transform(X_train)
    X_te = sx.transform(X_test)

    K = states.shape[1]
    r2_scores = np.zeros(K)
    for k in range(K):
        sy = StandardScaler()
        y_tr = sy.fit_transform(y_train[:, k:k+1]).ravel()
        y_te = sy.transform(y_test[:, k:k+1]).ravel()

        reg = LinearRegression().fit(X_tr, y_tr)
        y_pred = reg.predict(X_te)

        ss_res = ((y_te - y_pred) ** 2).sum()
        ss_tot = ((y_te - y_te.mean()) ** 2).sum()
        r2_scores[k] = 1 - ss_res / max(ss_tot, 1e-12)

    return r2_scores


def compute_pr(latents):
    """Participation ratio from Ledoit-Wolf covariance."""
    cov = LedoitWolf().fit(latents).covariance_
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    eigvals = np.maximum(eigvals, 1e-6)
    pr = (eigvals.sum() ** 2) / (eigvals ** 2).sum()
    return pr, eigvals


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg):
    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path is None:
        print("ERROR: --ckpt_path required")
        return

    print(f"Loading model from {ckpt_path}")
    model = swm.wm.utils.load_pretrained(ckpt_path)
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        cfg.eval.dataset_name,
        keys_to_cache=["pixels", "state", "action"],
        cache_dir=dataset_path,
    )

    print("Computing projector latents...")
    latents, states = compute_projector_latents(model, dataset, max_samples=5000)
    print(f"Latents: {latents.shape}, States: {states.shape}")

    # R² (honest train/test split)
    r2 = compute_r2(latents, states)
    print(f"\nR² per factor (test set): {np.round(r2, 4)}")
    print(f"Mean R²: {r2.mean():.4f}")

    # Participation ratio
    pr, eigvals = compute_pr(latents)
    print(f"\nParticipation ratio: {pr:.2f} / {latents.shape[1]}")
    print(f"PR/dim: {pr / latents.shape[1]:.3f}")
    print(f"Top-10 eigenvalues: {np.round(eigvals[:10], 4)}")
    print(f"Bottom-10: {np.round(eigvals[-10:], 4)}")

    results = {
        "ckpt_path": str(ckpt_path),
        "r2_per_factor": r2.tolist(),
        "mean_r2": float(r2.mean()),
        "participation_ratio": float(pr),
        "pr_over_dim": float(pr / latents.shape[1]),
        "top10_eigvals": eigvals[:10].tolist(),
        "bottom10_eigvals": eigvals[-10:].tolist(),
        "n_samples": len(latents),
        "n_test": int(len(latents) * 0.3),
    }

    out_path = Path(ckpt_path).parent / "r2_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    run()
