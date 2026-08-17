"""Training script for ERAR (Effective-Rank-Aware Replay) on LeWorldModel.

Key fixes from code review:
  - Part A: WeightedRandomSampler rebuilt each epoch from ERAR weights
  - Part B: Only augments CONTEXT (target stays clean), training-only
  - Coverage: Uses projector latents (not raw encoder CLS)
  - Ledoit-Wolf: sklearn implementation (numerically stable)
  - Validation: ERAR augmentation disabled
"""

import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback
from erar import ERARManager


# ---------------------------------------------------------------------------
# ERAR Callback — plugs into Lightning training loop
# ---------------------------------------------------------------------------

class ERARCallback(Callback):
    """Recomputes coverage metrics each epoch and rebuilds DataLoader sampler."""

    def __init__(self, erar_manager: ERARManager, cfg):
        super().__init__()
        self.erar = erar_manager
        self.cfg = cfg
        self.dataset = None  # set after dataset creation

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Cache pixel observations for coverage computation."""
        if not self.erar.should_update(trainer.current_epoch):
            return
        pixels = batch.get('pixels')
        if pixels is not None:
            # Cache every 10th batch to limit memory
            if batch_idx % 10 == 0:
                self.erar.cache_obs(pixels[:16])  # keep first 16 frames

    def on_train_epoch_end(self, trainer, pl_module):
        """Rebuild DataLoader with updated ERAR weights."""
        if not self.erar.should_update(trainer.current_epoch):
            return

        cached = self.erar.monitor.get_cached_obs()
        if cached is None or len(cached) < 100:
            return

        # Update coverage
        encoder = pl_module.model.encoder
        projector = pl_module.model.projector
        pr = self.erar.update(encoder, projector, cached, trainer.current_epoch)

        pl_module.log("erar/participation_ratio", pr,
                       on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("erar/pr_over_dim", pr / self.cfg.embed_dim,
                       on_step=False, on_epoch=True, sync_dist=True)

        # Rebuild train DataLoader with coverage-weighted sampler
        weights = self.erar.get_weights()
        if weights is not None and self.dataset is not None:
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=torch.from_numpy(weights).float(),
                num_samples=len(weights),
                replacement=True,
            )
            train_loader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.cfg.loader.batch_size,
                sampler=sampler,
                num_workers=self.cfg.loader.get("num_workers", 6),
                persistent_workers=self.cfg.loader.get("persistent_workers", True),
                prefetch_factor=self.cfg.loader.get("prefetch_factor", 3),
                pin_memory=self.cfg.loader.get("pin_memory", True),
            )
            # Update the data module's train loader
            trainer.train_dataloader = train_loader

        print(f"[ERAR] Epoch {trainer.current_epoch}: PR={pr:.2f} "
              f"(PR/d={pr/self.cfg.embed_dim:.3f}), "
              f"coverage_ok={self.erar.monitor.is_coverage_adequate()}")


# ---------------------------------------------------------------------------
# Forward pass with ERAR (training-only augmentation)
# ---------------------------------------------------------------------------

def lejepa_forward_erar(self, batch, stage, cfg):
    """Forward pass: encode, optionally augment context (train only), predict, loss."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)

    emb = output["emb"]            # (B, T, D) — projector output
    act_emb = output["act_emb"]

    # --- ERAR Part B: Augment context only (training only, targets clean) ---
    if (hasattr(self, 'erar_manager') and self.erar_manager is not None
            and stage == "train"):
        under_dirs = self.erar_manager.get_undercovered_directions()
        if under_dirs is not None and under_dirs.shape[1] > 0:
            ctx_emb = emb[:, :ctx_len]           # (B, ctx_len, D)
            ctx_aug = self.erar_manager.augment_context(ctx_emb, under_dirs)
            emb = torch.cat([ctx_aug, emb[:, ctx_len:]], dim=1)  # target untouched
            output["emb"] = emb

            aug_rate = (ctx_aug != emb[:, :ctx_len]).any(dim=-1).float().mean()
            self.log(f"{stage}/erar_aug_rate", aug_rate, on_step=True, sync_dist=True)
    # --- End ERAR ---

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]       # CLEAN target — never augmented

    pred_emb = self.model.predict(ctx_emb, ctx_act)

    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    from hdf5_loader import load_hdf5_dataset

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    is_hdf5 = False

    # Try stable_worldmodel loader first; fall back to HDF5 loader
    try:
        dataset = swm.data.load_dataset(
            dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
        )
    except (FileNotFoundError, ValueError) as e:
        h5_path = os.path.join(
            cache_dir or os.path.expanduser("~/.stable_worldmodel/datasets"),
            dataset_name + ".h5"
        )
        if not os.path.exists(h5_path):
            h5_path = os.path.expanduser(f"~/.stable-wm/{dataset_name}.h5")
        if os.path.exists(h5_path):
            print(f"[INFO] Loading HDF5 dataset directly from {h5_path}")
            dataset = load_hdf5_dataset(
                h5_path,
                num_steps=dataset_cfg.get("num_steps", 4),
                frameskip=dataset_cfg.get("frameskip", 5),
            )
            is_hdf5 = True
        else:
            raise RuntimeError(
                f"Could not load dataset {dataset_name!r}. "
                f"stable_worldmodel failed ({e}), HDF5 not found at {h5_path}"
            )

    if not hasattr(dataset, 'column_names'):
        dataset.column_names = list(dataset._file.keys()) if hasattr(dataset, '_file') else []
    # Image transform: skip for HDF5 (images already 224x224 normalized in loader)
    if is_hdf5:
        transforms = []  # HDF5 loader already provides normalized 224x224 images
    else:
        transforms = [get_img_preprocessor(
            source='pixels', target='pixels', img_size=cfg.img_size
        )]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
        # HDF5: raw per-step action dim; original: frameskip * per-step dim
        raw_action_dim = dataset.get_dim("action")
        if is_hdf5:
            cfg.model.action_encoder.input_dim = raw_action_dim
        else:
            cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * raw_action_dim

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False
    )

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    # --- ERAR: Initialize manager ---
    erar_enabled = cfg.get("erar", {}).get("enabled", False)
    erar_manager = None

    if erar_enabled:
        erar_cfg = cfg.erar
        erar_manager = ERARManager(
            encoder=world_model.encoder,
            projector=world_model.projector,
            embed_dim=cfg.embed_dim,
            ema_decay=erar_cfg.get("ema_decay", 0.99),
            update_every=erar_cfg.get("update_every", 1000),
            tau=erar_cfg.get("tau", 0.5),
            alpha=erar_cfg.get("alpha", 0.1),
            noise_scale=erar_cfg.get("noise_scale", 0.1),
            aug_prob=erar_cfg.get("aug_prob", 0.05),
            device="cuda",
        )
        print(f"[ERAR] Enabled: alpha={erar_cfg.get('alpha', 0.1)}, "
              f"aug_prob={erar_cfg.get('aug_prob', 0.05)}, "
              f"tau={erar_cfg.get('tau', 0.5)}")

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    forward_fn = partial(lejepa_forward_erar, cfg=cfg)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=forward_fn,
        optim=optimizers,
    )
    world_model.erar_manager = erar_manager

    ##########################
    ##       callbacks      ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
    )
    callbacks = [object_dump_callback]

    if erar_manager is not None:
        erar_cb = ERARCallback(erar_manager, cfg)
        erar_cb.dataset = train_set
        callbacks.append(erar_cb)

    ##########################
    ##       training       ##
    ##########################

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()

    # Save ERAR metrics
    if erar_manager is not None:
        pr_hist = erar_manager.monitor.pr_history
        np.save(run_dir / "erar_pr_history.npy", np.array(pr_hist))


if __name__ == "__main__":
    run()
