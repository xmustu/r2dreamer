import copy
import math
import pathlib
from collections import OrderedDict

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR

import networks
import rssm
import sf_rssm
import ensemble_rssm
import tt_rssm
import tt_rssm_v2
import tools
from networks import Projector
import sper
import sper_v2
from optim import LaProp, clip_grad_agc_
from tools import to_f32


class Dreamer(nn.Module):
    def __init__(self, config, obs_space, act_space):
        super().__init__()
        self.device = torch.device(config.device)
        self.act_entropy = float(config.act_entropy)
        self.kl_free = float(config.kl_free)
        self.imag_horizon = int(config.imag_horizon)
        self.horizon = int(config.horizon)
        self.lamb = float(config.lamb)
        self.return_ema = networks.ReturnEMA(device=self.device)
        self.act_dim = act_space.n if hasattr(act_space, "n") else sum(act_space.shape)
        self.rep_loss = str(config.rep_loss)

        # World model components
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        # Strip dynamics_mode from model inputs — it's domain metadata for
        # replay scheduling, not an observation the world model should learn.
        shapes.pop("dynamics_mode", None)
        self.encoder = networks.MultiEncoder(config.encoder, shapes)
        self.embed_size = self.encoder.out_dim
        self.sf_enabled = bool(getattr(config.rssm, 'sf_enabled', False))
        self.ensemble_enabled = bool(getattr(config.rssm, 'ensemble_enabled', False))
        self.tt_enabled = bool(getattr(config.rssm, 'tt_enabled', False))
        self.tt_v2_enabled = bool(getattr(config.rssm, 'tt_v2_enabled', False))
        if self.sf_enabled:
            self.rssm = sf_rssm.SparseFactorizedRSSM(
                config.rssm,
                self.embed_size,
                self.act_dim,
            )
        elif self.tt_v2_enabled:
            self.rssm = tt_rssm_v2.TwoTimescaleRSSMv2(
                config.rssm,
                self.embed_size,
                self.act_dim,
            )
        elif self.tt_enabled:
            self.rssm = tt_rssm.TwoTimescaleRSSM(
                config.rssm,
                self.embed_size,
                self.act_dim,
            )
        elif self.ensemble_enabled:
            self.rssm = ensemble_rssm.EnsembleRSSM(
                config.rssm,
                self.embed_size,
                self.act_dim,
            )
        else:
            self.rssm = rssm.RSSM(
                config.rssm,
                self.embed_size,
                self.act_dim,
            )
        self.reward = networks.MLPHead(config.reward, self.rssm.feat_size)
        self.cont = networks.MLPHead(config.cont, self.rssm.feat_size)

        # SPER: Spatial Perception Enhancement for RSSM
        self.sper_enabled = bool(getattr(config, "sper_enabled", False))
        if self.sper_enabled:
            self._sper_lambda_motion = float(getattr(config, "sper_lambda_motion", 0.01))
            self._sper_lambda_depth = float(getattr(config, "sper_lambda_depth", 0.01))
            self._sper = sper.SPER(
                feat_size=self.rssm.feat_size,
                embed_size=self.embed_size,
                hidden_size=int(getattr(config, "sper_hidden", 256)),
                detach_feat=bool(getattr(config, "sper_detach_feat", False)),
            )

        # SPER v2: 原设计（真实目标 — BCE 运动掩码 + L1 深度 + BCE 接触 + SGFM）。
        # 损失权重统一由 loss_scales 缩放（2026-08-26 修复双重缩放；
        # 旧 config 键 sper_v2_lambda_* 保留但不再使用）。
        # fuse_actor=False → 推理阶段三个预测头完全移除（零推理开销）。
        self.sper_v2_enabled = bool(getattr(config, "sper_v2_enabled", False))
        if self.sper_v2_enabled:
            self._fuse_actor = bool(getattr(config, "sper_v2_fuse_actor", True))
            _contact_dim = int(shapes["contact"][0]) if "contact" in shapes else int(
                getattr(config, "sper_v2_contact_dim", 8)
            )
            self._sper_v2 = sper_v2.SPERv2(
                feat_size=self.rssm.feat_size,
                hidden_size=int(getattr(config, "sper_v2_hidden", 64)),
                detach_feat=bool(getattr(config, "sper_v2_detach_feat", False)),
                contact_dim=_contact_dim,
                contact_enabled=bool(getattr(config, "sper_v2_contact_enabled", True)),
                sgfm_enabled=bool(getattr(config, "sper_v2_sgfm_enabled", True)),
            )

        config.actor.shape = (act_space.n,) if hasattr(act_space, "n") else tuple(map(int, act_space.shape))
        self.act_discrete = False
        if hasattr(act_space, "multi_discrete"):
            config.actor.dist = config.actor.dist.multi_disc
            self.act_discrete = True
        elif hasattr(act_space, "discrete"):
            config.actor.dist = config.actor.dist.disc
            self.act_discrete = True
        else:
            config.actor.dist = config.actor.dist.cont

        # Actor-critic components
        self.actor = networks.MLPHead(config.actor, self.rssm.feat_size)
        self.value = networks.MLPHead(config.critic, self.rssm.feat_size)
        self.slow_target_update = int(config.slow_target_update)
        self.slow_target_fraction = float(config.slow_target_fraction)
        self._slow_value = copy.deepcopy(self.value)
        for param in self._slow_value.parameters():
            param.requires_grad = False
        self._slow_value_updates = 0

        self._loss_scales = dict(config.loss_scales)
        if self.sf_enabled:
            self._loss_scales.setdefault("sf_sparse", float(getattr(config.rssm, 'sf_sparse_scale', 1.0)))
            self._loss_scales.setdefault("sf_center", float(getattr(config.rssm, 'sf_center_scale', 1.0)))
        if self.ensemble_enabled:
            self.ensemble_lambda_max = float(getattr(config.rssm, 'ensemble_lambda_max', 0.1))
            self.ensemble_ramp_end = int(getattr(config.rssm, 'ensemble_ramp_end', 0.3))
        if self.tt_enabled:
            self._loss_scales.setdefault("tt_progress", 0.01)
            self._loss_scales.setdefault("tt_contact", 0.01)
            self._loss_scales.setdefault("tt_cont", 0.01)
        if self.sper_enabled:
            self._loss_scales.setdefault("sper_motion", 1.0)
            self._loss_scales.setdefault("sper_depth", 1.0)
        if self.sper_v2_enabled:
            # 注意：与 sper 共用 loss key（两者不同时启用，见集成文档）
            # 默认 0.01：与 config loss_scales 一致（三模态统一有效权重）
            self._loss_scales.setdefault("sper_motion", 0.01)
            self._loss_scales.setdefault("sper_depth", 0.01)
            self._loss_scales.setdefault("sper_contact", 0.01)
        self._log_grads = bool(config.log_grads)

        # Teacher loading for Nominal-Anchored Dynamics Randomization
        self._teacher_checkpoint = str(getattr(config, 'teacher_checkpoint', ''))
        self._teacher_actor = None
        self._teacher_value = None
        if self._teacher_checkpoint:
            self._load_teacher(config)

        modules = {
            "rssm": self.rssm,
            "actor": self.actor,
            "value": self.value,
            "reward": self.reward,
            "cont": self.cont,
            "encoder": self.encoder,
        }

        if self.rep_loss == "dreamer":
            self.decoder = networks.MultiDecoder(
                config.decoder,
                self.rssm._deter,
                self.rssm.flat_stoch,
                shapes,
            )
            recon = self._loss_scales.pop("recon")
            self._loss_scales.update({k: recon for k in self.decoder.all_keys})
            modules.update({"decoder": self.decoder})
        elif self.rep_loss == "r2dreamer" or self.rep_loss == "infonce":
            # add projector for latent to embedding
            self.prj = Projector(self.rssm.feat_size, self.embed_size)
            modules.update({"projector": self.prj})
            if self.sper_enabled:
                modules.update({"sper": self._sper})
            if self.sper_v2_enabled:
                modules.update({"sper_v2": self._sper_v2})
            self.barlow_lambd = float(config.r2dreamer.lambd)
        elif self.rep_loss == "dreamerpro":
            dpc = config.dreamer_pro
            self.warm_up = int(dpc.warm_up)
            self.num_prototypes = int(dpc.num_prototypes)
            self.proto_dim = int(dpc.proto_dim)
            self.temperature = float(dpc.temperature)
            self.sinkhorn_eps = float(dpc.sinkhorn_eps)
            self.sinkhorn_iters = int(dpc.sinkhorn_iters)
            self.ema_update_every = int(dpc.ema_update_every)
            self.ema_update_fraction = float(dpc.ema_update_fraction)
            self.freeze_prototypes_iters = int(dpc.freeze_prototypes_iters)
            self.aug_max_delta = float(dpc.aug.max_delta)
            self.aug_same_across_time = bool(dpc.aug.same_across_time)
            self.aug_bilinear = bool(dpc.aug.bilinear)

            self._prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.proto_dim))
            self.obs_proj = nn.Linear(self.embed_size, self.proto_dim)
            self.feat_proj = nn.Linear(self.rssm.feat_size, self.proto_dim)
            self._ema_encoder = copy.deepcopy(self.encoder)
            self._ema_obs_proj = copy.deepcopy(self.obs_proj)
            for param in self._ema_encoder.parameters():
                param.requires_grad = False
            for param in self._ema_obs_proj.parameters():
                param.requires_grad = False
            self._ema_updates = 0
            modules.update({
                "prototypes": self._prototypes,
                "obs_proj": self.obs_proj,
                "feat_proj": self.feat_proj,
                "ema_encoder": self._ema_encoder,
                "ema_obs_proj": self._ema_obs_proj,
            })
        # count number of parameters in each module
        for key, module in modules.items():
            if isinstance(module, nn.Parameter):
                print(f"{module.numel():>14,}: {key}")
            else:
                print(f"{sum(p.numel() for p in module.parameters()):>14,}: {key}")
        self._named_params = OrderedDict()
        for name, module in modules.items():
            if isinstance(module, nn.Parameter):
                self._named_params[name] = module
            else:
                for param_name, param in module.named_parameters():
                    self._named_params[f"{name}.{param_name}"] = param
        print(f"Optimizer has: {sum(p.numel() for p in self._named_params.values())} parameters.")

        def _agc(params):
            clip_grad_agc_(params, float(config.agc), float(config.pmin), foreach=True)

        self._agc = _agc
        self._optimizer = LaProp(
            self._named_params.values(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )
        self._scaler = GradScaler()

        def lr_lambda(step):
            if config.warmup:
                return min(1.0, (step + 1) / config.warmup)
            return 1.0

        self._scheduler = LambdaLR(self._optimizer, lr_lambda=lr_lambda)

        self.train()
        self.clone_and_freeze()
        if config.compile:
            print("Compiling update function with torch.compile...")
            self._cal_grad = torch.compile(self._cal_grad, mode="reduce-overhead")

    def _update_slow_target(self):
        """Update slow-moving value target network."""
        if self._slow_value_updates % self.slow_target_update == 0:
            with torch.no_grad():
                mix = self.slow_target_fraction
                for v, s in zip(self.value.parameters(), self._slow_value.parameters()):
                    s.data.copy_(mix * v.data + (1 - mix) * s.data)
        self._slow_value_updates += 1

    def train(self, mode=True):
        super().train(mode)
        # slow_value should be always eval mode
        self._slow_value.train(False)
        return self

    def clone_and_freeze(self):
        # NOTE: "requires_grad" affects whether a parameter is updated
        # not whether gradients flow through its operations
        self._frozen_encoder = copy.deepcopy(self.encoder)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.encoder.named_parameters(), self._frozen_encoder.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_rssm = copy.deepcopy(self.rssm)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.rssm.named_parameters(), self._frozen_rssm.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_reward = copy.deepcopy(self.reward)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.reward.named_parameters(), self._frozen_reward.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_cont = copy.deepcopy(self.cont)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.cont.named_parameters(), self._frozen_cont.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_actor = copy.deepcopy(self.actor)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.actor.named_parameters(), self._frozen_actor.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_value = copy.deepcopy(self.value)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.value.named_parameters(), self._frozen_value.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_slow_value = copy.deepcopy(self._slow_value)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self._slow_value.named_parameters(), self._frozen_slow_value.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

    def _load_teacher(self, config):
        """Load frozen teacher actor/value from a pre-trained DV3 checkpoint.

        Used by Nominal-Anchored Dynamics Randomization.
        Teacher provides distillation targets on nominal batches.
        """
        import torch
        ckpt_path = self._teacher_checkpoint
        if not ckpt_path or not any(
            p.exists() for p in [pathlib.Path(ckpt_path)]
        ):
            # Try common checkpoint names
            ckpt = pathlib.Path(ckpt_path)
            if not ckpt.exists():
                print(f"Teacher checkpoint not found: {ckpt_path}")
                return

        print(f"Loading teacher from: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        agent_sd = state_dict["agent_state_dict"]

        # Create teacher networks with same architecture
        self._teacher_actor = copy.deepcopy(self.actor)
        self._teacher_value = copy.deepcopy(self.value)

        # Load weights from checkpoint
        actor_sd = {k.replace("actor.", ""): v for k, v in agent_sd.items() if k.startswith("actor.")}
        value_sd = {k.replace("value.", ""): v for k, v in agent_sd.items() if k.startswith("value.")}
        self._teacher_actor.load_state_dict(actor_sd)
        self._teacher_value.load_state_dict(value_sd)

        # Freeze teacher
        for p in self._teacher_actor.parameters():
            p.requires_grad_(False)
        for p in self._teacher_value.parameters():
            p.requires_grad_(False)

        self._teacher_actor.eval()
        self._teacher_value.eval()
        print(f"Teacher loaded: actor={len(actor_sd)} params, value={len(value_sd)} params")

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Re-establish shared memory after moving the model to a new device
        self.clone_and_freeze()
        return self

    @torch.no_grad()
    def act(self, obs, state, eval=False):
        """Policy inference step."""
        # obs: dict of (B, *), state: (stoch: (B, S, K), deter: (B, D), prev_action: (B, A))
        torch.compiler.cudagraph_mark_step_begin()
        p_obs = self.preprocess(obs)
        # (B, E)
        embed = self._frozen_encoder(p_obs)
        prev_stoch, prev_deter, prev_action = (
            state["stoch"],
            state["deter"],
            state["prev_action"],
        )
        # (B, S, K), (B, D)
        stoch, deter, _ = self._frozen_rssm.obs_step(prev_stoch, prev_deter, prev_action, embed, obs["is_first"])
        # (B, F)
        feat = self._frozen_rssm.get_feat(stoch, deter)
        action_dist = self._frozen_actor(self._fuse_feat(feat))
        # (B, A)
        action = action_dist.mode if eval else action_dist.rsample()
        return action, TensorDict(
            {"stoch": stoch, "deter": deter, "prev_action": action},
            batch_size=state.batch_size,
        )

    @torch.no_grad()
    def get_initial_state(self, B):
        stoch, deter = self.rssm.initial(B)
        action = torch.zeros(B, self.act_dim, dtype=torch.float32, device=self.device)
        return TensorDict({"stoch": stoch, "deter": deter, "prev_action": action}, batch_size=(B,))

    @torch.no_grad()
    def video_pred(self, data, initial):
        torch.compiler.cudagraph_mark_step_begin()
        p_data = self.preprocess(data)
        return self._video_pred(p_data, initial)

    def _video_pred(self, data, initial):
        """Video prediction utility."""
        if self.rep_loss != "dreamer":
            raise NotImplementedError("video_pred requires decoder and is only supported when rep_loss == 'dreamer'.")

        B = min(data["action"].shape[0], 6)
        # (B, T, E)
        embed = self.encoder(data)

        post_stoch, post_deter, _ = self.rssm.observe(
            embed[:B, :5],
            data["action"][:B, :5],
            tuple(val[:B] for val in initial),
            data["is_first"][:B, :5],
        )
        recon = self.decoder(post_stoch, post_deter)["image"].mode()[:B]
        init_stoch, init_deter = post_stoch[:, -1], post_deter[:, -1]
        prior_stoch, prior_deter = self.rssm.imagine_with_action(
            init_stoch,
            init_deter,
            data["action"][:B, 5:],
        )
        openl = self.decoder(prior_stoch, prior_deter)["image"].mode()
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:B]
        error = (model - truth + 1.0) / 2.0
        return torch.cat([truth, model, error], 2)

    def update(self, replay_buffer):
        """Sample a batch from replay and perform one optimization step."""
        data, index, initial = replay_buffer.sample()
        torch.compiler.cudagraph_mark_step_begin()
        p_data = self.preprocess(data)
        self._update_slow_target()
        if self.rep_loss == "dreamerpro":
            self.ema_update()
        metrics = {}
        with autocast(device_type=self.device.type, dtype=torch.float16):
            (stoch, deter), mets = self._cal_grad(p_data, initial)
        self._scaler.unscale_(self._optimizer)  # unscale grads in params
        if self.rep_loss == "dreamerpro" and self._ema_updates < self.freeze_prototypes_iters:
            self._prototypes.grad.zero_()
        if self._log_grads:
            old_params = [p.data.clone().detach() for p in self._named_params.values()]
            grads = [p.grad for p in self._named_params.values() if p.grad is not None]  # log grads before clipping
            grad_norm = tools.compute_global_norm(grads)
            grad_rms = tools.compute_rms(grads)
            mets["opt/grad_norm"] = grad_norm
            mets["opt/grad_rms"] = grad_rms
        self._agc(self._named_params.values())  # clipping
        self._scaler.step(self._optimizer)  # update params
        self._scaler.update()  # adjust scale
        self._scheduler.step()  # increment scheduler
        self._optimizer.zero_grad(set_to_none=True)  # reset grads
        mets["opt/lr"] = self._scheduler.get_lr()[0]
        mets["opt/grad_scale"] = self._scaler.get_scale()
        # Log domain sampling stats when using DomainAwareBuffer
        if hasattr(replay_buffer, 'get_stats'):
            for k, v in replay_buffer.get_stats().items():
                mets[k] = v
        if self._log_grads:
            updates = [(new - old) for (new, old) in zip(self._named_params.values(), old_params)]
            update_rms = tools.compute_rms(updates)
            params_rms = tools.compute_rms(self._named_params.values())
            mets["opt/param_rms"] = params_rms
            mets["opt/update_rms"] = update_rms
        metrics.update(mets)
        # update latent vectors in replay buffer
        replay_buffer.update(index, stoch.detach(), deter.detach())
        return metrics

    def _cal_grad(self, data, initial):
        """Compute gradients for one batch.

        Notes
        -----
        This function computes:
        1) World model loss (dynamics + representation)
        2) Optional representation loss variants (Dreamer, R2-Dreamer, InfoNCE, DreamerPro)
        3) Imagination rollouts for actor-critic updates
        4) Replay-based value learning
        """
        # data: dict of (B, T, *), initial: (stoch: (B, S, K), deter: (B, D))
        losses = {}
        metrics = {}
        B, T = data.shape

        # === World model: posterior rollout and KL losses ===
        # (B, T, E)
        embed = self.encoder(data)
        # (B, T, S, K), (B, T, D), (B, T, S, K)
        post_stoch, post_deter, post_logit = self.rssm.observe(embed, data["action"], initial, data["is_first"])
        if self.sf_enabled:
            # SF-RSSM: compute prior logits per timestep with parent gating
            prior_logits = []
            all_gates = []
            all_gate_logits = []
            Bt, T = post_stoch.shape[:2]
            for t in range(T):
                if t == 0:
                    prev_stoch_t = initial[0]
                else:
                    prev_stoch_t = post_stoch[:, t - 1]
                prev_action_t = data["action"][:, t]
                from tools import rpad
                is_first_t = data["is_first"][:, t]
                prev_stoch_t = torch.where(
                    rpad(is_first_t, prev_stoch_t.dim() - int(is_first_t.dim())),
                    torch.zeros_like(prev_stoch_t), prev_stoch_t
                )
                prev_action_t = torch.where(
                    rpad(is_first_t, prev_action_t.dim() - int(is_first_t.dim())),
                    torch.zeros_like(prev_action_t), prev_action_t
                )
                deter_t = post_deter[:, t]
                _, logit_t, gates_t, gate_logits_t = self.rssm.prior(deter_t, prev_stoch_t, prev_action_t)
                prior_logits.append(logit_t)
                all_gates.append(gates_t)
                all_gate_logits.append(gate_logits_t)
            prior_logit = torch.stack(prior_logits, dim=1)
            avg_gates = torch.stack(all_gates, dim=1).mean(dim=(0, 1))
            sf_sparse_loss = self.rssm.sparsity_loss(avg_gates.unsqueeze(0))
            avg_gate_logits = torch.stack(all_gate_logits, dim=1).mean(dim=(0, 1))
            sf_center_loss = self.rssm.gate_center_loss(avg_gate_logits.unsqueeze(0))
        elif self.ensemble_enabled:
            # Ensemble RSSM: main prior = head_0 (DV3-quality KL),
            # all K heads used only for disagreement metric + actor penalty.
            prior_logits = []
            all_disagreements = []
            Bt, T = post_stoch.shape[:2]
            for t in range(T):
                deter_t = post_deter[:, t]
                logits_list, disagreement = self.rssm.ensemble_prior_logits(deter_t)
                # Use HEAD 0 as the main prior (standard DV3 behavior)
                prior_logits.append(logits_list[0])
                all_disagreements.append(disagreement)
            prior_logit = torch.stack(prior_logits, dim=1)  # (B, T, S, K)
            # Standard KL with head 0 only (same as DreamerV3 baseline)
            dyn_loss, rep_loss = self.rssm.kl_loss(post_logit, prior_logit, self.kl_free)
            losses["dyn"] = torch.mean(dyn_loss)
            losses["rep"] = torch.mean(rep_loss)
            # Average disagreement over batch and time for logging
            all_disc = torch.stack(all_disagreements, dim=1)  # (B, T)
            avg_disagreement = all_disc.mean().item()
        else:
            # (B, T, S, K)
            _, prior_logit = self.rssm.prior(post_deter)
        if not self.ensemble_enabled:
            dyn_loss, rep_loss = self.rssm.kl_loss(post_logit, prior_logit, self.kl_free)
            losses["dyn"] = torch.mean(dyn_loss)
            losses["rep"] = torch.mean(rep_loss)
        elif not self.sf_enabled:
            # Ensemble already set losses above; non-ensemble, non-sf case
            pass
        if self.sf_enabled:
            losses["sf_sparse"] = sf_sparse_loss
            losses["sf_center"] = sf_center_loss
        if self.ensemble_enabled:
            metrics["ensemble_disagreement"] = avg_disagreement
        # === Representation / auxiliary losses ===
        # (B, T, F)
        feat = self.rssm.get_feat(post_stoch, post_deter)
        if self.rep_loss == "dreamer":
            recon_losses = {
                key: torch.mean(-dist.log_prob(data[key])) for key, dist in self.decoder(post_stoch, post_deter).items()
            }
            losses.update(recon_losses)
        elif self.rep_loss == "r2dreamer":
            # R2-Dreamer: Barlow Twins style redundancy reduction between latent features and encoder embeddings.
            # Flatten batch/time dims for a single cross-correlation matrix.
            # (B, T, F) -> (B*T, F)
            x1 = self.prj(feat[:, :].reshape(B * T, -1))
            # (B, T, E) -> (B*T, E)
            x2 = embed.reshape(B * T, -1).detach()  # this detach is important

            x1_norm = (x1 - x1.mean(0)) / (x1.std(0) + 1e-8)
            x2_norm = (x2 - x2.mean(0)) / (x2.std(0) + 1e-8)

            c = torch.mm(x1_norm.T, x2_norm) / (B * T)
            invariance_loss = (torch.diagonal(c) - 1.0).pow(2).sum()
            off_diag_mask = ~torch.eye(x1.shape[-1], dtype=torch.bool, device=x1.device)
            redundancy_loss = c[off_diag_mask].pow(2).sum()
            losses["barlow"] = invariance_loss + self.barlow_lambd * redundancy_loss
        elif self.rep_loss == "infonce":
            # Contrastive (InfoNCE) objective between projected latent features and encoder embeddings.
            # (B, T, F) -> (B*T, F)
            x1 = self.prj(feat[:, :].reshape(B * T, -1))
            # (B, T, E) -> (B*T, E)
            x2 = embed.reshape(B * T, -1).detach()  # this detach is important
            logits = torch.matmul(x1, x2.T)
            norm_logits = logits - torch.max(logits, 1)[0][:, None]
            labels = torch.arange(norm_logits.shape[0]).long().to(self.device)
            losses["infonce"] = torch.nn.functional.cross_entropy(norm_logits, labels)
        elif self.rep_loss == "dreamerpro":
            # DreamerPro uses augmentation + EMA targets + Sinkhorn assignment.
            with torch.no_grad():
                data_aug = self.augment_data(data)
                initial_aug = (
                    # (B, ...) -> (2B, ...)
                    torch.cat([initial[0], initial[0]], dim=0),
                    torch.cat([initial[1], initial[1]], dim=0),
                )
                ema_proj = self.ema_proj(data_aug)

            embed_aug = self.encoder(data_aug)
            post_stoch_aug, post_deter_aug, _ = self.rssm.observe(
                embed_aug, data_aug["action"], initial_aug, data_aug["is_first"]
            )
            proto_losses = self.proto_loss(post_stoch_aug, post_deter_aug, embed_aug, ema_proj)
            losses.update(proto_losses)
        else:
            raise NotImplementedError

        # Two-Timescale structured prediction head losses (auxiliary)
        if self.tt_enabled or self.tt_v2_enabled:
            head_losses = self.rssm.compute_head_losses(post_stoch, post_deter, data)
            for k, v in head_losses.items():
                losses[k] = torch.mean(v)
            # Log gate diagnostics
            gate_mean = self.rssm.get_last_gate_mean()
            if gate_mean is not None:
                metrics['tt_global_gate'] = gate_mean.item()
        # SPER: Spatial Perception auxiliary losses (backward enabled — M2)
        if self.sper_enabled:
            sper_losses = self._sper.compute_losses(
                feat, embed, data,
                lambda_motion=self._sper_lambda_motion,
                lambda_depth=self._sper_lambda_depth,
            )
            losses.update(sper_losses)

        # SPER v2: 空间感知损失（BCE 运动掩码 + L1 深度 + BCE 接触，真实目标）。
        # 原始损失在此收集，权重统一由 loss_scales 缩放（sper_motion/depth/contact=0.01）
        if self.sper_v2_enabled:
            sper_v2_losses = self._sper_v2.compute_losses(feat, data)
            losses.update(sper_v2_losses)

        # reward and continue
        losses["rew"] = torch.mean(-self.reward(feat).log_prob(to_f32(data["reward"])))
        cont = 1.0 - to_f32(data["is_terminal"])
        losses["con"] = torch.mean(-self.cont(feat).log_prob(cont))
        # log
        metrics["dyn_entropy"] = torch.mean(self.rssm.get_dist(prior_logit).entropy())
        metrics["rep_entropy"] = torch.mean(self.rssm.get_dist(post_logit).entropy())
        if self.sf_enabled:
            metrics["sf_sparse_gate"] = avg_gates.mean().item()
            metrics["sf_gate_logit_mean"] = avg_gate_logits.mean().item()

        # === Imagination rollout for actor-critic ===
        # (B*T, S, K), (B*T, D)
        start = (
            post_stoch.reshape(-1, *post_stoch.shape[2:]).detach(),
            post_deter.reshape(-1, *post_deter.shape[2:]).detach(),
        )
        # (B, T, ...) -> (B*T, ...)
        if self.ensemble_enabled:
            imag_feat, imag_action, imag_disagreement = self._imagine(start, self.imag_horizon + 1)
        else:
            imag_feat, imag_action = self._imagine(start, self.imag_horizon + 1)
        imag_feat, imag_action = imag_feat.detach(), imag_action.detach()

        # (B*T, T_imag, 1)
        imag_reward = self._frozen_reward(imag_feat).mode()
        # (B*T, T_imag, 1)  probability of continuation
        imag_cont = self._frozen_cont(imag_feat).mean
        # (B*T, T_imag, 1)
        imag_value = self._frozen_value(imag_feat).mode()
        imag_slow_value = self._frozen_slow_value(imag_feat).mode()
        disc = 1 - 1 / self.horizon
        # (B*T, T_imag, 1)
        weight = torch.cumprod(imag_cont * disc, dim=1)
        last = torch.zeros_like(imag_cont)
        term = 1 - imag_cont
        ret = self._lambda_return(
            last, term, imag_reward, imag_value, imag_value, disc, self.lamb
        )  # (B*T, T_imag-1, 1)
        ret_offset, ret_scale = self.return_ema(ret)
        # (B*T, T_imag-1, 1)
        adv = (ret - imag_value[:, :-1]) / ret_scale

        # Ensemble disagreement penalty for actor
        if self.ensemble_enabled:
            # Normalize disagreement quantile
            disc_flat = imag_disagreement[:, :-1].detach()  # (B*T, T_imag-1)
            # Simple fixed penalty for M0
            penalty = self.ensemble_lambda_max * disc_flat.unsqueeze(-1)  # (B*T, T_imag-1, 1)
            adv = adv - penalty
            metrics["ensemble_disagreement_imagination"] = disc_flat.mean().item()

        policy = self.actor(self._fuse_feat(imag_feat))
        # (B*T, T_imag-1, 1)
        logpi = policy.log_prob(imag_action)[:, :-1].unsqueeze(-1)
        entropy = policy.entropy()[:, :-1].unsqueeze(-1)
        losses["policy"] = torch.mean(weight[:, :-1].detach() * -(logpi * adv.detach() + self.act_entropy * entropy))

        imag_value_dist = self.value(imag_feat)
        # (B*T, T_imag, 1)
        tar_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)
        losses["value"] = torch.mean(
            weight[:, :-1].detach()
            * (-imag_value_dist.log_prob(tar_padded.detach()) - imag_value_dist.log_prob(imag_slow_value.detach()))[
                :, :-1
            ].unsqueeze(-1)
        )
        # log
        ret_normed = (ret - ret_offset) / ret_scale
        metrics["ret"] = torch.mean(ret_normed)
        metrics["ret_005"] = self.return_ema.ema_vals[0]
        metrics["ret_095"] = self.return_ema.ema_vals[1]
        metrics["adv"] = torch.mean(adv)
        metrics["adv_std"] = torch.std(adv)
        metrics["con"] = torch.mean(imag_cont)
        metrics["rew"] = torch.mean(imag_reward)
        metrics["val"] = torch.mean(imag_value)
        metrics["tar"] = torch.mean(ret)
        metrics["slowval"] = torch.mean(imag_slow_value)
        metrics["weight"] = torch.mean(weight)
        metrics["action_entropy"] = torch.mean(entropy)
        metrics.update(tools.tensorstats(imag_action, "action"))

        # === Replay-based value learning (keep gradients through world model) ===
        last, term, reward = (
            to_f32(data["is_last"]),
            to_f32(data["is_terminal"]),
            to_f32(data["reward"]),
        )
        feat = self.rssm.get_feat(post_stoch, post_deter)
        boot = ret[:, 0].reshape(B, T, 1)
        value = self._frozen_value(feat).mode()
        slow_value = self._frozen_slow_value(feat).mode()
        disc = 1 - 1 / self.horizon
        weight = 1.0 - last
        ret = self._lambda_return(last, term, reward, value, boot, disc, self.lamb)
        ret_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)

        # Keep this attached to the world model so gradients can flow through
        value_dist = self.value(feat)
        losses["repval"] = torch.mean(
            weight[:, :-1]
            * (-value_dist.log_prob(ret_padded.detach()) - value_dist.log_prob(slow_value.detach()))[:, :-1].unsqueeze(
                -1
            )
        )
        # log
        metrics.update(tools.tensorstats(ret, "ret_replay"))
        metrics.update(tools.tensorstats(value, "value_replay"))
        metrics.update(tools.tensorstats(slow_value, "slow_value_replay"))

        # === Distillation loss for Nominal-Anchored Dynamics Randomization ===
        # Applied only on nominal (non-randomized) batches when teacher is loaded
        if self._teacher_actor is not None and "dynamics_mode" in data:
            try:
                is_nominal = bool(data["dynamics_mode"].mean().item() < 0.5)
            except Exception:
                is_nominal = False
            if is_nominal:
                feat_for_distill = self.rssm.get_feat(post_stoch, post_deter)
                distill_policy_weight = float(self._loss_scales.get("distill_policy", 0.0))
                distill_value_weight = float(self._loss_scales.get("distill_value", 0.05))
                # Policy distillation: KL(teacher || student) — optional
                if distill_policy_weight > 0:
                    with torch.no_grad():
                        teacher_policy = self._teacher_actor(feat_for_distill.detach())
                    student_policy = self.actor(feat_for_distill)
                    distill_policy = torch.distributions.kl.kl_divergence(
                        teacher_policy, student_policy
                    ).mean()
                    losses["distill_policy"] = distill_policy * distill_policy_weight
                    metrics["distill_policy"] = distill_policy.item()
                # Value distillation: MSE(teacher || student)
                if distill_value_weight > 0:
                    with torch.no_grad():
                        teacher_value = self._teacher_value(feat_for_distill.detach())
                    student_value = self.value(feat_for_distill)
                    distill_value = (teacher_value.mode() - student_value.mode()).pow(2).mean()
                    losses["distill_value"] = distill_value * distill_value_weight
                    metrics["distill_value"] = distill_value.item()

        total_loss = sum([v * self._loss_scales[k] for k, v in losses.items()])
        self._scaler.scale(total_loss).backward()

        metrics.update({f"loss/{name}": loss for name, loss in losses.items()})
        metrics.update({"opt/loss": total_loss})
        return (post_stoch, post_deter), metrics

    @torch.no_grad()
    def _imagine(self, start, imag_horizon):
        """Roll out the policy in latent space."""
        feats = []
        actions = []
        disagreements = [] if self.ensemble_enabled else None
        stoch, deter = start
        for _ in range(imag_horizon):
            feat = self._frozen_rssm.get_feat(stoch, deter)
            action = self._frozen_actor(self._fuse_feat(feat)).rsample()
            feats.append(feat)
            actions.append(action)
            if self.ensemble_enabled:
                stoch, deter, disc = self._frozen_rssm.img_step(stoch, deter, action)
                disagreements.append(disc)
            else:
                stoch, deter = self._frozen_rssm.img_step(stoch, deter, action)

        feats = torch.stack(feats, dim=1)
        actions = torch.stack(actions, dim=1)

        if self.ensemble_enabled:
            disagreements = torch.stack(disagreements, dim=1)  # (B*T, T_imag)
            return feats, actions, disagreements
        return feats, actions

    @torch.no_grad()
    def _lambda_return(self, last, term, reward, value, boot, disc, lamb):
        """
        lamb=1 means discounted Monte Carlo return.
        lamb=0 means fixed 1-step return.
        """
        assert last.shape == term.shape == reward.shape == value.shape == boot.shape
        live = (1 - to_f32(term))[:, 1:] * disc
        cont = (1 - to_f32(last))[:, 1:] * lamb
        interm = reward[:, 1:] + (1 - cont) * live * boot[:, 1:]
        out = [boot[:, -1]]
        for i in reversed(range(live.shape[1])):
            out.append(interm[:, i] + live[:, i] * cont[:, i] * out[-1])
        return torch.stack(list(reversed(out))[:-1], 1)

    def _fuse_feat(self, feat):
        """SGFM 决策阶段融合：actor 输入 = feat + 门控融合残差。

        fuse_actor=False 时推理/训练调用点返回纯 feat——三个预测头在推理阶段
        完全移除（零推理开销），SGFM 仅作为训练期信号存在。
        """
        if self.sper_v2_enabled and self._fuse_actor and self._sper_v2.sgfm is not None:
            return feat + self._sper_v2.fused_residual(feat)
        return feat

    @torch.no_grad()
    def preprocess(self, data):
        if "image" in data:
            data["image"] = to_f32(data["image"]) / 255.0
        return data

    @torch.no_grad()
    def augment_data(self, data):
        data_aug = {k: torch.cat([v, v], axis=0) for k, v in data.items()}
        # (B, T, H, W, C) -> (B, T, C, H, W)
        image = data_aug["image"].permute(0, 1, 4, 2, 3)
        data_aug["image"] = self.random_translate(
            image,
            self.aug_max_delta,
            same_across_time=self.aug_same_across_time,
            bilinear=self.aug_bilinear,
        )
        # (B, T, C, H, W) -> (B, T, H, W, C)
        data_aug["image"] = data_aug["image"].permute(0, 1, 3, 4, 2)
        return data_aug

    @torch.no_grad()
    def ema_proj(self, data):
        with torch.no_grad():
            embed = self._ema_encoder(data)
            proj = self._ema_obs_proj(embed)
        return F.normalize(proj, p=2, dim=-1)

    @torch.no_grad()
    def ema_update(self):
        prototypes = F.normalize(self._prototypes, p=2, dim=-1)
        self._prototypes.data.copy_(prototypes)
        if self._ema_updates % self.ema_update_every == 0:
            mix = self.ema_update_fraction if self._ema_updates > 0 else 1.0
            for s, d in zip(self.encoder.parameters(), self._ema_encoder.parameters()):
                d.data.copy_(mix * s.data + (1 - mix) * d.data)
            for s, d in zip(self.obs_proj.parameters(), self._ema_obs_proj.parameters()):
                d.data.copy_(mix * s.data + (1 - mix) * d.data)
        self._ema_updates += 1

    def sinkhorn(self, scores):
        """Sinkhorn-Knopp normalization.

        Notes
        -----
        Given a score matrix, we iteratively normalize rows and columns in log
        space so that the resulting assignment matrix is approximately doubly
        stochastic.
        """
        shape = scores.shape
        K = shape[0]
        scores = scores.reshape(-1)
        log_Q = F.log_softmax(scores / self.sinkhorn_eps, dim=0)
        log_Q = log_Q.reshape(K, -1)
        N = log_Q.shape[1]
        for _ in range(self.sinkhorn_iters):
            log_row_sums = torch.logsumexp(log_Q, dim=1, keepdim=True)
            log_Q = log_Q - log_row_sums - math.log(K)
            log_col_sums = torch.logsumexp(log_Q, dim=0, keepdim=True)
            log_Q = log_Q - log_col_sums - math.log(N)
        log_Q = log_Q + math.log(N)
        Q = torch.exp(log_Q)
        return Q.reshape(shape)

    def proto_loss(self, post_stoch, post_deter, embed, ema_proj):
        prototypes = F.normalize(self._prototypes, p=2, dim=-1)

        obs_proj = self.obs_proj(embed)
        obs_norm = torch.norm(obs_proj, dim=-1)
        obs_proj = F.normalize(obs_proj, p=2, dim=-1)

        B, T = obs_proj.shape[:2]
        # (B, T, P) -> (B*T, P)
        obs_proj = obs_proj.reshape(B * T, -1)
        obs_scores = torch.matmul(obs_proj, prototypes.T)
        # (B*T, K) -> (B, T, K) -> (K, B, T)
        obs_scores = obs_scores.reshape(B, T, -1).permute(2, 0, 1)
        obs_scores = obs_scores[:, :, self.warm_up :]
        obs_logits = F.log_softmax(obs_scores / self.temperature, dim=0)
        obs_logits_1, obs_logits_2 = torch.chunk(obs_logits, 2, dim=1)

        # (B, T, P) -> (B*T, P)
        ema_proj = ema_proj.reshape(B * T, -1)
        ema_scores = torch.matmul(ema_proj, prototypes.T)
        # (B*T, K) -> (B, T, K) -> (K, B, T)
        ema_scores = ema_scores.reshape(B, T, -1).permute(2, 0, 1)
        ema_scores = ema_scores[:, :, self.warm_up :]
        ema_scores_1, ema_scores_2 = torch.chunk(ema_scores, 2, dim=1)

        with torch.no_grad():
            ema_targets_1 = self.sinkhorn(ema_scores_1)
            ema_targets_2 = self.sinkhorn(ema_scores_2)
        ema_targets = torch.cat([ema_targets_1, ema_targets_2], dim=1)

        feat = self.rssm.get_feat(post_stoch, post_deter)
        feat_proj = self.feat_proj(feat)
        feat_norm = torch.norm(feat_proj, dim=-1)
        feat_proj = F.normalize(feat_proj, p=2, dim=-1)

        # (B, T, P) -> (B*T, P)
        feat_proj = feat_proj.reshape(B * T, -1)
        feat_scores = torch.matmul(feat_proj, prototypes.T)
        # (B*T, K) -> (B, T, K) -> (K, B, T)
        feat_scores = feat_scores.reshape(B, T, -1).permute(2, 0, 1)
        feat_scores = feat_scores[:, :, self.warm_up :]
        feat_logits = F.log_softmax(feat_scores / self.temperature, dim=0)

        swav_loss = -0.5 * torch.mean(torch.sum(ema_targets_2 * obs_logits_1, dim=0)) - 0.5 * torch.mean(
            torch.sum(ema_targets_1 * obs_logits_2, dim=0)
        )
        temp_loss = -torch.mean(torch.sum(ema_targets * feat_logits, dim=0))
        norm_loss = torch.mean(torch.square(obs_norm - 1)) + torch.mean(torch.square(feat_norm - 1))

        return {
            "swav": swav_loss,
            "temp": temp_loss,
            "norm": norm_loss,
        }

    @torch.no_grad()
    def random_translate(self, x, max_delta, same_across_time=False, bilinear=False):
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        pad = int(max_delta)

        # Pad
        x_padded = F.pad(x_flat, (pad, pad, pad, pad), "replicate")
        h_padded, w_padded = H + 2 * pad, W + 2 * pad

        # Create base grid
        eps_h = 1.0 / h_padded
        eps_w = 1.0 / w_padded
        arange_h = torch.linspace(-1.0 + eps_h, 1.0 - eps_h, h_padded, device=x.device, dtype=x.dtype)[:H]
        arange_w = torch.linspace(-1.0 + eps_w, 1.0 - eps_w, w_padded, device=x.device, dtype=x.dtype)[:W]
        arange_h = arange_h.unsqueeze(1).repeat(1, W).unsqueeze(2)
        arange_w = arange_w.unsqueeze(0).repeat(H, 1).unsqueeze(2)
        base_grid = torch.cat([arange_w, arange_h], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(B * T, 1, 1, 1)

        # Create shift
        if same_across_time:
            shift = torch.randint(0, 2 * pad + 1, size=(B, 1, 1, 1, 2), device=x.device, dtype=x.dtype)
            shift = shift.repeat(1, T, 1, 1, 1).reshape(B * T, 1, 1, 2)
        else:
            shift = torch.randint(0, 2 * pad + 1, size=(B * T, 1, 1, 2), device=x.device, dtype=x.dtype)

        shift = shift * 2.0 / torch.tensor([w_padded, h_padded], device=x.device, dtype=x.dtype)

        # Apply shift and sample
        grid = base_grid + shift
        mode = "bilinear" if bilinear else "nearest"
        x_translated = F.grid_sample(x_padded, grid, mode=mode, padding_mode="zeros", align_corners=False)

        return x_translated.reshape(B, T, C, H, W)
