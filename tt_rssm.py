"""
Two-Timescale RSSM — structured memory decomposition in deterministic path.

Splits the RSSM deterministic hidden state into:
  - h_global: slow-updating, coarse-grained (task phase, goal progress)
  - h_local:  fast-updating, fine-grained  (contact, occlusion, object state)

Key design choice: ONLY modifies the deterministic path (Deter module).
The posterior/prior (obs_net/img_net) and KL loss are UNCHANGED —
this avoids the prior-KL collapse that killed SF-RSSM and Ensemble RSSM.

API-compatible with standard RSSM:
  - obs_step() returns 3 values (stoch, deter, logit) ← matches Dreamer.act()
  - observe() returns 3 values (stochs, deters, logits)
  - img_step() returns 2 values (stoch, deter)
  - prior() returns 2 values (stoch, logit)
"""

import math
import torch
from torch import nn

from networks import BlockLinear
from tools import weight_init_, rpad


class TwoTimescaleDeter(nn.Module):
    """Block-GRU deterministic transition with two-timescale update.

    After the standard GRU cell, the output is split into global and local
    components. The global component uses a learned slow update gate with
    bounded range (0.01–0.50) while the local component updates every step.

    Args:
        deter: total deterministic state dimension
        global_ratio: fraction of deter allocated to global memory (default 0.25)
    """

    def __init__(self, deter, stoch, act_dim, hidden, blocks, dynlayers,
                 global_ratio=0.25, act="SiLU"):
        super().__init__()
        self.blocks = int(blocks)
        self.dynlayers = int(dynlayers)
        act_fn = getattr(torch.nn, act)

        self.deter_dim = deter
        self.global_dim = max(1, int(deter * global_ratio))
        self.local_dim = deter - self.global_dim

        # --- Standard Deter input projections (unchanged from rssm.Deter) ---
        self._dyn_in0 = nn.Sequential(
            nn.Linear(deter, hidden, bias=True),
            nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act_fn()
        )
        self._dyn_in1 = nn.Sequential(
            nn.Linear(stoch, hidden, bias=True),
            nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act_fn()
        )
        self._dyn_in2 = nn.Sequential(
            nn.Linear(act_dim, hidden, bias=True),
            nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act_fn()
        )

        # --- Hidden layers (unchanged) ---
        self._dyn_hid = nn.Sequential()
        in_ch = (3 * hidden + deter // self.blocks) * self.blocks
        for i in range(self.dynlayers):
            self._dyn_hid.add_module(
                f"dyn_hid_{i}", BlockLinear(in_ch, deter, self.blocks))
            self._dyn_hid.add_module(
                f"norm_{i}", nn.RMSNorm(deter, eps=1e-04, dtype=torch.float32))
            self._dyn_hid.add_module(f"act_{i}", act_fn())
            in_ch = deter

        # --- GRU cell (unchanged) ---
        self._dyn_gru = BlockLinear(in_ch, 3 * deter, self.blocks)

        # --- Two-timescale gate ---
        # Bounded gate: raw ∈ [0,1] → scaled to [gate_min, gate_max]
        # This prevents full collapse (gate=0 freezes memory) or full identity
        self.gate_min = 0.01
        self.gate_max = 0.50
        gate_in = self.global_dim * 2 + 1  # new_global + old_global + RMS-surprise
        self._global_gate = nn.Sequential(
            nn.Linear(gate_in, 64, bias=True),
            nn.RMSNorm(64, eps=1e-04, dtype=torch.float32),
            nn.SiLU(),
            nn.Linear(64, 1, bias=True),
        )
        # Initialize bias for slow default update (~0.05 within bounded range)
        last_linear = self._global_gate[-1]
        raw_bias = math.log(self.gate_min / (1.0 - self.gate_min) + 1e-8)
        nn.init.constant_(last_linear.bias, raw_bias)

        self.flat2group = lambda x: x.reshape(*x.shape[:-1], self.blocks, -1)
        self.group2flat = lambda x: x.reshape(*x.shape[:-2], -1)

    def forward(self, stoch, deter, action):
        """Deterministic state transition with two-timescale update.

        Returns:
            deter: (B, D) updated deterministic state
            gate_mean: scalar tensor (no .item() — keeps CUDA graph compatibility)
        """
        B = action.shape[0]

        # --- Standard GRU forward pass (identical to rssm.Deter) ---
        stoch_flat = stoch.reshape(B, -1)
        action_norm = action / torch.clip(torch.abs(action), min=1.0).detach()

        x0 = self._dyn_in0(deter)
        x1 = self._dyn_in1(stoch_flat)
        x2 = self._dyn_in2(action_norm)

        x = torch.cat([x0, x1, x2], -1)
        x = x.unsqueeze(-2).expand(-1, self.blocks, -1)
        x = self.group2flat(torch.cat([self.flat2group(deter), x], -1))
        x = self._dyn_hid(x)
        x = self._dyn_gru(x)

        gates = torch.chunk(self.flat2group(x), 3, dim=-1)
        reset, cand, update = (self.group2flat(g) for g in gates)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)

        # Full candidate (standard GRU output)
        candidate_full = update * cand + (1 - update) * deter

        # --- Split into global and local ---
        h_global_old = deter[:, :self.global_dim]
        h_global_new = candidate_full[:, :self.global_dim]
        h_local_new = candidate_full[:, self.global_dim:]

        # Local: always full update (standard GRU behavior)
        h_local = h_local_new

        # Global: learned slow gate with RMS-normalized surprise
        # RMS norm prevents scale dependency on global_dim
        diff = h_global_new - h_global_old
        rms_surprise = diff.pow(2).mean(-1, keepdim=True).sqrt().detach()
        gate_input = torch.cat([h_global_new, h_global_old, rms_surprise], dim=-1)
        raw_gate = torch.sigmoid(self._global_gate(gate_input))  # (B, 1)
        # Scale to bounded range [gate_min, gate_max]
        global_gate = self.gate_min + (self.gate_max - self.gate_min) * raw_gate

        h_global = global_gate * h_global_new + (1 - global_gate) * h_global_old

        deter_out = torch.cat([h_global, h_local], dim=-1)

        # Return tensor (not .item()) for CUDA graph / torch.compile compat
        return deter_out, global_gate.detach().mean()


class TwoTimescaleRSSM(nn.Module):
    """RSSM with two-timescale deterministic memory.

    API-compatible with standard RSSM:
      - obs_step() returns (stoch, deter, logit) — 3 values for Dreamer.act()
      - observe()  returns (stochs, deters, logits) — 3 values
      - img_step() returns (stoch, deter) — 2 values
      - prior()    returns (stoch, logit) — 2 values
      - kl_loss()  unchanged

    Gate diagnostics stored internally; get_last_gate_mean() returns scalar tensor.
    """

    def __init__(self, config, embed_size, act_dim):
        super().__init__()
        self._stoch = int(config.stoch)
        self._deter = int(config.deter)
        self._hidden = int(config.hidden)
        self._discrete = int(config.discrete)
        act_fn = getattr(torch.nn, config.act)
        self._unimix_ratio = float(config.unimix_ratio)
        self._initial = str(config.initial)
        self._device = torch.device(config.device)
        self._act_dim = act_dim
        self._obs_layers = int(config.obs_layers)
        self._img_layers = int(config.img_layers)
        self._dyn_layers = int(config.dyn_layers)
        self._blocks = int(config.blocks)
        self.flat_stoch = self._stoch * self._discrete
        self.feat_size = self.flat_stoch + self._deter

        # Two-timescale config
        self.global_ratio = float(getattr(config, 'tt_global_ratio', 0.25))
        self.tt_enable_heads = bool(getattr(config, 'tt_enable_heads', False))

        # Two-Timescale deterministic transition
        self._deter_net = TwoTimescaleDeter(
            self._deter, self.flat_stoch, act_dim, self._hidden,
            blocks=self._blocks, dynlayers=self._dyn_layers,
            global_ratio=self.global_ratio,
            act=config.act,
        )

        # Posterior net (UNCHANGED from standard RSSM)
        self._obs_net = nn.Sequential()
        inp_dim = self._deter + embed_size
        for i in range(self._obs_layers):
            self._obs_net.add_module(
                f"obs_net_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
            self._obs_net.add_module(
                f"obs_net_n_{i}",
                nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
            self._obs_net.add_module(f"obs_net_a_{i}", act_fn())
            inp_dim = self._hidden
        self._obs_net.add_module(
            "obs_net_logit",
            nn.Linear(inp_dim, self._stoch * self._discrete, bias=True))

        # Prior net (UNCHANGED from standard RSSM)
        self._img_net = nn.Sequential()
        inp_dim = self._deter
        for i in range(self._img_layers):
            self._img_net.add_module(
                f"img_net_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
            self._img_net.add_module(
                f"img_net_n_{i}",
                nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
            self._img_net.add_module(f"img_net_a_{i}", act_fn())
            inp_dim = self._hidden
        self._img_net.add_module(
            "img_net_logit",
            nn.Linear(inp_dim, self._stoch * self._discrete, bias=True))

        # --- Structured prediction heads (auxiliary probes, disabled by default) ---
        # When enabled, use lightweight shared-trunk design to limit param overhead
        if self.tt_enable_heads:
            probe_hidden = min(self._hidden, 128)
            # Shared trunk
            self._probe_trunk = nn.Sequential(
                nn.Linear(self.feat_size, probe_hidden, bias=True),
                nn.RMSNorm(probe_hidden, eps=1e-04, dtype=torch.float32),
                nn.SiLU(),
            )
            # Lightweight per-task heads
            self._progress_head = nn.Linear(probe_hidden, 1, bias=True)
            self._contact_head = nn.Linear(probe_hidden, 1, bias=True)
            self._term_head = nn.Linear(probe_hidden, 1, bias=True)

        self.apply(weight_init_)

        # Internal diagnostic storage (tensor, not float)
        self.register_buffer("_last_gate_mean", torch.tensor(0.5))

    def get_last_gate_mean(self):
        """Return last batch's average global gate (scalar tensor)."""
        return self._last_gate_mean

    def initial(self, batch_size):
        deter = torch.zeros(
            batch_size, self._deter, dtype=torch.float32, device=self._device)
        stoch = torch.zeros(
            batch_size, self._stoch, self._discrete,
            dtype=torch.float32, device=self._device)
        return stoch, deter

    # ── Public API (matches standard RSSM exactly) ──

    def observe(self, embed, action, initial, reset):
        """Posterior rollout. Returns (stochs, deters, logits)."""
        L = action.shape[1]
        stoch, deter = initial
        stochs, deters, logits = [], [], []
        gate_means = []
        for i in range(L):
            stoch, deter, logit, gate_mean = self._obs_step_with_gate(
                stoch, deter, action[:, i], embed[:, i], reset[:, i])
            stochs.append(stoch)
            deters.append(deter)
            logits.append(logit)
            gate_means.append(gate_mean)
        stochs = torch.stack(stochs, dim=1)
        deters = torch.stack(deters, dim=1)
        logits = torch.stack(logits, dim=1)
        # Average gate mean over timesteps
        self._last_gate_mean = torch.stack(gate_means).mean()
        return stochs, deters, logits

    def obs_step(self, stoch, deter, prev_action, embed, reset):
        """Single posterior step. Returns (stoch, deter, logit) — 3 values for Dreamer.act()."""
        stoch, deter, logit, _ = self._obs_step_with_gate(
            stoch, deter, prev_action, embed, reset)
        return stoch, deter, logit

    def img_step(self, stoch, deter, prev_action):
        """Single prior step. Returns (stoch, deter)."""
        deter, _ = self._deter_net(stoch, deter, prev_action)
        stoch, _ = self.prior(deter)
        return stoch, deter

    def imagine_with_action(self, stoch, deter, actions):
        """Roll out prior dynamics."""
        L = actions.shape[1]
        stochs, deters = [], []
        for i in range(L):
            stoch, deter = self.img_step(stoch, deter, actions[:, i])
            stochs.append(stoch)
            deters.append(deter)
        stochs = torch.stack(stochs, dim=1)
        deters = torch.stack(deters, dim=1)
        return stochs, deters

    def prior(self, deter):
        """Prior distribution. Returns (stoch, logit)."""
        logit = self._img_net(deter)
        logit = logit.reshape(*logit.shape[:-1], self._stoch, self._discrete)
        stoch = self.get_dist(logit).rsample()
        return stoch, logit

    def get_feat(self, stoch, deter):
        """Flatten stoch and concatenate with deter."""
        stoch = stoch.reshape(*stoch.shape[:-2], self._stoch * self._discrete)
        return torch.cat([stoch, deter], -1)

    def get_dist(self, logit):
        import distributions as dists
        from torch import distributions as torchd
        return torchd.independent.Independent(
            dists.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1)

    def kl_loss(self, post_logit, prior_logit, free):
        """KL loss (unchanged from standard RSSM)."""
        import distributions as dists
        kld = dists.kl
        rep_loss = kld(post_logit, prior_logit.detach()).sum(-1)
        dyn_loss = kld(post_logit.detach(), prior_logit).sum(-1)
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        return dyn_loss, rep_loss

    # ── Private helpers ──

    def _obs_step_with_gate(self, stoch, deter, prev_action, embed, reset):
        """Single posterior step with gate info. Returns (stoch, deter, logit, gate_mean)."""
        # Separate rpad per tensor to ensure correct broadcast shapes
        stoch = torch.where(
            rpad(reset, stoch.dim() - int(reset.dim())).bool(),
            torch.zeros_like(stoch), stoch)
        deter = torch.where(
            rpad(reset, deter.dim() - int(reset.dim())).bool(),
            torch.zeros_like(deter), deter)
        prev_action = torch.where(
            rpad(reset, prev_action.dim() - int(reset.dim())).bool(),
            torch.zeros_like(prev_action), prev_action)

        # Two-timescale deter transition
        deter, gate_mean = self._deter_net(stoch, deter, prev_action)

        # Posterior (unchanged)
        x = torch.cat([deter, embed], dim=-1)
        logit = self._obs_net(x)
        logit = logit.reshape(*x.shape[:-1], self._stoch, self._discrete)
        stoch = self.get_dist(logit).rsample()

        return stoch, deter, logit, gate_mean

    # ── Structured prediction heads (auxiliary probes, on detached features) ──

    def compute_head_losses(self, stoch, deter, data):
        """Compute auxiliary prediction losses from detached features.

        Only term_head active by default (uses is_terminal from replay).
        Progress and contact require labels stored as 'log_progress'/'log_contact'
        in replay buffer (with 'log_' prefix to prevent encoder leakage).

        Returns dict of loss_name → scalar tensor.
        """
        if not self.tt_enable_heads:
            return {}

        feat = self.get_feat(stoch, deter).detach()
        losses = {}

        if hasattr(self, '_progress_head') and 'log_progress' in data:
            pred = self._progress_head(self._probe_trunk(feat))
            target = data['log_progress'].float()
            losses['tt_progress'] = F.mse_loss(pred, target)

        if hasattr(self, '_contact_head') and 'log_contact' in data:
            pred = self._contact_head(self._probe_trunk(feat))
            target = data['log_contact'].float()
            losses['tt_contact'] = F.binary_cross_entropy_with_logits(pred, target)

        if hasattr(self, '_term_head'):
            pred = self._term_head(self._probe_trunk(feat))
            # Predict continuation (1 - is_terminal), consistent with DreamerV3 cont head
            target = 1.0 - data['is_terminal'].float()
            losses['tt_cont'] = F.binary_cross_entropy_with_logits(pred, target)

        return losses

    # ── Memory specialization diagnostic ──

    @torch.no_grad()
    def memory_specialization_diagnostic(self, stoch, deter):
        """Causal ablation: zero h_global vs h_local, measure prediction delta.

        Uses ellipsis slicing for compatibility with any leading batch dims.
        """
        feat_full = self.get_feat(stoch, deter)
        global_dim = self._deter_net.global_dim

        deter_noglobal = deter.clone()
        deter_noglobal[..., :global_dim] = 0.0
        feat_noglobal = self.get_feat(stoch, deter_noglobal)

        deter_nolocal = deter.clone()
        deter_nolocal[..., global_dim:] = 0.0
        feat_nolocal = self.get_feat(stoch, deter_nolocal)

        results = {}
        if not self.tt_enable_heads:
            return results

        for name, head_attr in [('progress', '_progress_head'),
                                ('contact', '_contact_head'),
                                ('term', '_term_head')]:
            if not hasattr(self, head_attr):
                continue
            h = getattr(self, head_attr)
            p_full = h(self._probe_trunk(feat_full))
            p_nog = h(self._probe_trunk(feat_noglobal))
            p_nol = h(self._probe_trunk(feat_nolocal))
            results[f"{name}_delta_global"] = (p_full - p_nog).abs().mean().item()
            results[f"{name}_delta_local"] = (p_full - p_nol).abs().mean().item()

        return results


import torch.nn.functional as F
