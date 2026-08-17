"""
Two-Timescale RSSM v2 — scalar heuristic gate (no MLP overhead).

Key change from v1: replaced the learned MLP gate (64→1, ~66K params)
with a single learnable scalar alpha: gate = sigmoid(alpha * RMS(Δh)).

This eliminates the 2× FPS overhead while preserving adaptive two-timescale behavior.

Same safety principle as v1: ONLY the deterministic path is modified.
Posterior, prior, and KL loss are UNCHANGED.
"""

import torch
from torch import nn

from networks import BlockLinear
from tools import weight_init_, rpad


class TwoTimescaleDeterV2(nn.Module):
    """Block-GRU with scalar two-timescale gate.

    gate = sigmoid(alpha * RMS(h_global_new - h_global_old))

    alpha is a single learnable scalar (initialized to 0 → gate ≈ 0.5).
    Positive alpha → high-surprise steps get faster global updates.
    Negative alpha → high-surprise steps get slower updates (unlikely but possible).
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

        # Standard Deter input projections (unchanged from rssm.Deter)
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

        # Hidden layers (unchanged)
        self._dyn_hid = nn.Sequential()
        in_ch = (3 * hidden + deter // self.blocks) * self.blocks
        for i in range(self.dynlayers):
            self._dyn_hid.add_module(
                f"dyn_hid_{i}", BlockLinear(in_ch, deter, self.blocks))
            self._dyn_hid.add_module(
                f"norm_{i}", nn.RMSNorm(deter, eps=1e-04, dtype=torch.float32))
            self._dyn_hid.add_module(f"act_{i}", act_fn())
            in_ch = deter

        # GRU cell (unchanged)
        self._dyn_gru = BlockLinear(in_ch, 3 * deter, self.blocks)

        # v2: Scalar gate — single learnable parameter
        # alpha=0 init → sigmoid(0)=0.5 (neutral: half-update global)
        self._gate_alpha = nn.Parameter(torch.tensor(0.0))

        self.flat2group = lambda x: x.reshape(*x.shape[:-1], self.blocks, -1)
        self.group2flat = lambda x: x.reshape(*x.shape[:-2], -1)

    def forward(self, stoch, deter, action):
        """Deterministic transition with scalar two-timescale gate.

        Returns: (deter_out, gate_mean_tensor)
        """
        B = action.shape[0]

        # Standard GRU forward pass (unchanged)
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

        candidate_full = update * cand + (1 - update) * deter

        # Split into global and local
        h_global_old = deter[:, :self.global_dim]
        h_global_new = candidate_full[:, :self.global_dim]
        h_local_new = candidate_full[:, self.global_dim:]

        # Local: always full update
        h_local = h_local_new

        # v2: Scalar gate — sigmoid(alpha * RMS(delta))
        diff = h_global_new - h_global_old
        rms_delta = diff.pow(2).mean(-1, keepdim=True).sqrt().detach()
        # alpha * rms: alpha controls sensitivity to surprise
        global_gate = torch.sigmoid(self._gate_alpha * rms_delta)  # (B, 1)

        h_global = global_gate * h_global_new + (1 - global_gate) * h_global_old

        deter_out = torch.cat([h_global, h_local], dim=-1)

        return deter_out, global_gate.detach().mean()


class TwoTimescaleRSSMv2(nn.Module):
    """RSSM with v2 scalar two-timescale gate.

    API-compatible with standard RSSM. Same as v1 but uses scalar gate.
    """

    def __init__(self, config, embed_size, act_dim):
        super().__init__()
        self._stoch = int(config.stoch)
        self._deter = int(config.deter)
        self._hidden = int(config.hidden)
        self._discrete = int(config.discrete)
        act_fn = getattr(torch.nn, config.act)
        self._unimix_ratio = float(config.unimix_ratio)
        self._device = torch.device(config.device)
        self._act_dim = act_dim
        self._obs_layers = int(config.obs_layers)
        self._img_layers = int(config.img_layers)
        self._dyn_layers = int(config.dyn_layers)
        self._blocks = int(config.blocks)
        self.flat_stoch = self._stoch * self._discrete
        self.feat_size = self.flat_stoch + self._deter

        self.global_ratio = float(getattr(config, 'tt_global_ratio', 0.25))

        self._deter_net = TwoTimescaleDeterV2(
            self._deter, self.flat_stoch, act_dim, self._hidden,
            blocks=self._blocks, dynlayers=self._dyn_layers,
            global_ratio=self.global_ratio, act=config.act,
        )

        # Posterior (unchanged)
        self._obs_net = nn.Sequential()
        inp_dim = self._deter + embed_size
        for i in range(self._obs_layers):
            self._obs_net.add_module(f"obs_net_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
            self._obs_net.add_module(f"obs_net_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
            self._obs_net.add_module(f"obs_net_a_{i}", act_fn())
            inp_dim = self._hidden
        self._obs_net.add_module("obs_net_logit", nn.Linear(inp_dim, self._stoch * self._discrete, bias=True))

        # Prior (unchanged)
        self._img_net = nn.Sequential()
        inp_dim = self._deter
        for i in range(self._img_layers):
            self._img_net.add_module(f"img_net_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
            self._img_net.add_module(f"img_net_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
            self._img_net.add_module(f"img_net_a_{i}", act_fn())
            inp_dim = self._hidden
        self._img_net.add_module("img_net_logit", nn.Linear(inp_dim, self._stoch * self._discrete, bias=True))

        self.apply(weight_init_)
        self.register_buffer("_last_gate_mean", torch.tensor(0.5))

    def get_last_gate_mean(self):
        return self._last_gate_mean

    def initial(self, batch_size):
        deter = torch.zeros(batch_size, self._deter, dtype=torch.float32, device=self._device)
        stoch = torch.zeros(batch_size, self._stoch, self._discrete, dtype=torch.float32, device=self._device)
        return stoch, deter

    def observe(self, embed, action, initial, reset):
        L = action.shape[1]
        stoch, deter = initial
        stochs, deters, logits = [], [], []
        gate_means = []
        for i in range(L):
            stoch, deter, logit, gm = self._obs_step_with_gate(stoch, deter, action[:, i], embed[:, i], reset[:, i])
            stochs.append(stoch); deters.append(deter); logits.append(logit); gate_means.append(gm)
        self._last_gate_mean = torch.stack(gate_means).mean()
        return torch.stack(stochs, dim=1), torch.stack(deters, dim=1), torch.stack(logits, dim=1)

    def obs_step(self, stoch, deter, prev_action, embed, reset):
        s, d, l, _ = self._obs_step_with_gate(stoch, deter, prev_action, embed, reset)
        return s, d, l

    def img_step(self, stoch, deter, prev_action):
        deter, _ = self._deter_net(stoch, deter, prev_action)
        stoch, _ = self.prior(deter)
        return stoch, deter

    def imagine_with_action(self, stoch, deter, actions):
        L = actions.shape[1]
        stochs, deters = [], []
        for i in range(L):
            stoch, deter = self.img_step(stoch, deter, actions[:, i])
            stochs.append(stoch); deters.append(deter)
        return torch.stack(stochs, dim=1), torch.stack(deters, dim=1)

    def prior(self, deter):
        logit = self._img_net(deter)
        logit = logit.reshape(*logit.shape[:-1], self._stoch, self._discrete)
        return self.get_dist(logit).rsample(), logit

    def get_feat(self, stoch, deter):
        stoch = stoch.reshape(*stoch.shape[:-2], self._stoch * self._discrete)
        return torch.cat([stoch, deter], -1)

    def get_dist(self, logit):
        import distributions as dists
        from torch import distributions as torchd
        return torchd.independent.Independent(dists.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1)

    def kl_loss(self, post_logit, prior_logit, free):
        import distributions as dists
        kld = dists.kl
        rep_loss = kld(post_logit, prior_logit.detach()).sum(-1)
        dyn_loss = kld(post_logit.detach(), prior_logit).sum(-1)
        return torch.clip(dyn_loss, min=free), torch.clip(rep_loss, min=free)

    def compute_head_losses(self, stoch, deter, data):
        """Stub — v2 has no structured heads. Returns empty dict for API compat."""
        return {}

    def _obs_step_with_gate(self, stoch, deter, prev_action, embed, reset):
        stoch = torch.where(rpad(reset, stoch.dim()-int(reset.dim())).bool(), torch.zeros_like(stoch), stoch)
        deter = torch.where(rpad(reset, deter.dim()-int(reset.dim())).bool(), torch.zeros_like(deter), deter)
        prev_action = torch.where(rpad(reset, prev_action.dim()-int(reset.dim())).bool(), torch.zeros_like(prev_action), prev_action)
        deter, gate_mean = self._deter_net(stoch, deter, prev_action)
        x = torch.cat([deter, embed], dim=-1)
        logit = self._obs_net(x).reshape(*x.shape[:-1], self._stoch, self._discrete)
        return self.get_dist(logit).rsample(), deter, logit, gate_mean
