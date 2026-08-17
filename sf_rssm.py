"""
Sparse Factorized RSSM — structurally constrained priors for DreamerV3.

Modifies the RSSM prior to use:
  - Fixed factor partitions of the stochastic state z_t
  - Factor-specific h_t projections: h_t^i = W_i * stop_grad(h_t) + b_i
  - Binary concrete (Gumbel-sigmoid) parent gates G_{j->i} with L0 sparsity
  - Shared transition network conditioned on (h_t^i, gathered parents, a_{t-1})
"""
import torch
from torch import nn
import torch.nn.functional as F

from tools import weight_init_


def binary_concrete(logits, temperature=0.5, hard=False):
    """Binary concrete / Gumbel-sigmoid relaxation.

    Compatible back to PyTorch 2.0 (no gumbel_sigmoid needed).
    """
    u = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
    g = torch.log(u) - torch.log1p(-u)
    y = torch.sigmoid((logits + g) / temperature)
    if hard:
        y_hard = (y > 0.5).float()
        y = y_hard - y.detach() + y  # straight-through
    return y


class FactorGate(nn.Module):
    """Learnable parent gates G_{j->i} using binary concrete relaxation.

    For K factors, produces KxK gate matrix where G[j,i] = gate from j to i.
    Self-loops (diagonal) are always 1. Action is always a parent (not gated).
    """

    def __init__(self, num_factors, deter_dim, hidden=64):
        super().__init__()
        self.num_factors = num_factors
        self.factor_embed = nn.Embedding(num_factors, hidden)
        self.gate_net = nn.Sequential(
            nn.Linear(2 * hidden + deter_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, deter, temperature=0.5, hard=False):
        """Compute parent gate matrix.

        Args:
            deter: (B, D) deterministic state
            temperature: binary concrete temperature
            hard: if True, use hard threshold with straight-through

        Returns:
            gates: (B, K, K) where gates[b,j,i] = gate from factor j to i
            gate_logits: (B, K, K) raw logits (for logging)
        """
        B, D = deter.shape
        K = self.num_factors

        # Build all ordered factor pairs: (K, K, 2H)
        emb = self.factor_embed(torch.arange(K, device=deter.device))
        emb_pairs = torch.cat([
            emb.unsqueeze(1).expand(-1, K, -1),
            emb.unsqueeze(0).expand(K, -1, -1),
        ], dim=-1)

        # (B, 1, 1, D) + (1, K, K, 2H) -> (B, K, K, 2H+D)
        deter_expanded = deter.view(B, 1, 1, D).expand(-1, K, K, -1)
        gate_input = torch.cat([emb_pairs.unsqueeze(0).expand(B, -1, -1, -1), deter_expanded], dim=-1)

        gate_logits = self.gate_net(gate_input).squeeze(-1)  # (B, K, K)

        if self.training:
            gates = binary_concrete(gate_logits, temperature=temperature, hard=hard)
        else:
            gates = (gate_logits > 0).float()

        # Self-loops always 1 (a factor is always its own parent)
        eye = torch.eye(K, device=deter.device).unsqueeze(0).expand(B, -1, -1)
        gates = gates * (1 - eye) + eye

        return gates, gate_logits

    def center_loss(self, gate_logits):
        """Penalize mean of off-diagonal gate logits deviating from 0.

        Without this, all logits drift negative → hard threshold zeros all gates.
        Centering at 0 ensures roughly half the off-diagonal edges survive.
        """
        K = self.num_factors
        mask = ~torch.eye(K, dtype=torch.bool, device=gate_logits.device)
        off_diag = gate_logits[:, mask]  # (B, K*(K-1))
        mean_logit = off_diag.mean()
        return mean_logit ** 2


class FactorPrior(nn.Module):
    """Factorized prior: p(z_t^i | h_t^i, parent_factors(z_{t-1}), a_{t-1}).

    Uses a shared transition network with factor-specific conditioning.

    Args:
        shared_ht: if True, use a single shared projection for all factors
                   (ablation: removes the h_t bottleneck)
    """

    def __init__(self, deter_dim, stoch_dim, discrete, num_factors, act_dim,
                 hidden=512, layers=2, shared_ht=False):
        super().__init__()
        assert stoch_dim % num_factors == 0, f"stoch_dim {stoch_dim} must be divisible by num_factors {num_factors}"
        assert deter_dim % num_factors == 0, f"deter_dim {deter_dim} must be divisible by num_factors {num_factors}"

        self.num_factors = num_factors
        self.stoch_per_factor = stoch_dim // num_factors  # stoch categories per factor
        self.discrete = discrete
        self.factor_size = self.stoch_per_factor * discrete  # flat size per factor
        self.shared_ht = shared_ht

        # Factor-specific h_t projections: h_t^i = W_i * stop_grad(h_t) + b_i
        self.h_factor_size = deter_dim // num_factors
        if shared_ht:
            # Single shared projection — removes the bottleneck
            self.h_projections = nn.ModuleList([
                nn.Linear(deter_dim, self.h_factor_size, bias=True)
            ])
        else:
            self.h_projections = nn.ModuleList([
                nn.Linear(deter_dim, self.h_factor_size, bias=True)
                for _ in range(num_factors)
            ])

        # Input: h_t^i + all parent z (flattened, masked) + action
        z_flat_total = stoch_dim * discrete  # total flat z size
        in_dim = self.h_factor_size + z_flat_total + act_dim

        layers_list = []
        for i in range(layers):
            layers_list.append(nn.Linear(in_dim if i == 0 else hidden, hidden, bias=True))
            layers_list.append(nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32))
            layers_list.append(nn.SiLU())
        self.shared_net = nn.Sequential(*layers_list)

        # Per-factor output heads: (hidden) -> (stoch_per_factor * discrete)
        self.factor_heads = nn.ModuleList([
            nn.Linear(hidden, self.factor_size, bias=True)
            for _ in range(num_factors)
        ])

        self.apply(weight_init_)

    def forward(self, deter, stoch_flat, action, gates):
        """Compute factorized prior distributions.

        Args:
            deter: (B, D) deterministic state (h_t) — already detached
            stoch_flat: (B, S*K) flattened stochastic state (z_{t-1})
            action: (B, A) previous action
            gates: (B, K, K) parent gate matrix

        Returns:
            logits: (B, S, K) logits for the factorized prior
        """
        B = deter.shape[0]
        K = self.num_factors

        # Factor-specific h_t projections
        if self.shared_ht:
            shared_h = self.h_projections[0](deter)
            h_factors = [shared_h] * self.num_factors  # K x (B, H/K) — all identical
        else:
            h_factors = [proj(deter) for proj in self.h_projections]  # K x (B, H/K)

        # Reshape stoch into per-factor groups: (B, S*K_discrete) -> (B, K, factor_size)
        stoch_per_factor = stoch_flat.view(B, K, self.factor_size)

        all_logits = []
        for i in range(K):
            # Parent gates for child i: (B, K)
            parent_gates = gates[:, :, i]
            # Weighted parent stoch: (B, K, factor_size)
            weighted_stoch = stoch_per_factor * parent_gates.unsqueeze(-1)
            # (B, K * factor_size) = (B, S*K_discrete)
            parent_input = weighted_stoch.view(B, -1)

            factor_input = torch.cat([h_factors[i], parent_input, action], dim=-1)
            hidden = self.shared_net(factor_input)
            logits_i = self.factor_heads[i](hidden)          # (B, stoch_per_factor * discrete)
            logits_i = logits_i.view(B, self.stoch_per_factor, self.discrete)
            all_logits.append(logits_i)

        # (B, S, discrete) where S = num_factors * stoch_per_factor = stoch_dim
        logits = torch.cat(all_logits, dim=1)
        return logits


class SparseFactorizedRSSM(nn.Module):
    """RSSM with sparse factorized prior."""

    def __init__(self, config, embed_size, act_dim):
        super().__init__()
        self._stoch = int(config.stoch)
        self._deter = int(config.deter)
        self._hidden = int(config.hidden)
        self._discrete = int(config.discrete)
        self._unimix_ratio = float(config.unimix_ratio)
        self._device = torch.device(config.device)
        self._act_dim = act_dim
        self.flat_stoch = self._stoch * self._discrete
        self.feat_size = self.flat_stoch + self._deter

        # Factor config
        self.num_factors = int(getattr(config, 'sf_num_factors', 8))
        self.sparse_lambda = float(getattr(config, 'sf_sparse_lambda', 0.01))
        self.gate_center_lambda = float(getattr(config, 'sf_gate_center_lambda', 0.0))
        self.gate_temperature = float(getattr(config, 'sf_gate_temperature', 0.5))

        # Training schedule
        self.sf_start_step = int(getattr(config, 'sf_start_step', 5000))

        # Deter transition (unchanged)
        import rssm
        self._deter_net = rssm.Deter(
            self._deter, self.flat_stoch, act_dim,
            self._hidden,
            blocks=int(config.blocks),
            dynlayers=int(config.dyn_layers),
            act=str(config.act),
        )

        # Posterior net (unchanged from standard RSSM)
        act_fn = getattr(torch.nn, str(config.act))
        self._obs_net = nn.Sequential()
        inp_dim = self._deter + embed_size
        for i in range(int(config.obs_layers)):
            self._obs_net.add_module(f"obs_net_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
            self._obs_net.add_module(f"obs_net_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
            self._obs_net.add_module(f"obs_net_a_{i}", act_fn())
            inp_dim = self._hidden
        self._obs_net.add_module("obs_net_logit", nn.Linear(inp_dim, self._stoch * self._discrete, bias=True))

        # Factorized prior
        gate_hidden = int(getattr(config, 'sf_gate_hidden', 64))
        shared_ht = bool(getattr(config, 'sf_shared_ht', False))
        self._factor_gate = FactorGate(self.num_factors, self._deter, hidden=gate_hidden)
        self._factor_prior = FactorPrior(
            self._deter, self._stoch, self._discrete,
            self.num_factors, act_dim,
            hidden=self._hidden, layers=2,
            shared_ht=shared_ht,
        )

        self.apply(weight_init_)

    def initial(self, batch_size):
        deter = torch.zeros(batch_size, self._deter, dtype=torch.float32, device=self._device)
        stoch = torch.zeros(batch_size, self._stoch, self._discrete, dtype=torch.float32, device=self._device)
        return stoch, deter

    def observe(self, embed, action, initial, reset):
        """Posterior rollout."""
        L = action.shape[1]
        stoch, deter = initial
        stochs, deters, logits = [], [], []
        for i in range(L):
            stoch, deter, logit = self.obs_step(stoch, deter, action[:, i], embed[:, i], reset[:, i])
            stochs.append(stoch)
            deters.append(deter)
            logits.append(logit)
        stochs = torch.stack(stochs, dim=1)
        deters = torch.stack(deters, dim=1)
        logits = torch.stack(logits, dim=1)
        return stochs, deters, logits

    def obs_step(self, stoch, deter, prev_action, embed, reset):
        """Single posterior step."""
        from tools import rpad
        stoch = torch.where(rpad(reset, stoch.dim() - int(reset.dim())), torch.zeros_like(stoch), stoch)
        deter = torch.where(rpad(reset, deter.dim() - int(reset.dim())), torch.zeros_like(deter), deter)
        prev_action = torch.where(
            rpad(reset, prev_action.dim() - int(reset.dim())), torch.zeros_like(prev_action), prev_action
        )
        deter = self._deter_net(stoch, deter, prev_action)
        x = torch.cat([deter, embed], dim=-1)
        logit = self._obs_net(x)
        # Reshape: (B, stoch*discrete) -> (B, stoch, discrete)
        logit = logit.reshape(*x.shape[:-1], self._stoch, self._discrete)
        stoch = self.get_dist(logit).rsample()
        return stoch, deter, logit

    def img_step(self, stoch, deter, prev_action):
        """Single prior step with factorized prior."""
        deter = self._deter_net(stoch, deter, prev_action)
        stoch, _, _, _ = self.prior(deter, stoch, prev_action)
        return stoch, deter

    def imagine_with_action(self, stoch, deter, actions):
        """Roll out prior dynamics given a sequence of actions."""
        L = actions.shape[1]
        stochs, deters = [], []
        for i in range(L):
            stoch, deter = self.img_step(stoch, deter, actions[:, i])
            stochs.append(stoch)
            deters.append(deter)
        stochs = torch.stack(stochs, dim=1)
        deters = torch.stack(deters, dim=1)
        return stochs, deters

    def prior(self, deter, prev_stoch=None, prev_action=None):
        """Factorized prior using sparse parent gates.

        Args:
            deter: (B, D) current deterministic state
            prev_stoch: (B, S, discrete) previous stochastic state
            prev_action: (B, A) previous action

        Returns:
            stoch, logit, gates, gate_logits
        """
        # Compute gates with DETACHED deter (prevent prior-loss leakage through gate path)
        gates, gate_logits = self._factor_gate(deter.detach(), temperature=self.gate_temperature)
        stoch_flat = prev_stoch.reshape(*prev_stoch.shape[:-2], self.flat_stoch)
        # Factorized prior also uses detached deter
        logit = self._factor_prior(deter.detach(), stoch_flat, prev_action, gates)
        stoch = self.get_dist(logit).rsample()
        return stoch, logit, gates, gate_logits

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
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        return dyn_loss, rep_loss

    def sparsity_loss(self, gates):
        """L0 sparsity loss on parent gates (scaled by lambda)."""
        K = self.num_factors
        mask = ~torch.eye(K, dtype=torch.bool, device=gates.device)
        mean_gate = gates[:, mask].mean()
        return self.sparse_lambda * mean_gate

    def gate_center_loss(self, gate_logits):
        """Center loss: penalize mean off-diagonal logit deviating from 0.

        Prevents gate logits from all drifting negative (→ diagonal-only collapse).
        λ_center controls the strength.
        """
        if self.gate_center_lambda <= 0:
            return torch.tensor(0.0, device=gate_logits.device)
        return self.gate_center_lambda * self._factor_gate.center_loss(gate_logits)

    def gate_bypass_diagnostic(self, deter, prev_stoch, prev_action, target_stoch, gates):
        """Diagnostic: how much do parent gates contribute to prior likelihood?

        Args:
            deter: (B, D) current deter
            prev_stoch: (B, S, discrete) previous stoch for parent input
            prev_action: (B, A)
            target_stoch: (B, S, discrete) the *target* z_t to evaluate likelihood on
            gates: (B, K, K) learned gates

        Returns:
            dict with delta_nll (higher = gates matter more) and log_likelihood_ratio.
        """
        prev_flat = prev_stoch.reshape(prev_stoch.shape[0], self.flat_stoch)

        full_logit = self._factor_prior(deter.detach(), prev_flat, prev_action, gates)
        full_logp = self.get_dist(full_logit).log_prob(target_stoch).mean()

        zero_gates = torch.zeros_like(gates)
        zeroed_logit = self._factor_prior(deter.detach(), prev_flat, prev_action, zero_gates)
        zeroed_logp = self.get_dist(zeroed_logit).log_prob(target_stoch).mean()

        return {
            "delta_nll": (-zeroed_logp - -full_logp).item(),
            "log_likelihood_ratio": (full_logp - zeroed_logp).item(),
        }
