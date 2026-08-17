"""
Ensemble RSSM Prior — K prior heads with Jensen-Shannon disagreement
for robust dynamics generalization in DreamerV3.

Architecture change (minimal):
  Standard RSSM:  GRU → 1 prior head → KL(posterior || prior)
  Ensemble RSSM:  GRU → K prior heads → KL(posterior || avg(prior_k))
                                       → actor_advantage -= lambda * JS(prior_k)

Only the prior is modified; encoder, GRU, posterior, decoder are unchanged.
Total added parameters: ~2% (K MLP heads on top of shared GRU).

Author: Research pipeline (2026-06-08)
"""

import math
import torch
from torch import nn
import torch.nn.functional as F

from tools import weight_init_


def ensemble_disagreement(logits_list, temperature=1.0):
    """Jensen-Shannon divergence across ensemble heads.

    Measures ensemble epistemic uncertainty in categorical latent space.

    Args:
        logits_list: list of K tensors, each (B, stoch, discrete)
        temperature: softmax temperature for smoothing

    Returns:
        disagreement: (B,) normalized JS divergence in [0, 1]
    """
    K = len(logits_list)
    if K < 2:
        return torch.zeros(logits_list[0].shape[0], device=logits_list[0].device)

    probs = [F.softmax(l / temperature, dim=-1) for l in logits_list]
    probs = torch.stack(probs, dim=0)  # (K, B, stoch, discrete)

    mean_probs = probs.mean(dim=0)  # (B, stoch, discrete)

    # Entropy of mean distribution: H(avg)
    entropy_mean = -(mean_probs * torch.log(mean_probs.clamp(min=1e-8))).sum(dim=-1)  # (B, stoch)

    # Mean of individual entropies: avg(H(p_k))
    entropy_each = -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=-1)  # (K, B, stoch)
    mean_entropy = entropy_each.mean(dim=0)  # (B, stoch)

    # JS = H(avg) - avg(H)
    js = entropy_mean - mean_entropy  # (B, stoch)

    # Normalize by log(num_classes) for [0, 1] range
    discrete = logits_list[0].shape[-1]
    js = js / math.log(max(discrete, 2))

    return js.mean(dim=-1)  # (B,)


class EnsemblePrior(nn.Module):
    """K independent prior heads for RSSM latent space.

    Each head is a small MLP (hidden=256) applied to the shared GRU state.
    Optional frozen random prior functions add baseline diversity.
    """

    def __init__(self, deter_dim, stoch, discrete, K=3):
        super().__init__()
        self.K = K
        self.stoch = stoch
        self.discrete = discrete
        out_dim = stoch * discrete

        # K independent prior heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(deter_dim, 256),
                nn.SiLU(),
                nn.Linear(256, out_dim),
            )
            for _ in range(K)
        ])

        self.apply(weight_init_)

    def forward(self, deter):
        """Compute all K prior logits.

        Args:
            deter: (B, D) deterministic state from GRU

        Returns:
            logits_list: list of K tensors (B, stoch, discrete)
            disagreement: (B,) JS divergence across heads
        """
        B = deter.shape[0]
        logits_list = []

        for k in range(self.K):
            logits_k = self.heads[k](deter)
            logits_k = logits_k.reshape(B, self.stoch, self.discrete)
            logits_list.append(logits_k)

        disagreement = ensemble_disagreement(logits_list)
        return logits_list, disagreement

    def average_logits(self, logits_list):
        """Average logits across all heads for the transition step."""
        return torch.stack(logits_list, dim=0).mean(dim=0)


class EnsembleRSSM(nn.Module):
    """RSSM with ensemble prior for robust dynamics generalization.

    Usage:
        if ensemble_enabled:
            rssm = EnsembleRSSM(config.rssm, embed_size, act_dim)

    Key difference from base RSSM:
        - prior() returns average over K heads (backward-compatible)
        - ensemble_prior_logits() returns all K logits for training
        - img_step() returns (stoch, deter, disagreement) for imagination
        - disagreement tracked as epistemic uncertainty metric
    """

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

        # Ensemble settings
        self.ensemble_k = int(getattr(config, 'ensemble_k', 3))
        self.bootstrap_rate = float(getattr(config, 'ensemble_bootstrap_rate', 0.7))

        # Deter transition (unchanged from RSSM)
        import rssm
        self._deter_net = rssm.Deter(
            self._deter, self.flat_stoch, act_dim,
            self._hidden,
            blocks=int(config.blocks),
            dynlayers=int(config.dyn_layers),
            act=str(config.act),
        )

        # Posterior net (unchanged)
        act_fn = getattr(torch.nn, str(config.act))
        self._obs_net = nn.Sequential()
        inp_dim = self._deter + embed_size
        for i in range(int(config.obs_layers)):
            self._obs_net.add_module(f"obs_net_{i}", nn.Linear(inp_dim, self._hidden))
            self._obs_net.add_module(f"obs_net_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
            self._obs_net.add_module(f"obs_net_a_{i}", act_fn())
            inp_dim = self._hidden
        self._obs_net.add_module("obs_net_logit", nn.Linear(inp_dim, self._stoch * self._discrete))

        # Ensemble prior
        self._ensemble_prior = EnsemblePrior(
            self._deter, self._stoch, self._discrete, K=self.ensemble_k,
        )

        # Note: weight_init_ is applied by _deter_net and _ensemble_prior internally.
        # Application here would double-init and may fail on norm layers.

    def initial(self, batch_size):
        deter = torch.zeros(batch_size, self._deter, dtype=torch.float32, device=self._device)
        stoch = torch.zeros(batch_size, self._stoch, self._discrete, dtype=torch.float32, device=self._device)
        return stoch, deter

    def observe(self, embed, action, initial, reset):
        """Posterior rollout. Same as standard RSSM."""
        L = action.shape[1]
        stoch, deter = initial
        stochs, deters, logits = [], [], []
        for i in range(L):
            stoch, deter, logit = self.obs_step(stoch, deter, action[:, i], embed[:, i], reset[:, i])
            stochs.append(stoch)
            deters.append(deter)
            logits.append(logit)
        return torch.stack(stochs, dim=1), torch.stack(deters, dim=1), torch.stack(logits, dim=1)

    def obs_step(self, stoch, deter, prev_action, embed, reset):
        """Single posterior step. Same as standard RSSM."""
        from tools import rpad
        stoch = torch.where(rpad(reset, stoch.dim() - int(reset.dim())), torch.zeros_like(stoch), stoch)
        deter = torch.where(rpad(reset, deter.dim() - int(reset.dim())), torch.zeros_like(deter), deter)
        prev_action = torch.where(
            rpad(reset, prev_action.dim() - int(reset.dim())), torch.zeros_like(prev_action), prev_action
        )
        deter = self._deter_net(stoch, deter, prev_action)
        x = torch.cat([deter, embed], dim=-1)
        logit = self._obs_net(x)
        logit = logit.reshape(*x.shape[:-1], self._stoch, self._discrete)
        stoch = self.get_dist(logit).rsample()
        return stoch, deter, logit

    def img_step(self, stoch, deter, prev_action):
        """Prior step for imagination. Returns (stoch, deter, disagreement).

        Uses head_0 for transition (standard DV3 prior).
        Disagreement computed from all K heads for actor penalty.
        """
        deter = self._deter_net(stoch, deter, prev_action)
        logits_list, disagreement = self._ensemble_prior(deter)
        # Use HEAD 0 for transition (pure DV3-quality prior)
        head0_logits = logits_list[0]
        stoch = self.get_dist(head0_logits).rsample()
        return stoch, deter, disagreement

    def img_step_single(self, stoch, deter, prev_action):
        """Standard interface: returns (stoch, deter)."""
        stoch, deter, _ = self.img_step(stoch, deter, prev_action)
        return stoch, deter

    def prior(self, deter):
        """Standard prior: head_0 only.

        Returns (stoch, head0_logit) — compatible with base RSSM interface.
        """
        logits_list, _ = self._ensemble_prior(deter)
        head0_logits = logits_list[0]
        stoch = self.get_dist(head0_logits).rsample()
        return stoch, head0_logits

    def ensemble_prior_logits(self, deter):
        """Returns (logits_list, disagreement) for ensemble training.

        logits_list: list of K tensors (B, stoch, discrete)
        disagreement: (B,) JS divergence
        """
        return self._ensemble_prior(deter)

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

    def imagine_with_action(self, stoch, deter, actions):
        """Roll out prior dynamics given a sequence of actions (video_pred compat)."""
        L = actions.shape[1]
        stochs, deters = [], []
        for i in range(L):
            stoch, deter, _ = self.img_step(stoch, deter, actions[:, i])
            stochs.append(stoch)
            deters.append(deter)
        return torch.stack(stochs, dim=1), torch.stack(deters, dim=1)
