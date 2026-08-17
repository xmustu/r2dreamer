"""
Domain-aware replay buffer for DreamerV3 mixed-dynamics training.

Supports three sampling modes:
  - "mixed": Standard sampling from the full buffer (default = NoDistill)
  - "homogeneous": Each minibatch is >80% from ONE dynamics domain.
    Domain target alternates to match buffer composition (70/30).
  - "balanced_alternating": Strict alternation between 100%-nominal
    and 100%-perturbed batches.

Uses dynamics_mode stored in each transition (0.0 = nominal, 1.0 = perturbed)
to filter batches via rejection sampling.

CRITICAL: The underlying torchrl ReplayBuffer stores transitions with the
"episode" key for trajectory boundary tracking. All transitions within one
episode share the same dynamics_mode (set at episode reset). This means
rejection-sampled batches are naturally domain-homogeneous at the trajectory
level, not just the transition level.
"""

import torch
from torchrl.data.replay_buffers import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler


class DomainAwareBuffer:
    """Wraps the standard DreamerV3 Buffer with domain-aware post-sampling.

    Uses rejection sampling to enforce domain homogeneity. The 70/30 buffer
    composition is preserved at the BUFFER level; batch-level domain targets
    are chosen to reflect this composition.
    """

    def __init__(self, base_config, domain_config=None):
        """Initialize domain-aware buffer.

        Args:
            base_config: Top-level buffer config (has device, batch_size, etc.)
            domain_config: Model-level buffer config (has domain_sampling_mode, nominal_prob)
        """
        if domain_config is None:
            domain_config = base_config

        self.device = torch.device(base_config.device)
        self.storage_device = torch.device(base_config.storage_device)
        self.batch_size = int(base_config.batch_size)
        self.batch_length = int(base_config.batch_length)
        self.num_eps = 0

        # Domain sampling mode
        self._mode = str(getattr(domain_config, 'domain_sampling_mode', 'mixed'))
        self._nominal_prob = float(getattr(domain_config, 'nominal_prob', 0.7))
        # Internal counter for alternating modes
        self._sample_count = 0
        # Statistics
        self._nominal_batches = 0
        self._perturbed_batches = 0
        self._mixed_batches = 0
        self._resample_count = 0
        self._fallthrough_count = 0

        self._buffer = ReplayBuffer(
            storage=LazyTensorStorage(
                max_size=base_config.max_size, device=self.storage_device, ndim=2
            ),
            sampler=SliceSampler(
                num_slices=self.batch_size,
                end_key=None,
                traj_key="episode",
                truncated_key=None,
                strict_length=True,
            ),
            prefetch=0,
            batch_size=self.batch_size * (self.batch_length + 1),  # +1 for context
        )

    def add_transition(self, data):
        """Add a transition batch to the buffer."""
        self._buffer.extend(data.unsqueeze(1))

    def _get_domain_fraction(self, sample_td):
        """Return fraction of nominal transitions in the batch."""
        if "dynamics_mode" in sample_td.keys():
            dyn_mode = sample_td["dynamics_mode"]
            return (dyn_mode < 0.5).float().mean().item()
        return 1.0  # No dynamics_mode → assume all nominal

    def sample(self):
        """Sample a batch with domain-aware filtering.

        homogeneous: Target domain chosen to match buffer ratio (70/30).
            Each batch must be >80% from the target domain.
        balanced_alternating: Strict 50/50 alternation. Each batch must
            be >99% from the target domain (effectively 100% pure).
        mixed: No filtering (standard DreamerV3 behavior).
        """
        max_resamples = 20  # More attempts for rare perturbed batches
        target_nominal = None  # Which domain we're targeting this batch

        # Determine target domain BEFORE sampling
        if self._mode == "homogeneous":
            # Target matches buffer composition: 70% nominal, 30% perturbed
            target_nominal = (self._sample_count % 10) < (self._nominal_prob * 10)
        elif self._mode == "balanced_alternating":
            # Strict alternation
            target_nominal = (self._sample_count % 2 == 0)
        # else: mixed mode, no target

        # For balanced_alternating, require near-100% purity
        threshold = 0.99 if self._mode == "balanced_alternating" else 0.8

        for attempt in range(max_resamples):
            sample_td, info = self._buffer.sample(return_info=True)
            # (B*(T+1), ...) -> (B, T+1, ...)
            sample_td = sample_td.view(-1, self.batch_length + 1)
            src_dev = sample_td.device
            if src_dev.type == "cpu" and self.device.type == "cuda":
                sample_td = sample_td.pin_memory().to(self.device, non_blocking=True)
            elif src_dev != self.device:
                sample_td = sample_td.to(self.device, non_blocking=True)

            if self._mode == "mixed":
                break

            frac_nominal = self._get_domain_fraction(sample_td)

            if target_nominal:
                if frac_nominal >= threshold:
                    break
            else:
                if (1.0 - frac_nominal) >= threshold:
                    break
            self._resample_count += 1
        else:
            # Max resamples reached — use last sample, log fallthrough
            self._fallthrough_count += 1

        # Track statistics
        frac_nominal = self._get_domain_fraction(sample_td)
        if frac_nominal >= 0.5:
            self._nominal_batches += 1
        else:
            self._perturbed_batches += 1

        self._sample_count += 1

        # Prepare return values (matching standard Buffer.sample() format)
        initial = (sample_td["stoch"][:, 0], sample_td["deter"][:, 0])
        data = sample_td[:, 1:]
        data.set_("action", sample_td["action"][:, :-1])  # action is 1 step back
        index = [ind.view(-1, self.batch_length + 1)[:, 1:] for ind in info["index"]]
        return data, index, initial

    def update(self, index, stoch, deter):
        """Update latent states in buffer (same as standard Buffer)."""
        index = [ind.reshape(-1) for ind in index]
        stoch = stoch.reshape(-1, *stoch.shape[2:])
        deter = deter.reshape(-1, *deter.shape[2:])
        self._buffer[index[1], index[0]].set_("stoch", stoch)
        self._buffer[index[1], index[0]].set_("deter", deter)

    def count(self):
        if self._buffer.storage.shape is None:
            return 0
        return self._buffer.storage.shape.numel()

    def get_stats(self):
        """Return domain sampling statistics for logging."""
        total = self._nominal_batches + self._perturbed_batches
        return {
            "domain/nominal_batch_frac": self._nominal_batches / max(total, 1),
            "domain/perturbed_batch_frac": self._perturbed_batches / max(total, 1),
            "domain/resample_rate": self._resample_count / max(self._sample_count, 1),
            "domain/fallthrough_rate": self._fallthrough_count / max(self._sample_count, 1),
        }
