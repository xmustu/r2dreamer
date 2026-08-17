"""
PyTorch Gradient Monitor for Dreamer Multi-Objective Training Analysis.

Drop-in module for r2dreamer PyTorch codebase. Instruments the training loop
to log per-loss gradient norms, pairwise cosine similarities, and
conflict ratios for world-model loss terms.

Usage (patch into dreamer.py update() and _cal_grad()):
    monitor = GradientMonitor(
        named_params=agent._named_params,
        shared_param_keys=['rssm', 'encoder'],
        log_every=100,
        output_dir='gradient_logs',
    )

    # In _cal_grad, after computing all losses but before backward:
    per_loss_grads = monitor.compute_per_loss_grads(
        losses, scaled_losses, retain_graph=True
    )

    # Then in update(), before optimizer step:
    monitor.log_step(step, per_loss_grads)

This computes gradients on the SHARED TRUNK parameters (RSSM + encoder),
excluding head parameters (decoder, reward, continue, actor, critic).
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set
import json
import os


class GradientMonitor:
    """Logs per-loss gradient statistics for r2dreamer world-model training.

    Computes per-loss gradients on shared trunk parameters (RSSM + encoder),
    then logs gradient norms, pairwise cosine similarities, group-level
    conflict metrics, and parameter-group decomposition.

    Designed for sparse logging (every N steps) to keep overhead < 5%.
    """

    def __init__(
        self,
        named_params: Dict[str, nn.Parameter],
        loss_keys: List[str] = None,
        log_every: int = 100,
        output_dir: str = 'gradient_logs',
        window_size: int = 1000,
        device: str = 'cuda:0',
    ):
        self.named_params = named_params
        self.loss_keys = loss_keys or ['dyn', 'rep', 'rew', 'con']
        self.log_every = log_every
        self.output_dir = output_dir
        self.window_size = window_size
        self.device = device

        # Identify shared trunk vs head parameters by name
        self._trunk_param_names = self._identify_trunk_params()
        self._head_param_names = set(named_params.keys()) - self._trunk_param_names

        # Per-step storage (circular buffer for moving window)
        self._buffer: Dict[str, List[Dict]] = defaultdict(list)
        self.logs: List[Dict] = []

        os.makedirs(output_dir, exist_ok=True)

        print(f"[GradientMonitor] Trunk params: {len(self._trunk_param_names)} "
              f"(e.g. {list(self._trunk_param_names)[:5]})")
        print(f"[GradientMonitor] Head params: {len(self._head_param_names)} "
              f"(e.g. {list(self._head_param_names)[:5]})")

    TRUNK_PATTERNS = ['rssm', 'encoder', 'projector', 'prj', 'deter', 'stoch', 'prior', 'obs_']

    def _identify_trunk_params(self) -> Set[str]:
        """Identify which parameter names belong to the shared trunk.

        Trunk = RSSM + encoder + projector (shared representation).
        Head = decoder, reward, cont, actor, value, _slow_value, prototypes.
        """
        trunk = set()
        for name in self.named_params:
            name_lower = name.lower()
            is_head = any(h in name_lower for h in [
                'decoder', 'reward', 'cont', 'actor', 'critic',
                'value', 'slow_value', 'prototype', '_slow', 'return_ema'
            ])
            if not is_head:
                is_trunk = any(t in name_lower for t in self.TRUNK_PATTERNS)
                if is_trunk:
                    trunk.add(name)
        return trunk

    def get_trunk_params(self) -> List[nn.Parameter]:
        """Return ordered list of shared trunk parameters."""
        return [self.named_params[n] for n in sorted(self._trunk_param_names)]

    def compute_per_loss_grads(
        self,
        losses: Dict[str, torch.Tensor],
        loss_scales: Dict[str, float] = None,
        retain_graph: bool = True,
    ) -> Dict[str, Dict]:
        """Compute per-loss gradients on shared trunk parameters.

        IMPORTANT: Each loss term is scaled by its loss_scale BEFORE
        gradient computation, matching the actual gradient flow.

        Uses torch.autograd.grad with retain_graph=True for each loss.

        Args:
            losses: Dict of {loss_name: scalar_tensor}
            loss_scales: Dict of {loss_name: scale_factor} (defaults: DreamerV3)
            retain_graph: Keep computation graph for subsequent grads

        Returns:
            Dict of {loss_name: {'grads': List[Tensor], 'norm': float}}
        """
        if loss_scales is None:
            loss_scales = {
                'dyn': 1.0, 'rep': 0.1, 'rew': 1.0, 'con': 1.0,
                'barlow': 0.05, 'recon': 1.0, 'policy': 1.0,
                'value': 1.0, 'repval': 0.3, 'sf_sparse': 1.0,
            }

        trunk_params = self.get_trunk_params()
        grads_dict = {}

        for key in self.loss_keys:
            if key not in losses:
                continue

            scale = loss_scales.get(key, 1.0)
            scaled_loss = scale * losses[key]

            # Compute gradient of THIS loss on trunk params
            grads = torch.autograd.grad(
                scaled_loss, trunk_params,
                retain_graph=retain_graph,
                allow_unused=True,
            )

            # Compute gradient norm (L2 over all trunk params)
            grad_norm_sq = 0.0
            valid_grads = []
            for g, p in zip(grads, trunk_params):
                if g is not None:
                    grad_norm_sq += g.data.norm().item() ** 2
                    valid_grads.append(g.data.clone())
                else:
                    valid_grads.append(torch.zeros_like(p.data))

            grad_norm = np.sqrt(grad_norm_sq)

            grads_dict[key] = {
                'grads': valid_grads,
                'norm': float(grad_norm),
            }

        return grads_dict

    @staticmethod
    def cosine_similarity(grads_a: List[torch.Tensor], grads_b: List[torch.Tensor]) -> float:
        """Compute cosine similarity between two flattened gradient vectors.

        cos = ⟨∇L_a, ∇L_b⟩ / (||∇L_a|| · ||∇L_b||)
        """
        flat_a = torch.cat([g.reshape(-1).float() for g in grads_a])
        flat_b = torch.cat([g.reshape(-1).float() for g in grads_b])

        dot = torch.dot(flat_a, flat_b).item()
        norm_a = flat_a.norm().item()
        norm_b = flat_b.norm().item()

        cos = dot / (norm_a * norm_b + 1e-8)
        return float(cos)

    def compute_pairwise_cosines(self, grads_dict: Dict) -> Dict[str, float]:
        """Compute pairwise cosine similarities for all loss pairs."""
        cosines = {}
        keys = list(grads_dict.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                pair_key = f"{keys[i]}_{keys[j]}"
                cosines[pair_key] = self.cosine_similarity(
                    grads_dict[keys[i]]['grads'],
                    grads_dict[keys[j]]['grads']
                )
        return cosines

    def compute_group_conflict(self, grads_dict: Dict) -> Dict:
        """Compute group-level conflict: KL-group vs Prediction-group.

        KL group: dyn + rep (KL losses)
        Prediction group: rew + con (+ barlow/infonce/recon when present)
        """
        kl_keys = [k for k in ['dyn', 'rep'] if k in grads_dict]
        pred_keys = [k for k in ['rew', 'con', 'barlow', 'infonce', 'recon'] if k in grads_dict]

        if not kl_keys or not pred_keys:
            return {}

        # Sum gradient vectors within groups
        kl_grads = self._sum_grad_lists([grads_dict[k]['grads'] for k in kl_keys])
        pred_grads = self._sum_grad_lists([grads_dict[k]['grads'] for k in pred_keys])

        group_cos = self.cosine_similarity(kl_grads, pred_grads)

        # Compute norms
        kl_norm = np.sqrt(sum(g.norm().item()**2 for g in kl_grads))
        pred_norm = np.sqrt(sum(g.norm().item()**2 for g in pred_grads))

        return {
            'kl_group_norm': float(kl_norm),
            'pred_group_norm': float(pred_norm),
            'group_cosine': group_cos,
            'kl_group_conflict': float(group_cos < 0),
        }

    @staticmethod
    def _sum_grad_lists(grad_lists: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """Sum multiple gradient lists element-wise."""
        if not grad_lists:
            return []
        result = [torch.zeros_like(g) for g in grad_lists[0]]
        for grads in grad_lists:
            for i, g in enumerate(grads):
                result[i] += g
        return result

    def log_step(self, step: int, grads_dict: Dict):
        """Log gradient statistics for one training step."""
        if step % self.log_every != 0:
            return

        norms = {k: v['norm'] for k, v in grads_dict.items()}
        pairwise_cos = self.compute_pairwise_cosines(grads_dict)
        group_metrics = self.compute_group_conflict(grads_dict)

        conflict_count = sum(1 for v in pairwise_cos.values() if v < 0)
        total_pairs = max(len(pairwise_cos), 1)
        conflict_ratio = conflict_count / total_pairs

        entry = {
            'step': int(step),
            'norms': {k: float(v) for k, v in norms.items()},
            'pairwise_cosines': {k: float(v) for k, v in pairwise_cos.items()},
            'conflict_ratio': conflict_ratio,
            **{k: float(v) if isinstance(v, (float, np.floating)) else v
               for k, v in group_metrics.items()},
        }

        self._buffer['steps'].append(entry)
        if len(self._buffer['steps']) > self.window_size:
            self._buffer['steps'].pop(0)

        self.logs.append(entry)

    def get_moving_statistics(self) -> Dict:
        """Compute moving-window aggregate statistics."""
        if not self._buffer['steps']:
            return {}

        recent = self._buffer['steps'][-self.window_size:]

        avg_norms = defaultdict(list)
        for entry in recent:
            for k, v in entry['norms'].items():
                avg_norms[k].append(v)
        avg_norms = {k: float(np.mean(v)) for k, v in avg_norms.items()}

        avg_cosines = defaultdict(list)
        for entry in recent:
            for k, v in entry['pairwise_cosines'].items():
                avg_cosines[k].append(v)
        avg_cosines = {k: float(np.mean(v)) for k, v in avg_cosines.items()}

        avg_conflict = float(np.mean([e['conflict_ratio'] for e in recent]))
        avg_group_cos = float(np.mean([
            e.get('group_cosine', 0) for e in recent
        ]))

        return {
            'window_size': len(recent),
            'avg_norms': avg_norms,
            'avg_cosines': avg_cosines,
            'avg_conflict_ratio': avg_conflict,
            'avg_group_cosine': avg_group_cos,
        }

    def save_logs(self, filename: str = 'gradient_logs.json'):
        """Save all logged gradient statistics to disk."""
        filepath = os.path.join(self.output_dir, filename)
        stats = self.get_moving_statistics()
        output = {
            'config': {
                'loss_keys': self.loss_keys,
                'log_every': self.log_every,
                'window_size': self.window_size,
                'n_trunk_params': len(self._trunk_param_names),
                'n_head_params': len(self._head_param_names),
            },
            'final_statistics': stats,
            'n_logged_steps': len(self.logs),
            'logs': self.logs,
        }
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        return filepath

    def generate_conflict_heatmap(self) -> Dict:
        """Generate data for (time x loss-pair) conflict heatmap."""
        if not self.logs:
            return {}
        times = [entry['step'] for entry in self.logs]
        pairs = list(self.logs[0]['pairwise_cosines'].keys())
        matrix = np.zeros((len(times), len(pairs)))
        for i, entry in enumerate(self.logs):
            for j, pair in enumerate(pairs):
                matrix[i, j] = entry['pairwise_cosines'].get(pair, 0)
        return {
            'times': times,
            'pairs': pairs,
            'cosine_matrix': matrix.tolist(),
        }


def phase_analysis(gradient_logs: List[Dict]) -> Dict:
    """Offline phase analysis using change-point detection.

    Args:
        gradient_logs: List of per-step gradient log entries

    Returns:
        Dict with phase boundaries, per-phase statistics
    """
    conflict_ratios = np.array([e['conflict_ratio'] for e in gradient_logs])
    steps = np.array([e['step'] for e in gradient_logs])
    group_cosines = np.array([
        e.get('group_cosine', 0) for e in gradient_logs
    ])

    _pelt_available = False
    try:
        import ruptures as rpt
        _pelt_available = True
        signal = conflict_ratios.reshape(-1, 1)
        algo = rpt.Pelt(model="rbf", min_size=max(5, len(signal) // 20))
        change_points = algo.fit_predict(signal)
        phase_boundaries = [steps[cp - 1] for cp in change_points[:-1]]
    except ImportError:
        n = len(steps)
        phase_boundaries = [steps[n // 3], steps[2 * n // 3]]

    phases = []
    boundaries = [0] + list(phase_boundaries) + [steps[-1] + 1]
    for i in range(len(boundaries) - 1):
        mask = (steps >= boundaries[i]) & (steps < boundaries[i + 1])
        if mask.sum() == 0:
            continue
        phases.append({
            'phase_id': i,
            'start_step': int(boundaries[i]),
            'end_step': int(boundaries[i + 1] - 1),
            'n_steps': int(mask.sum()),
            'mean_conflict_ratio': float(conflict_ratios[mask].mean()),
            'std_conflict_ratio': float(conflict_ratios[mask].std()),
            'mean_group_cosine': float(group_cosines[mask].mean()),
        })

    return {
        'n_phases': len(phases),
        'phase_boundaries': [int(b) for b in phase_boundaries],
        'phases': phases,
        'detection_method': 'PELT' if _pelt_available else 'tertile_fallback',
    }
