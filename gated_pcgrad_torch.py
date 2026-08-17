"""
PyTorch Group-Level Conflict-Gated PCGrad for Dreamer World-Model Training.

Drop-in for r2dreamer. Applies PCGrad projection only when group-level
cosine similarity < tau, preventing unnecessary surgery.

Usage (in dreamer.py _cal_grad):
    from gated_pcgrad_torch import gated_group_pcgrad, pairwise_pcgrad

    per_loss_grads = {}  # {loss_name: List[Tensor]} — grads w.r.t. ALL params
    # ... compute per_loss_grads ...

    intervention = gated_group_pcgrad(per_loss_grads, tau=0.0)
    # intervention is List[Tensor] — combined gradient for optimizer
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


def _cosine(grads_a: List[torch.Tensor], grads_b: List[torch.Tensor]) -> float:
    flat_a = torch.cat([g.detach().reshape(-1).float() for g in grads_a])
    flat_b = torch.cat([g.detach().reshape(-1).float() for g in grads_b])
    dot = torch.dot(flat_a, flat_b).item()
    na = flat_a.norm().item()
    nb = flat_b.norm().item()
    return dot / (na * nb + 1e-8)


def _project(grads_a: List[torch.Tensor], grads_b: List[torch.Tensor]) -> List[torch.Tensor]:
    """Project grads_a onto normal plane of grads_b."""
    flat_a = torch.cat([g.detach().reshape(-1).float() for g in grads_a])
    flat_b = torch.cat([g.detach().reshape(-1).float() for g in grads_b])
    dot = torch.dot(flat_a, flat_b)
    norm_b_sq = flat_b.dot(flat_b) + 1e-8
    scale = (dot / norm_b_sq).item()
    return [ga - scale * gb for ga, gb in zip(grads_a, grads_b)]


def sum_grads(grads_dict: Dict[str, List[torch.Tensor]]) -> List[torch.Tensor]:
    """Sum parameter-wise gradients across all losses."""
    all_grads = list(grads_dict.values())
    if not all_grads:
        return []
    result = [torch.zeros_like(g) for g in all_grads[0]]
    for grads in all_grads:
        for i, g in enumerate(grads):
            result[i] = result[i] + g
    return result


def gated_group_pcgrad(
    grads_dict: Dict[str, List[torch.Tensor]],
    tau: float = 0.0,
    kl_keys: Tuple[str, ...] = ('dyn', 'rep'),
    pred_keys: Tuple[str, ...] = ('rew', 'con'),
    symmetric: bool = True,
) -> List[torch.Tensor]:
    """Group-level conflict-gated PCGrad.

    Groups losses into KL group (dyn+rep) and Prediction group (rew+con+others).
    Projects only when group-level cosine similarity < tau.

    Args:
        grads_dict: {loss_name: [param_grad_tensors]} — gradients w.r.t. SAME parameter list
        tau: Conflict threshold (default 0: project when opposing directions)
        kl_keys: Loss keys in KL group
        pred_keys: Loss keys in prediction group
        symmetric: If True, project both groups onto each other's normal planes

    Returns:
        Combined gradient list (same length as input grad lists)
    """
    available_kl = [k for k in kl_keys if k in grads_dict]
    available_pred = [k for k in pred_keys if k in grads_dict]

    if not available_kl or not available_pred:
        return sum_grads(grads_dict)

    # Sum gradients within each group
    grad_kl = sum_grads({k: grads_dict[k] for k in available_kl})
    grad_pred = sum_grads({k: grads_dict[k] for k in available_pred})

    # Check group-level conflict
    group_cos = _cosine(grad_kl, grad_pred)

    if group_cos < tau:
        # Apply surgery
        if symmetric:
            kl_proj = _project(grad_kl, grad_pred)
            pred_proj = _project(grad_pred, grad_kl)
            return [a + b for a, b in zip(kl_proj, pred_proj)]
        else:
            pred_proj = _project(grad_pred, grad_kl)
            return [a + b for a, b in zip(grad_kl, pred_proj)]
    else:
        # No conflict — sum normally
        return [a + b for a, b in zip(grad_kl, grad_pred)]


def pairwise_pcgrad(
    grads_dict: Dict[str, List[torch.Tensor]],
    loss_order: Optional[List[str]] = None,
) -> List[torch.Tensor]:
    """Full pairwise PCGrad — baseline comparison.

    Projects each gradient onto normal planes of ALL conflicting previous gradients.
    O(K^2) complexity.
    """
    if loss_order is None:
        loss_order = list(grads_dict.keys())

    grads_list = [grads_dict[k] for k in loss_order if k in grads_dict]
    if len(grads_list) <= 1:
        return grads_list[0] if grads_list else []

    projected = [grads_list[0]]
    for i in range(1, len(grads_list)):
        grad_i = [g.clone() for g in grads_list[i]]
        for j in range(i):
            cos = _cosine(grad_i, projected[j])
            if cos < 0:
                grad_i = _project(grad_i, projected[j])
        projected.append(grad_i)

    return sum_grads({f"_{i}": g for i, g in enumerate(projected)})


def apply_intervention(
    grads_dict: Dict[str, List[torch.Tensor]],
    params: List[nn.Parameter],
    mode: str = 'none',
    tau: float = 0.0,
) -> None:
    """Apply gradient intervention by setting .grad on parameters.

    This is the main entry point for patching into dreamer.py update():
    Computes per-loss grads, applies intervention, and sets param.grad.

    Args:
        grads_dict: {loss_name: [per-param gradients]}
        params: Ordered parameter list matching grad tensor order
        mode: 'none' | 'pcgrad' | 'gated'
        tau: Conflict threshold for 'gated' mode
    """
    if mode == 'none':
        combined = sum_grads(grads_dict)
    elif mode == 'pcgrad':
        combined = pairwise_pcgrad(grads_dict)
    elif mode == 'gated':
        combined = gated_group_pcgrad(grads_dict, tau=tau)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Set .grad on each parameter
    for param, grad in zip(params, combined):
        if param.grad is not None:
            param.grad.copy_(grad)
        else:
            param.grad = grad.clone()
