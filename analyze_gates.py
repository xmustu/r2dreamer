#!/usr/bin/env python3
"""
Gate Structure Analysis (B4) — Interpretability of Learned Parent Gates.

Evaluates Claim C2: "Learned parent gates reveal interpretable kinematic structure."

Computes:
  1. Average gate matrix G (K×K) from rollouts
  2. Edge sparsity pattern
  3. AUROC vs ground-truth kinematic DAG (for cheetah, walker)
  4. Gate-bypass diagnostic: ΔNLL between full-gate and zero-gate prior

Usage:
  python analyze_gates.py <checkpoint.pt> <config_dir> <device> <output.json>
      [--task dmc_walker_walk] [--rollout-steps 500]
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from dreamer import Dreamer
from envs.dmc import DeepMindControl
from envs import wrappers


# Ground-truth kinematic adjacency for known DMC bodies
# For walker: torso(0)→thigh(1)→leg(2)→foot(3), plus the other leg
# For cheetah: torso(0)→thighs(1,4)→shins(2,5)→feet(3,6)  [simplified]
# We map K=8 factors to these body parts heuristically
KINEMATIC_DAG = {
    "walker_walk": np.array([
        # torso=0, r_thigh=1, r_leg=2, r_foot=3, l_thigh=4, l_leg=5, l_foot=6, extra=7
        [1, 0, 0, 0, 0, 0, 0, 0],  # torso → self
        [1, 1, 0, 0, 0, 0, 0, 0],  # thigh → torso+self
        [0, 1, 1, 0, 0, 0, 0, 0],  # leg → thigh+self
        [0, 0, 1, 1, 0, 0, 0, 0],  # foot → leg+self
        [1, 0, 0, 0, 1, 0, 0, 0],  # l_thigh → torso+self
        [0, 0, 0, 0, 1, 1, 0, 0],  # l_leg → l_thigh+self
        [0, 0, 0, 0, 0, 1, 1, 0],  # l_foot → l_leg+self
        [1, 0, 0, 0, 0, 0, 0, 1],  # extra → torso+self
    ]),
    "cheetah_run": np.array([
        # torso=0, f_thigh=1, f_shin=2, f_foot=3, b_thigh=4, b_shin=5, b_foot=6, head=7
        [1, 0, 0, 0, 0, 0, 0, 0],  # torso → self
        [1, 1, 0, 0, 0, 0, 0, 0],  # f_thigh → torso+self
        [0, 1, 1, 0, 0, 0, 0, 0],  # f_shin → f_thigh+self
        [0, 0, 1, 1, 0, 0, 0, 0],  # f_foot → f_shin+self
        [1, 0, 0, 0, 1, 0, 0, 0],  # b_thigh → torso+self
        [0, 0, 0, 0, 1, 1, 0, 0],  # b_shin → b_thigh+self
        [0, 0, 0, 0, 0, 1, 1, 0],  # b_foot → b_shin+self
        [1, 0, 0, 0, 0, 0, 0, 1],  # head → torso+self
    ]),
}


def make_eval_env(task_name, seed=42, image_size=64):
    env = DeepMindControl(task_name, action_repeat=2, size=(image_size, image_size), seed=seed)
    return env


def load_agent(checkpoint_path, config_path, device="cuda:0"):
    import omegaconf
    import gymnasium as gym

    config = omegaconf.OmegaConf.load(config_path)

    # Infer action dimension from checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["agent_state_dict"]
    actor_out = state_dict["actor.last.weight"].shape[0]
    act_dist = getattr(config.model.actor.dist, "cont", None)
    if act_dist is not None and getattr(act_dist, "name", "") == "bounded_normal":
        act_dim = actor_out // 2
    else:
        act_dim = actor_out

    obs_space = gym.spaces.Dict({
        "image": gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
        "is_first": gym.spaces.Box(0, 1, (1,), dtype=np.float32),
        "is_last": gym.spaces.Box(0, 1, (1,), dtype=np.float32),
        "is_terminal": gym.spaces.Box(0, 1, (1,), dtype=np.float32),
    })
    act_space = gym.spaces.Box(-1, 1, (act_dim,), dtype=np.float32)

    # Disable compile for analysis — only use agent.act()
    if hasattr(config.model, "compile"):
        config.model.compile = False

    agent = Dreamer(config.model, obs_space, act_space).to(torch.device(device))
    agent.load_state_dict(state_dict)
    agent.eval()
    agent.requires_grad_(False)
    return agent


def compute_auroc(pred_matrix, gt_matrix):
    """Compute AUROC for edge detection (excluding diagonal/self-loops)."""
    K = pred_matrix.shape[0]
    mask = ~np.eye(K, dtype=bool)
    pred_flat = pred_matrix[mask].flatten()
    gt_flat = gt_matrix[mask].flatten()

    # Sort by prediction
    order = np.argsort(pred_flat)[::-1]
    gt_sorted = gt_flat[order]

    # Compute TPR/FPR
    n_pos = gt_sorted.sum()
    n_neg = len(gt_sorted) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    tpr = np.cumsum(gt_sorted) / n_pos
    fpr = np.cumsum(1 - gt_sorted) / n_neg

    # AUROC via trapezoidal rule
    return float(np.trapz(tpr, fpr))


@torch.no_grad()
def collect_gate_matrices(agent, env, rollout_steps=500, device="cuda:0"):
    """Collect gate matrices over multiple environment interactions."""
    obs = env.reset()
    state = agent.get_initial_state(1)

    all_gates = []
    all_hard_gates = []

    for step in range(rollout_steps):
        # Prepare obs
        image = obs["image"].copy()
        is_first_val = obs.get("is_first", step == 0)
        if isinstance(is_first_val, (bool, np.bool_)):
            is_first_flag = bool(is_first_val)
        else:
            is_first_flag = bool(is_first_val)

        obs_tensor = {
            "image": torch.from_numpy(image).unsqueeze(0).to(device),
            "is_first": torch.tensor([[is_first_flag]], dtype=torch.bool, device=device),
            "is_last": torch.tensor([[False]], dtype=torch.bool, device=device),
            "is_terminal": torch.tensor([[False]], dtype=torch.bool, device=device),
        }

        # Get action
        action, state = agent.act(obs_tensor, state, eval=True)
        action_np = action.squeeze(0).cpu().numpy()

        # Collect gates from RSSM internals
        prev_stoch = state["stoch"]
        prev_deter = state["deter"]
        prev_action = state["prev_action"]

        with torch.no_grad():
            # Manually compute gates for this step
            gates, gate_logits = agent.rssm._factor_gate(
                prev_deter,
                temperature=agent.rssm.gate_temperature,
            )
            hard_gates = (gate_logits > 0).float()
            # Fix self-loops (mirroring FactorGate.forward)
            K = gates.shape[-1]
            eye = torch.eye(K, device=gates.device).unsqueeze(0)
            hard_gates = hard_gates * (1 - eye) + eye

        all_gates.append(gates.squeeze(0).cpu().numpy())
        all_hard_gates.append(hard_gates.squeeze(0).cpu().numpy())

        # Env step
        obs, reward, done, _info = env.step(action_np)
        if done:
            obs = env.reset()
            state = agent.get_initial_state(1)

    return {
        "soft_gates": np.stack(all_gates),     # (T, K, K)
        "hard_gates": np.stack(all_hard_gates), # (T, K, K)
    }


@torch.no_grad()
def gate_bypass_diagnostic(agent, checkpoint_path, config_path, task_name, rollout_steps=200, device="cuda:0"):
    """Compute ΔNLL between full-gate and zero-gate prior."""
    env = make_eval_env(task_name, seed=123)
    obs = env.reset()
    state = agent.get_initial_state(1)

    total_delta_nll = 0.0
    total_ll_ratio = 0.0
    count = 0

    for step in range(rollout_steps):
        image = obs["image"].copy()
        is_first_val = obs.get("is_first", step == 0)
        if isinstance(is_first_val, (bool, np.bool_)):
            is_first_flag = bool(is_first_val)
        else:
            is_first_flag = bool(is_first_val)

        obs_tensor = {
            "image": torch.from_numpy(image).unsqueeze(0).to(device),
            "is_first": torch.tensor([[is_first_flag]], dtype=torch.bool, device=device),
            "is_last": torch.tensor([[False]], dtype=torch.bool, device=device),
            "is_terminal": torch.tensor([[False]], dtype=torch.bool, device=device),
        }

        action, new_state = agent.act(obs_tensor, state, eval=True)
        action_np = action.squeeze(0).cpu().numpy()

        prev_stoch = state["stoch"]
        prev_deter = state["deter"]
        prev_action = state["prev_action"]
        target_stoch = new_state["stoch"]

        # Run bypass diagnostic
        try:
            diag = agent.rssm.gate_bypass_diagnostic(
                prev_deter, prev_stoch, prev_action, target_stoch,
                torch.ones(1, agent.rssm.num_factors, agent.rssm.num_factors, device=device)
                if not hasattr(agent.rssm, '_factor_gate') else
                agent.rssm._factor_gate(prev_deter.detach(), temperature=0.1)[0]
            )
            total_delta_nll += diag["delta_nll"]
            total_ll_ratio += diag["log_likelihood_ratio"]
            count += 1
        except Exception:
            pass

        state = new_state
        obs, reward, done, _info = env.step(action_np)
        if done:
            obs = env.reset()
            state = agent.get_initial_state(1)

    env.close()

    if count == 0:
        return {"delta_nll": 0.0, "ll_ratio": 0.0, "count": 0}

    return {
        "delta_nll": total_delta_nll / count,
        "ll_ratio": total_ll_ratio / count,
        "count": count,
    }


def main():
    parser = argparse.ArgumentParser(description="Gate Structure Analysis (B4)")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("config_dir", type=str)
    parser.add_argument("device", type=str, default="cuda:0")
    parser.add_argument("output", type=str)
    parser.add_argument("--task", type=str, default="walker_walk")
    parser.add_argument("--rollout-steps", type=int, default=500)
    args = parser.parse_args()

    config_path = Path(args.config_dir)
    if config_path.is_dir():
        config_path = config_path / ".hydra" / "config.yaml"
        if not config_path.exists():
            config_path = Path(args.config_dir) / "config.yaml"

    print(f"Loading agent: {args.checkpoint}")
    agent = load_agent(args.checkpoint, str(config_path), args.device)

    if not agent.sf_enabled:
        print("ERROR: Agent is not SF-enabled. Gate analysis requires SF-RSSM.")
        sys.exit(1)

    print(f"Collecting gate matrices over {args.rollout_steps} steps...")
    env = make_eval_env(args.task, seed=42)
    gate_data = collect_gate_matrices(agent, env, args.rollout_steps, args.device)
    env.close()

    # Average gate matrix
    avg_soft = gate_data["soft_gates"].mean(axis=0)  # (K, K)
    avg_hard = gate_data["hard_gates"].mean(axis=0)  # (K, K)
    std_hard = gate_data["hard_gates"].std(axis=0)   # (K, K)

    K = avg_hard.shape[0]

    # Edge sparsity
    non_diag_mask = ~np.eye(K, dtype=bool)
    edge_fraction = avg_hard[non_diag_mask].mean()

    # AUROC vs kinematic DAG
    auroc = None
    gt_dag = KINEMATIC_DAG.get(args.task)
    if gt_dag is not None:
        auroc = compute_auroc(avg_hard, gt_dag)
        print(f"  AUROC (edge recovery): {auroc:.4f}")
    else:
        print(f"  No ground-truth DAG for task '{args.task}' — AUROC not computed")

    print(f"  Edge fraction (non-diagonal): {edge_fraction:.4f}")

    # Gate-bypass diagnostic
    print("Running gate-bypass diagnostic...")
    bypass = gate_bypass_diagnostic(
        agent, args.checkpoint, str(config_path),
        args.task, rollout_steps=200, device=args.device
    )
    print(f"  ΔNLL (higher = gates matter more): {bypass['delta_nll']:.6f}")
    print(f"  Log-likelihood ratio: {bypass['ll_ratio']:.6f}")
    print(f"  Samples: {bypass['count']}")

    # Save results
    output = {
        "checkpoint": args.checkpoint,
        "task": args.task,
        "num_factors": K,
        "rollout_steps": args.rollout_steps,
        "avg_gate_matrix_soft": avg_soft.tolist(),
        "avg_gate_matrix_hard": avg_hard.tolist(),
        "std_gate_matrix_hard": std_hard.tolist(),
        "edge_fraction": float(edge_fraction),
        "auroc_edge_recovery": auroc,
        "gt_dag": gt_dag.tolist() if gt_dag is not None else None,
        "gate_bypass": bypass,
        "summary": {
            "sparsity": float(edge_fraction),
            "auroc": auroc,
            "delta_nll": bypass["delta_nll"],
            "ll_ratio": bypass["ll_ratio"],
        },
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
