"""
OOD Evaluation Script for Sparse Factorized RSSM.
Evaluates trained DreamerV3/SF-RSSM checkpoints on DeepMind Control Suite
with visual distractor and dynamics shift variants.
"""
import argparse
import json
import os
import pathlib
import sys
from collections import defaultdict

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf, DictConfig

sys.path.append(str(pathlib.Path(__file__).parent))
# Import the dreamer module — we patch dreamer's Dreamer class
# so that sf_rssm is available when loading checkpoints
import sf_rssm  # noqa: F401 — registers SF-RSSM modules for pickle

from dreamer import Dreamer
from envs import make_envs


def create_distractor_env(cfg, distractor_type, seed=0):
    """Create a DMC environment with visual distractors.

    We modify environment attributes to simulate distractors:
    - 'color_bg': randomize background color
    - 'texture_noise': add random texture over background
    - 'camera_angle': perturb camera angle
    """
    import warnings
    import copy

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env_cfg = copy.deepcopy(cfg.env)
        # Mark for distractor pipeline via custom config
        env_cfg.distractor = distractor_type
        train_envs, eval_envs, obs_space, act_space = make_envs(env_cfg)
    return eval_envs[0], obs_space, act_space


def evaluate(agent, env, num_episodes=10, seed=0):
    """Evaluate agent on environment, return mean episode return."""
    env.seed(seed)
    returns = []
    for ep in range(num_episodes):
        obs = env.reset()
        state = agent.get_initial_state(B=1)
        total_reward = 0.0
        done = False
        step = 0
        while not done and step < 1000:
            with torch.no_grad():
                obs_tensor = {
                    k: torch.as_tensor(v, device=agent.device, dtype=torch.float32).unsqueeze(0)
                    if not isinstance(v, dict)
                    else {
                        kk: torch.as_tensor(vv, device=agent.device, dtype=torch.float32).unsqueeze(0)
                        for kk, vv in v.items()
                    }
                    for k, v in obs.items()
                }
                # Add is_first
                obs_tensor["is_first"] = torch.zeros(1, dtype=torch.bool, device=agent.device)
                if step == 0:
                    obs_tensor["is_first"][0] = True

                action, state = agent.act(obs_tensor, state, eval=True)
                action = action.squeeze(0).cpu().numpy()

            obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1
        returns.append(total_reward)
    return np.mean(returns), np.std(returns)


def evaluate_dynamics_shift(agent, env_cfg, shift_param, shift_value, num_episodes=10, seed=0):
    """Evaluate with dynamics parameter shift.

    Note: DMC via dm_control doesn't easily support runtime dynamics changes.
    We simulate this by creating a modified environment.
    For a more rigorous approach, use the Physics wrapper.
    """
    import warnings
    import copy

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev_cfg = copy.deepcopy(env_cfg)
        # Store shift info for environment factory
        ev_cfg.dynamics_shift = {shift_param: shift_value}
        train_envs, eval_envs, obs_space, act_space = make_envs(ev_cfg)
    env = eval_envs[0]
    return evaluate(agent, env, num_episodes, seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to hydra config")
    parser.add_argument("--tasks", type=str, nargs="+", default=["walker_walk", "cheetah_run", "cartpole_swingup"])
    parser.add_argument("--distractors", type=str, nargs="+", default=["none", "color_bg", "texture_noise", "camera_angle"])
    parser.add_argument("--dynamics_shifts", type=str, nargs="+", default=["none", "friction_plus", "mass_plus", "damping_plus"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--output", type=str, default="ood_results.json")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # Load config
    config = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(OmegaConf.structured(DictConfig({})), config)

    results = defaultdict(dict)
    device = torch.device(args.device)

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    agent_cfg = cfg.model
    agent_cfg.device = str(device)

    # Get obs/act spaces from a task
    from dm_control import suite
    task_name = args.tasks[0]
    domain, task = task_name.split("_", 1)
    dm_env = suite.load(domain, task)

    results["method"] = "sf_rssm" if getattr(agent_cfg.rssm, 'sf_enabled', False) else "dreamerv3"
    results["per_task"] = {}

    for task_name in args.tasks:
        domain, task = task_name.split("_", 1)
        task_results = {}

        # In-distribution evaluation
        in_dist_returns = []
        env_cfg = copy.deepcopy(cfg)

        for seed in args.seeds:
            from dm_control import suite
            dm_env = suite.load(domain, task)
            from envs.wrappers import DMCWrapper
            env = DMCWrapper(dm_env, action_repeat=cfg.env.action_repeat)

            eval_return, eval_std = evaluate(agent, env, args.episodes, seed)
            in_dist_returns.append(eval_return)

        in_dist_mean = np.mean(in_dist_returns)
        task_results["in_dist"] = float(in_dist_mean)

        # Visual distractor evaluation
        for dist in args.distractors:
            if dist == "none":
                continue
            dist_returns = []
            for seed in args.seeds:
                dist_return, _ = evaluate_with_distractor(agent, domain, task, dist, args.episodes, seed)
                dist_returns.append(dist_return)
            dist_mean = np.mean(dist_returns)
            retention = dist_mean / (in_dist_mean + 1e-8)
            task_results[f"dist_{dist}"] = float(dist_mean)
            task_results[f"dist_{dist}_retention"] = float(retention)

        # Dynamics shift evaluation
        for shift in args.dynamics_shifts:
            if shift == "none":
                continue
            shift_returns = []
            for seed in args.seeds:
                shift_return, _ = evaluate_dynamics_shift(agent, env_cfg, shift, 0.5, args.episodes, seed)
                shift_returns.append(shift_return)
            shift_mean = np.mean(shift_returns)
            retention = shift_mean / (in_dist_mean + 1e-8)
            task_results[f"dyn_{shift}"] = float(shift_mean)
            task_results[f"dyn_{shift}_retention"] = float(retention)

        results["per_task"][task_name] = task_results

    # Aggregate OOD retention
    all_retentions = []
    for task_name, task_data in results["per_task"].items():
        for key, val in task_data.items():
            if "retention" in key:
                all_retentions.append(val)
    results["mean_ood_retention"] = float(np.mean(all_retentions))
    results["std_ood_retention"] = float(np.std(all_retentions))

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Results saved to {args.output}")
    print(f"Mean OOD Retention: {results['mean_ood_retention']:.3f} ± {results['std_ood_retention']:.3f}")


def evaluate_with_distractor(agent, domain, task, distractor_type, num_episodes, seed):
    """Evaluate with visual distractors applied to observations."""
    from dm_control import suite
    import numpy as np

    np.random.seed(seed)
    dm_env = suite.load(domain, task)
    from dm_control.suite.wrappers import pixels

    # Load env with pixel observations
    env = pixels.Wrapper(dm_env, pixels_only=False)
    # Apply distractor
    env.step = make_distractor_step(env.step, distractor_type, seed + num_episodes)
    return evaluate_obs_env(agent, env, num_episodes, seed)


def make_distractor_step(original_step, dist_type, seed):
    """Wraps env.step to add visual distractors to observations."""
    rng = np.random.RandomState(seed)

    def distractor_step(action):
        timestep = original_step(action)
        obs = timestep.observation
        if "pixels" in obs:
            img = obs["pixels"].astype(np.float32)
            if dist_type == "color_bg":
                # Add constant color offset to background
                bg_mask = img.mean(axis=-1) > 200  # simple heuristic for white bg
                color = rng.randint(0, 255, size=3).astype(np.float32)
                img[bg_mask] = 0.7 * img[bg_mask] + 0.3 * color
            elif dist_type == "texture_noise":
                # Add random Gaussian texture
                noise = rng.randn(*img.shape).astype(np.float32) * 10
                img = np.clip(img + noise, 0, 255)
            elif dist_type == "camera_angle":
                # Slight rotation via shift
                shift = rng.randint(-3, 4)
                img = np.roll(img, shift, axis=1)  # horizontal shift
            obs["pixels"] = img.astype(np.uint8)
            timestep = timestep._replace(observation=obs)
        return timestep
    return distractor_step


def evaluate_obs_env(agent, env, num_episodes, seed):
    """Evaluate on an environment that returns pixel observations."""
    returns = []
    for ep in range(num_episodes):
        timestep = env.reset()
        state = agent.get_initial_state(B=1)
        total_reward = 0.0
        done = False
        step = 0
        while not done and step < 1000:
            obs = timestep.observation
            obs_dict = {"image": torch.as_tensor(obs["pixels"], device=agent.device, dtype=torch.float32).unsqueeze(0)}
            obs_dict["is_first"] = torch.zeros(1, dtype=torch.bool, device=agent.device)
            if step == 0:
                obs_dict["is_first"][0] = True

            with torch.no_grad():
                action, state = agent.act(obs_dict, state, eval=True)
                action = action.squeeze(0).cpu().numpy()

            timestep = env.step(action)
            total_reward += timestep.reward
            done = timestep.last()
            step += 1
        returns.append(total_reward)
    return np.mean(returns), np.std(returns)


if __name__ == "__main__":
    main()
