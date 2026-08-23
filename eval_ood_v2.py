#!/usr/bin/env python3
"""
OOD Evaluation Script (v2) — Perturbation Taxonomy for Mechanism-Aware World Models.

Evaluates trained DreamerV3 / SF-RSSM agents under two categories of distribution shift:

Mechanism-Aligned (structural prior SHOULD help):
  M1: Dynamics — friction coefficient ±50%
  M2: Dynamics — torso mass ±50%
  M3: Spatial — camera horizontal shift ±8px
  M4: Spatial — camera rotation ±5°

Encoder-Corrupting (structural prior CANNOT help — negative control):
  E1: Texture — Gaussian noise σ=0.05 on pixels
  E2: Color — random background color replacement
  E3: Blur — Gaussian blur σ=2.0

Usage:
  python eval_ood_v2.py <checkpoint.pt> <config_dir> <device> <output.json>
      [--task dmc_walker_walk] [--episodes 10] [--seed 42]
      [--perturbations M1,M2,M3,M4,E1,E2,E3]
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
from scipy.ndimage import gaussian_filter

# Add r2dreamer to path
sys.path.insert(0, str(Path(__file__).parent))

from dreamer import Dreamer
from envs.dmc import DeepMindControl
from envs import wrappers


# ---------------------------------------------------------------------------
# Perturbation implementations
# ---------------------------------------------------------------------------

def perturb_friction(env, scale=1.5):
    """M1: Scale all geom friction coefficients.

    Args:
        env: DeepMindControl env wrapper
        scale: multiplier for friction (1.5 = +50%, 0.5 = -50%)
    """
    physics = env._env.physics
    friction = physics.model.geom_friction.copy()
    friction[:, 0] *= scale  # sliding friction
    physics.model.geom_friction[:] = friction


def perturb_mass(env, scale=1.5):
    """M2: Scale torso/root body mass.

    For most DMC tasks, the root body is the first body (index 1 after world).
    """
    physics = env._env.physics
    mass = physics.model.body_mass.copy()
    # Scale all non-world bodies
    for i in range(1, len(mass)):
        mass[i] *= scale
    physics.model.body_mass[:] = mass


def perturb_damping(env, scale=1.5):
    """M2-alt: Scale joint damping coefficients."""
    physics = env._env.physics
    damping = physics.model.dof_damping.copy()
    damping *= scale
    physics.model.dof_damping[:] = damping


def perturb_actuator_gain(env, scale=1.3):
    """M5: Scale actuator gain (control authority)."""
    physics = env._env.physics
    gain = physics.model.actuator_gainprm.copy()
    gain[:, 0] *= scale  # First column is typically the gain
    physics.model.actuator_gainprm[:] = gain


def perturb_gravity(env, scale=1.3):
    """M6: Scale gravity vector."""
    physics = env._env.physics
    gravity = physics.model.opt.gravity.copy()
    gravity[2] *= scale  # z-component of gravity
    physics.model.opt.gravity[:] = gravity


def perturb_joint_stiffness(env, scale=1.5):
    """M7: Scale joint stiffness."""
    physics = env._env.physics
    stiffness = physics.model.jnt_stiffness.copy()
    stiffness *= scale
    physics.model.jnt_stiffness[:] = stiffness


def perturb_timestep(env, scale=1.5):
    """M8: Scale control timestep (effectively changes simulation speed)."""
    physics = env._env.physics
    ts = physics.model.opt.timestep
    physics.model.opt.timestep = ts * scale


def perturb_camera_shift(env, dx_pixels=8, image_size=64):
    """M3: Shift camera horizontally by modifying the render camera matrix.

    In dm_control, the camera is defined by the MuJoCo camera in the model.
    We shift by modifying the rendered image via a translation matrix.
    This is a post-render shift for simplicity.
    """
    pass  # Applied post-render in perturb_frame()


def perturb_camera_rotation(env, degrees=5.0):
    """M4: Slight camera rotation.

    Applied post-render for simplicity.
    """
    pass  # Applied post-render in perturb_frame()


# ---------------------------------------------------------------------------
# Frame-level perturbations (post-render)
# ---------------------------------------------------------------------------

def apply_frame_shift(frame, dx_pixels=8):
    """Shift image horizontally by dx pixels (wrap-around)."""
    if dx_pixels == 0:
        return frame
    return np.roll(frame, dx_pixels, axis=1)


def apply_frame_rotation(frame, degrees=5.0):
    """Rotate image by degrees (with crop to original size)."""
    from scipy.ndimage import rotate
    rotated = rotate(frame, degrees, axes=(0, 1), reshape=False, mode='reflect')
    return rotated


def apply_texture_noise(frame, sigma=0.05):
    """E1: Add Gaussian noise to pixels (image in [0, 255])."""
    noise = np.random.randn(*frame.shape).astype(np.float32) * sigma * 255.0
    noisy = frame.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_color_background(frame):
    """E2: Replace background with random solid color.

    Simple heuristic: replace pixels near the edges (likely background).
    """
    bg_color = np.random.randint(0, 256, size=3, dtype=np.uint8)
    result = frame.copy()
    # Replace a border region (approximates background replacement)
    border = 8
    result[:border, :] = bg_color
    result[-border:, :] = bg_color
    result[:, :border] = bg_color
    result[:, -border:] = bg_color
    return result


def apply_blur(frame, sigma=2.0):
    """E3: Gaussian blur on image."""
    blurred = np.zeros_like(frame, dtype=np.float32)
    for c in range(frame.shape[-1]):
        blurred[:, :, c] = gaussian_filter(frame[:, :, c].astype(np.float32), sigma=sigma)
    return np.clip(blurred, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Perturbation registry
# ---------------------------------------------------------------------------

PERTURBATION_SPECS = {
    # Mechanism-Aligned
    "M1_high_friction": {
        "category": "mechanism",
        "subtype": "friction",
        "description": "Friction +50%",
        "physics_fn": lambda env: perturb_friction(env, scale=1.5),
        "frame_fn": None,
    },
    "M1_low_friction": {
        "category": "mechanism",
        "subtype": "friction",
        "description": "Friction -50%",
        "physics_fn": lambda env: perturb_friction(env, scale=0.5),
        "frame_fn": None,
    },
    "M2_high_mass": {
        "category": "mechanism",
        "subtype": "mass",
        "description": "Mass +50%",
        "physics_fn": lambda env: perturb_mass(env, scale=1.5),
        "frame_fn": None,
    },
    "M2_low_mass": {
        "category": "mechanism",
        "subtype": "mass",
        "description": "Mass -50%",
        "physics_fn": lambda env: perturb_mass(env, scale=0.5),
        "frame_fn": None,
    },
    "M3_camera_shift": {
        "category": "mechanism",
        "subtype": "spatial",
        "description": "Camera shift +8px",
        "physics_fn": None,
        "frame_fn": lambda frame: apply_frame_shift(frame, dx_pixels=8),
    },
    "M4_camera_rotation": {
        "category": "mechanism",
        "subtype": "spatial",
        "description": "Camera rotation +5°",
        "physics_fn": None,
        "frame_fn": lambda frame: apply_frame_rotation(frame, degrees=5.0),
    },
    # Additional dynamics perturbations (v1 expanded set)
    "M5_high_damping": {
        "category": "mechanism",
        "subtype": "damping",
        "description": "Joint damping +50%",
        "physics_fn": lambda env: perturb_damping(env, scale=1.5),
        "frame_fn": None,
    },
    "M5_low_damping": {
        "category": "mechanism",
        "subtype": "damping",
        "description": "Joint damping -50%",
        "physics_fn": lambda env: perturb_damping(env, scale=0.5),
        "frame_fn": None,
    },
    "M6_high_actuator": {
        "category": "mechanism",
        "subtype": "actuator",
        "description": "Actuator gain +30%",
        "physics_fn": lambda env: perturb_actuator_gain(env, scale=1.3),
        "frame_fn": None,
    },
    "M6_low_actuator": {
        "category": "mechanism",
        "subtype": "actuator",
        "description": "Actuator gain -30%",
        "physics_fn": lambda env: perturb_actuator_gain(env, scale=0.7),
        "frame_fn": None,
    },
    "M7_high_gravity": {
        "category": "mechanism",
        "subtype": "gravity",
        "description": "Gravity +30%",
        "physics_fn": lambda env: perturb_gravity(env, scale=1.3),
        "frame_fn": None,
    },
    "M7_low_gravity": {
        "category": "mechanism",
        "subtype": "gravity",
        "description": "Gravity -30%",
        "physics_fn": lambda env: perturb_gravity(env, scale=0.7),
        "frame_fn": None,
    },
    "M8_high_stiffness": {
        "category": "mechanism",
        "subtype": "stiffness",
        "description": "Joint stiffness +50%",
        "physics_fn": lambda env: perturb_joint_stiffness(env, scale=1.5),
        "frame_fn": None,
    },
    "M8_low_stiffness": {
        "category": "mechanism",
        "subtype": "stiffness",
        "description": "Joint stiffness -50%",
        "physics_fn": lambda env: perturb_joint_stiffness(env, scale=0.5),
        "frame_fn": None,
    },
    "M9_slow_timestep": {
        "category": "mechanism",
        "subtype": "timestep",
        "description": "Control timestep ×1.5 (slower control)",
        "physics_fn": lambda env: perturb_timestep(env, scale=1.5),
        "frame_fn": None,
    },
    "M9_fast_timestep": {
        "category": "mechanism",
        "subtype": "timestep",
        "description": "Control timestep ×0.67 (faster control)",
        "physics_fn": lambda env: perturb_timestep(env, scale=0.67),
        "frame_fn": None,
    },
    # Encoder-Corrupting (negative controls)
    "E1_texture_noise": {
        "category": "encoder",
        "subtype": "texture",
        "description": "Gaussian noise σ=0.05",
        "physics_fn": None,
        "frame_fn": lambda frame: apply_texture_noise(frame, sigma=0.05),
    },
    "E2_color_bg": {
        "category": "encoder",
        "subtype": "color",
        "description": "Random background color",
        "physics_fn": None,
        "frame_fn": apply_color_background,
    },
    "E3_blur": {
        "category": "encoder",
        "subtype": "blur",
        "description": "Gaussian blur σ=2.0",
        "physics_fn": None,
        "frame_fn": lambda frame: apply_blur(frame, sigma=2.0),
    },
}


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_eval_env(task_name, seed=42, image_size=64):
    """Create a single eval environment for the given DMC task.

    Returns the raw DeepMindControl env (no action normalization wrapper)
    so that physics perturbations can access env._env.physics directly.
    The agent already outputs actions in the correct range.
    """
    env = DeepMindControl(
        task_name,
        action_repeat=2,
        size=(image_size, image_size),
        seed=seed,
    )
    return env


# ---------------------------------------------------------------------------
# Agent loader
# ---------------------------------------------------------------------------

def load_agent(checkpoint_path, config_path, device="cuda:0"):
    """Load a trained Dreamer agent from checkpoint."""
    import omegaconf
    import gymnasium as gym

    # Load config
    config = omegaconf.OmegaConf.load(config_path)

    # Infer action dimension from checkpoint to avoid shape mismatch
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["agent_state_dict"]

    # actor.last.weight shape is (act_dim*2, 256) for bounded_normal (mean+std)
    actor_out = state_dict["actor.last.weight"].shape[0]
    # For continuous actions with bounded_normal: output = 2 * act_dim
    # For discrete: output = act_dim (onehot)
    # Check if bounded_normal by looking at config
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

    # Disable torch.compile for OOD eval — we only use agent.act(), not update()
    # torch.compile("reduce-overhead") can take 10+ min and is unnecessary here
    import omegaconf
    if hasattr(config.model, "compile"):
        config.model.compile = False

    # Override ALL device fields in config — otherwise RSSM/encoder create
    # tensors on the training GPU (e.g. cuda:2) while the agent is moved to
    # the eval device → "Expected all tensors to be on the same device".
    def _override_devices(cfg):
        if isinstance(cfg, omegaconf.DictConfig):
            for k, v in cfg.items():
                if k == "device" and isinstance(v, str):
                    cfg[k] = device
                else:
                    _override_devices(v)
        elif isinstance(cfg, omegaconf.ListConfig):
            for v in cfg:
                _override_devices(v)

    _override_devices(config)

    agent = Dreamer(config.model, obs_space, act_space).to(torch.device(device))
    agent.load_state_dict(state_dict)
    agent.eval()
    agent.requires_grad_(False)

    return agent


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_episode(agent, env, perturbation_spec=None, max_steps=1000, device="cuda:0"):
    """Run a single episode, optionally with perturbation.

    Returns:
        total_reward: float
        episode_length: int
    """
    # Reset env
    obs = env.reset()

    # Apply physics perturbation AFTER reset (so physics model is initialized)
    if perturbation_spec and perturbation_spec.get("physics_fn"):
        perturbation_spec["physics_fn"](env)

    # Get initial state
    state = agent.get_initial_state(1)

    total_reward = 0.0
    step = 0

    for step in range(max_steps):
        # Apply frame perturbation BEFORE encoding
        image = obs["image"].copy()
        if perturbation_spec and perturbation_spec.get("frame_fn"):
            image = perturbation_spec["frame_fn"](image)

        # Prepare obs dict for agent
        # is_first must be boolean for torch.where in rssm.obs_step
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

        # Agent step
        action, state = agent.act(obs_tensor, state, eval=True)
        action_np = action.squeeze(0).cpu().numpy()

        # Env step
        obs, reward, done, _info = env.step(action_np)
        total_reward += float(reward)

        if done:
            break

    return total_reward, step + 1


@torch.no_grad()
def run_evaluation(agent, env_factory, task_name, perturbations, num_episodes=10, seed=42, device="cuda:0"):
    """Evaluate agent under clean and perturbed conditions.

    Returns:
        dict: {perturbation_name: {returns: [...], mean: float, std: float, retention: float}}
    """
    results = {}

    # Clean baseline
    clean_returns = []
    for ep in range(num_episodes):
        env = env_factory(task_name, seed=seed + ep * 100)
        ret, length = evaluate_episode(agent, env, perturbation_spec=None, device=device)
        clean_returns.append(ret)
        env._env.close()
    clean_mean = np.mean(clean_returns)
    clean_std = np.std(clean_returns)
    results["clean"] = {
        "returns": clean_returns,
        "mean": float(clean_mean),
        "std": float(clean_std),
        "category": "baseline",
    }

    # Perturbed evaluations
    for pert_name in perturbations:
        if pert_name not in PERTURBATION_SPECS:
            print(f"  WARNING: Unknown perturbation '{pert_name}', skipping")
            continue

        spec = PERTURBATION_SPECS[pert_name]
        pert_returns = []
        for ep in range(num_episodes):
            env = env_factory(task_name, seed=seed + ep * 100)
            ret, length = evaluate_episode(agent, env, perturbation_spec=spec, device=device)
            pert_returns.append(ret)
            # Clean up physics model modifications for next episode
            try:
                env._env.close()
            except Exception:
                pass

        pert_mean = np.mean(pert_returns)
        pert_std = np.std(pert_returns)
        retention = pert_mean / clean_mean if clean_mean != 0 else float("nan")

        results[pert_name] = {
            "returns": pert_returns,
            "mean": float(pert_mean),
            "std": float(pert_std),
            "retention": float(retention),
            "category": spec["category"],
            "subtype": spec["subtype"],
            "description": spec["description"],
        }

        print(f"  {pert_name:25s} | mean={pert_mean:8.2f} | retention={retention:.3f} | {spec['description']}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OOD Evaluation v2 — Perturbation Taxonomy")
    parser.add_argument("checkpoint", type=str, help="Path to latest.pt checkpoint")
    parser.add_argument("config_dir", type=str, help="Path to .hydra/config.yaml or directory containing it")
    parser.add_argument("device", type=str, default="cuda:0", help="Device for inference")
    parser.add_argument("output", type=str, help="Output JSON file path")
    parser.add_argument("--task", type=str, default="walker_walk",
                        help="DMC task name (domain_task, e.g., walker_walk)")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per condition")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--perturbations", type=str,
                        default="M1_high_friction,M1_low_friction,M2_high_mass,M2_low_mass,"
                                "M3_camera_shift,M4_camera_rotation,"
                                "M5_high_damping,M5_low_damping,"
                                "M6_high_actuator,M6_low_actuator,"
                                "M7_high_gravity,M7_low_gravity,"
                                "M8_high_stiffness,M8_low_stiffness,"
                                "M9_slow_timestep,M9_fast_timestep,"
                                "E1_texture_noise,E2_color_bg,E3_blur",
                        help="Comma-separated perturbation names to evaluate")
    args = parser.parse_args()

    # Resolve config path
    config_path = Path(args.config_dir)
    if config_path.is_dir():
        config_path = config_path / ".hydra" / "config.yaml"
        if not config_path.exists():
            config_path = Path(args.config_dir) / "config.yaml"

    if not config_path.exists():
        print(f"ERROR: Config not found at {args.config_dir}")
        sys.exit(1)

    print(f"Loading agent from: {args.checkpoint}")
    print(f"Config: {config_path}")
    print(f"Device: {args.device}")
    print(f"Task: dmc_{args.task}")
    print(f"Episodes per condition: {args.episodes}")
    print()

    # Load agent
    agent = load_agent(args.checkpoint, str(config_path), args.device)
    print("Agent loaded successfully.")
    print(f"SF-enabled: {agent.sf_enabled}")
    print()

    # Parse perturbations
    perturbation_list = [p.strip() for p in args.perturbations.split(",") if p.strip()]

    # Categorize
    mech_perts = [p for p in perturbation_list
                  if p in PERTURBATION_SPECS and PERTURBATION_SPECS[p]["category"] == "mechanism"]
    enc_perts = [p for p in perturbation_list
                 if p in PERTURBATION_SPECS and PERTURBATION_SPECS[p]["category"] == "encoder"]

    print(f"Mechanism-aligned perturbations ({len(mech_perts)}): {', '.join(mech_perts)}")
    print(f"Encoder-corrupting perturbations ({len(enc_perts)}): {', '.join(enc_perts)}")
    print()

    # Run evaluation
    print("Evaluating...")
    print("-" * 70)
    results = run_evaluation(
        agent, make_eval_env, args.task,
        perturbations=perturbation_list,
        num_episodes=args.episodes,
        seed=args.seed,
        device=args.device,
    )
    print("-" * 70)

    # Summary
    clean_mean = results["clean"]["mean"]
    print(f"\nClean mean return: {clean_mean:.2f}")

    # By category
    for category, label in [("mechanism", "Mechanism-Aligned"), ("encoder", "Encoder-Corrupting")]:
        cat_results = {k: v for k, v in results.items()
                       if v.get("category") == category}
        if cat_results:
            retentions = [v["retention"] for v in cat_results.values()
                          if not np.isnan(v["retention"])]
            if retentions:
                avg_ret = np.mean(retentions)
                print(f"{label}: avg retention = {avg_ret:.3f} ({', '.join(f'{r:.3f}' for r in retentions)})")

    # Save results
    output_data = {
        "checkpoint": args.checkpoint,
        "config": str(config_path),
        "task": args.task,
        "device": args.device,
        "num_episodes": args.episodes,
        "seed": args.seed,
        "sf_enabled": agent.sf_enabled,
        "results": results,
        "summary": {
            "clean_mean": float(clean_mean),
            "mechanism_retention": float(np.mean([
                v["retention"] for k, v in results.items()
                if v.get("category") == "mechanism" and not np.isnan(v.get("retention", float("nan")))
            ])) if mech_perts else None,
            "encoder_retention": float(np.mean([
                v["retention"] for k, v in results.items()
                if v.get("category") == "encoder" and not np.isnan(v.get("retention", float("nan")))
            ])) if enc_perts else None,
        },
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
