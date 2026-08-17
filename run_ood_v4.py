"""OOD eval — single env (no multiprocessing), uses R2-Dreamer env factory."""
import sys, pathlib, json, warnings, copy, os
import numpy as np
import torch
from omegaconf import OmegaConf

if __name__ != '__main__':
    # Prevent multiprocessing recursion
    print("SKIP_IMPORT")
    sys.exit(0)

warnings.filterwarnings("ignore")
HOME = pathlib.Path.home()
sys.path.insert(0, str(HOME / "r2dreamer"))
os.chdir(str(HOME / "r2dreamer"))

import sf_rssm, dreamer, tools, envs as envs_mod

CKPT_PATH = sys.argv[1]
CFG_PATH = sys.argv[2]
DEV = sys.argv[3]
OUT = sys.argv[4]

torch.set_float32_matmul_precision("high")
device = torch.device(DEV)
ckpt = torch.load(CKPT_PATH, map_location=DEV)
cfg = OmegaConf.load(CFG_PATH)

# Create single eval env (no parallel)
from envs import make_env
eval_env = make_env(cfg.env, 0)
obs_space = eval_env.observation_space

# Reconstruct act_space from config
from gymnasium import spaces
act_dim_cfg = getattr(cfg.model.actor, 'shape', None)
if act_dim_cfg is not None:
    act_dim = act_dim_cfg[0] if isinstance(act_dim_cfg, (list, tuple)) else act_dim_cfg
else:
    act_dim = eval_env.action_space.shape[0] if hasattr(eval_env.action_space, 'shape') else 1
act_space = spaces.Box(-1, 1, (act_dim,))

# Build agent
agent = dreamer.Dreamer(cfg.model, obs_space, act_space).to(device)
agent.load_state_dict(ckpt['agent_state_dict'], strict=False)
agent.eval()

# Only first reset may create distractor_env issues — create fresh env per distractor
def make_fresh_env():
    from envs import make_env as mk
    return mk(cfg.env, 0)

class DistractorWrapper:
    def __init__(self, dist_type, seed):
        self.env = make_fresh_env()
        self.dist_type = dist_type
        self.rng = np.random.RandomState(seed)
    
    def reset(self):
        result = self.env.reset(); obs = result[0] if isinstance(result, tuple) else result
        return self._perturb(obs)
    
    def step(self, action):
        result = self.env.step(action); obs, reward, term, trunc = result[:4]
        return self._perturb(obs), reward, term, trunc
    
    def _perturb(self, obs):
        if self.dist_type == "none":
            return obs
        obs = copy.deepcopy(obs)
        if "image" in obs:
            img = obs["image"].copy().astype(np.float32)
            if self.dist_type == "color_bg":
                bg = img.mean(-1) > 0.55
                color = self.rng.randint(0, 256, 3).astype(np.float32) / 255.0
                img[bg] = 0.6 * img[bg] + 0.4 * color
            elif self.dist_type == "texture":
                noise = self.rng.randn(*img.shape).astype(np.float32) * 0.06
                img = np.clip(img + noise, 0.0, 1.0)
            elif self.dist_type == "shift":
                s = self.rng.randint(-8, 9)
                img = np.roll(img, s, axis=1)
            obs["image"] = img
        return obs

def run_eval(dist_type, episodes=10):
    dist_env = DistractorWrapper(dist_type, seed=42)
    returns = []
    for ep in range(episodes):
        obs = dist_env.reset(); obs = obs[0] if isinstance(obs, tuple) else obs
        state = agent.get_initial_state(B=1)
        total_r = 0.0









            if term or trunc:
                break
        returns.append(total_r)
    return float(np.mean(returns)), float(np.std(returns))

print(f"Evaluating: {CKPT_PATH.split('/')[-3]}")
results = {}
for dist in ["none", "color_bg", "texture", "shift"]:
    mu, std = run_eval(dist)
    results[dist] = {"mean": mu, "std": std}
    print(f"  {dist:12s}: {mu:8.1f} +/- {std:.1f}")

base = results["none"]["mean"]
results["in_dist_return"] = base
results["method"] = CKPT_PATH.split("/")[-3]
for dist in ["color_bg", "texture", "shift"]:
    results[f"{dist}_retention"] = results[dist]["mean"] / (base + 1e-8)

print(f"\nIn-dist return: {base:.1f}")
for d in ["color_bg", "texture", "shift"]:
    print(f"  {d} retention: {results[f'{d}_retention']:.3f}")

with open(OUT, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"Saved to {OUT}")
