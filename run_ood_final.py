"""OOD eval using R2-Dreamer env factory + agent checkpoint."""
import sys, pathlib, json, warnings, copy
import numpy as np
import torch
from omegaconf import OmegaConf

warnings.filterwarnings("ignore")
HOME = pathlib.Path.home()
sys.path.insert(0, str(HOME / "r2dreamer"))
import sf_rssm, dreamer, tools, envs as envs_mod

CKPT_PATH = sys.argv[1]   # latest.pt
CFG_PATH = sys.argv[2]    # .hydra/config.yaml
DEV = sys.argv[3]
OUT = sys.argv[4]

torch.set_float32_matmul_precision("high")
device = torch.device(DEV)

ckpt = torch.load(CKPT_PATH, map_location=DEV)
cfg = OmegaConf.load(CFG_PATH)

# Create envs the same way as training
train_envs, eval_envs, obs_space, act_space = envs_mod.make_envs(cfg.env)

# Build agent
agent = dreamer.Dreamer(cfg.model, obs_space, act_space).to(device)
agent.load_state_dict(ckpt['agent_state_dict'], strict=False)
agent.eval()

# Get reference env
env = eval_envs[0]

class DistractorWrapper:
    """Wraps a Gym env to add visual distractors to observations."""
    def __init__(self, env, dist_type, seed=42):
        self.env = env
        self.dist_type = dist_type
        self.rng = np.random.RandomState(seed)
    
    def reset(self):
        obs, info = self.env.reset()
        return self._perturb(obs), info
    
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return self._perturb(obs), reward, term, trunc, info
    
    def _perturb(self, obs):
        if self.dist_type == "none":
            return obs
        obs = copy.deepcopy(obs)
        if "image" in obs:
            img = obs["image"].copy()
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
            obs["image"] = img.astype(np.float32)
        return obs

def run_eval(dist_type, episodes=10):
    dist_env = DistractorWrapper(env, dist_type)
    returns = []
    for ep in range(episodes):
        obs, _ = dist_env.reset()
        state = agent.get_initial_state(B=1)
        total_r = 0.0
        while True:
            obs_t = {k: torch.as_tensor(v, device=device, dtype=torch.float32).unsqueeze(0) 
                      if not isinstance(v, dict) else v for k, v in obs.items()}
            obs_t["is_first"] = torch.zeros(1, dtype=torch.bool, device=device)
            if ep == 0: obs_t["is_first"] = torch.ones(1, dtype=torch.bool, device=device)
            with torch.no_grad():
                action, state = agent.act(obs_t, state, eval=True)
            obs, reward, term, trunc, info = dist_env.step(action.squeeze(0).cpu().numpy())
            total_r += reward
            if term or trunc: break
        returns.append(total_r)
    return float(np.mean(returns)), float(np.std(returns))

results = {}
is_sf = "sf" in str(CKPT_PATH)
results["method"] = "sf_rssm" if is_sf else "dreamerv3"
print(f"Evaluating: {results['method']}")

for dist in ["none", "color_bg", "texture", "shift"]:
    mu, std = run_eval(dist)
    results[dist] = {"mean": mu, "std": std}
    print(f"  {dist:12s}: {mu:8.1f} +/- {std:.1f}")

base = results["none"]["mean"]
results["in_dist_return"] = base
for dist in ["color_bg", "texture", "shift"]:
    results[f"{dist}_retention"] = results[dist]["mean"] / (base + 1e-8)

print(f"\nIn-dist: {base:.1f}")
for d in ["color_bg", "texture", "shift"]:
    print(f"  {d} retention: {results[f'{d}_retention']:.3f}")

with open(OUT, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"Saved to {OUT}")
