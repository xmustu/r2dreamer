"""OOD eval by loading full Dreamer agent and running act()."""
import sys, pathlib, json, warnings
import numpy as np
import torch
from omegaconf import OmegaConf

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path.home() / "r2dreamer"))
import sf_rssm, dreamer, rssm, networks, tools

CKPT_PATH = sys.argv[1]
CFG_PATH = sys.argv[2]
DEV = sys.argv[3]
OUT = sys.argv[4]
TASK = sys.argv[5]

torch.set_float32_matmul_precision("high")
device = torch.device(DEV)

ckpt = torch.load(CKPT_PATH, map_location=DEV)
cfg = OmegaConf.load(CFG_PATH)
cfg.model.device = DEV

from gymnasium import spaces
class FakeObsSpace:
    def __init__(self):
        self.spaces = {"image": spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)}
class FakeActSpace:
    def __init__(self, n):
        self.n = n
        self.shape = (n,)

act_dim = 1
agent = dreamer.Dreamer(cfg.model, FakeObsSpace(), FakeActSpace(act_dim)).to(device)
agent.load_state_dict(ckpt['agent_state_dict'], strict=False)
agent.eval()

from dm_control import suite
from dm_control.suite.wrappers import pixels
domain, task_name = TASK.replace("dmc_", "").split("_", 1)
dm_env = suite.load(domain, task_name)
env = pixels.Wrapper(dm_env, pixels_only=False)

def run_eval(distractor="none", episodes=5):
    rng = np.random.RandomState(42)
    returns = []
    for ep in range(episodes):
        timestep = env.reset()
        state = agent.get_initial_state(B=1)
        total_r = 0.0
        step = 0
        while step < 500:
            obs = timestep.observation
            img = obs["pixels"].astype(np.float32) / 255.0
            if distractor == "color_bg":
                bg = img.mean(-1) > 0.7
                color = rng.randint(0, 256, 3).astype(np.float32) / 255.0
                img[bg] = 0.7 * img[bg] + 0.3 * color
            elif distractor == "texture":
                img = np.clip(img + rng.randn(*img.shape).astype(np.float32) * 0.04, 0, 1)
            elif distractor == "shift":
                s = rng.randint(-5, 6)
                img = np.roll(img, s, axis=1)
            obs_tensor = {"image": torch.as_tensor(img, device=device, dtype=torch.float32).unsqueeze(0)}
            obs_tensor["is_first"] = torch.zeros(1, dtype=torch.bool, device=device)
            if step == 0: obs_tensor["is_first"][0] = True
            with torch.no_grad():
                action, state = agent.act(obs_tensor, state, eval=True)
                action = action.squeeze(0).cpu().numpy()
            timestep = env.step(action)
            if timestep.last():
                break
            total_r += timestep.reward
            step += 1
        returns.append(total_r)
    return float(np.mean(returns)), float(np.std(returns))

results = {"method": "dreamerv3" if "dv3" in CKPT_PATH else "sf_rssm"}
for dist in ["none", "color_bg", "texture", "shift"]:
    mu, std = run_eval(dist)
    results[dist] = {"mean": mu, "std": std}
    print(f"  {dist:12s}: {mu:7.1f} +/- {std:.1f}")

base = results["none"]["mean"]
for dist in ["color_bg", "texture", "shift"]:
    results[f"{dist}_retention"] = results[dist]["mean"] / (base + 1e-8)

results["in_dist_return"] = base
with open(OUT, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nIn-dist: {base:.1f}")
for d in ["color_bg", "texture", "shift"]:
    print(f"  {d} retention: {results[f'{d}_retention']:.3f}")
