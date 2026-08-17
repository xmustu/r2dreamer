"""Minimal OOD eval — uses R2-Dreamer make_env, Dreamer agent, simple loop."""
import sys, pathlib, json, warnings, copy, os
import numpy as np
import torch
from omegaconf import OmegaConf

if __name__ != '__main__':
    sys.exit(0)

warnings.filterwarnings("ignore")
HOME = pathlib.Path.home()
SDIR = str(HOME / "r2dreamer")
sys.path.insert(0, SDIR)
os.chdir(SDIR)
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

CKPT = sys.argv[1]; CFG = sys.argv[2]; DEV = sys.argv[3]; OUT = sys.argv[4]
device = torch.device(DEV)
torch.set_float32_matmul_precision("high")

ckpt = torch.load(CKPT, map_location=DEV)
cfg = OmegaConf.load(CFG)

import sf_rssm, dreamer, tools
from envs import make_env as mk

env = mk(cfg.env, 0)
obs_space = env.observation_space

from gymnasium import spaces
act_dim = env.action_space.shape[0]
act_space = spaces.Box(-1, 1, (act_dim,))

agent = dreamer.Dreamer(cfg.model, obs_space, act_space).to(device)
agent.load_state_dict(ckpt['agent_state_dict'], strict=False)
agent.eval()

def eval_one(dist_type, episodes=10):
    returns = []
    for ep in range(episodes):
        env = mk(cfg.env, 0)  # fresh env
        result = env.reset(); obs = result[0] if isinstance(result, tuple) else result
        state = agent.get_initial_state(B=1)
        total = 0.0
        step = 0
        while step < 500:
            # perturb
            img = obs["image"].copy()
            if dist_type == "color_bg":
                bg = img.mean(-1) > 0.55
                c = np.random.RandomState(42+ep*100+step).randint(0,256,3)/255.0
                img[bg] = 0.6*img[bg] + 0.4*c
            elif dist_type == "texture":
                img = np.clip(img + np.random.RandomState(42+ep*100+step).randn(*img.shape)*0.05, 0, 1)
            elif dist_type == "shift":
                img = np.roll(img, np.random.RandomState(42+ep*100+step).randint(-5,6), axis=1)
            ob = {"image": torch.as_tensor(img, dtype=torch.float32, device=device).unsqueeze(0)}
            ob["is_first"] = torch.tensor([step==0], dtype=torch.bool, device=device)
            with torch.no_grad():
                a, state = agent.act(ob, state, eval=True)
            r = env.step(a.squeeze(0).cpu().numpy())
            obs = r[0] if isinstance(r, tuple) else r
            rw = float(r[1]) if isinstance(r, tuple) and len(r) > 1 else 0.0
            done = (r[2] or r[3] if isinstance(r, tuple) and len(r) > 3 else False)
            step += 1
            if len(r) > 2 and (r[2] or r[3]): break
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns))

print(f"Evaluating {CKPT.split('/')[-3]}...")
res = {}
for d in ["none", "color_bg", "texture", "shift"]:
    m, s = eval_one(d)
    res[d] = {"mean": m, "std": s}
    print(f"  {d:12s}: {m:8.1f} +/- {s:.1f}")

base = res["none"]["mean"]
res["in_dist_return"] = base
res["method"] = CKPT.split("/")[-3]
for d in ["color_bg", "texture", "shift"]:
    res[f"{d}_retention"] = res[d]["mean"] / (base + 1e-8)
print(f"\nIn-dist: {base:.1f}")
for d in ["color_bg", "texture", "shift"]:
    print(f"  {d}: {res[f'{d}_retention']:.3f}")
with open(OUT, "w") as f:
    json.dump(res, f, indent=2, default=float)
print("Saved", OUT)
