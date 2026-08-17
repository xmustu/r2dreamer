"""OOD eval - working version with correct env interface."""
import sys, os, torch, numpy as np, json
os.environ['MUJOCO_GL'] = 'egl'
torch.set_float32_matmul_precision('high')

import sf_rssm, dreamer
from envs import make_env
from omegaconf import OmegaConf
from gymnasium import spaces

CKPT = sys.argv[1]; CFG = sys.argv[2]; DEV = sys.argv[3]; OUT = sys.argv[4]; EPISODES = int(sys.argv[5])

cfg = OmegaConf.load(CFG); cfg.model.device = DEV
device = torch.device(DEV)
ckpt = torch.load(CKPT, map_location=DEV)

env0 = make_env(cfg.env, 0)
obs_space = env0.observation_space
act_dim = env0.action_space.shape[0]
act_space = spaces.Box(-1, 1, (act_dim,))

agent = dreamer.Dreamer(cfg.model, obs_space, act_space).to(device)
agent.load_state_dict(ckpt['agent_state_dict'], strict=False)
agent.eval()

def eval_one(dist_type, episodes=EPISODES):
    returns = []
    for ep in range(episodes):
        env = make_env(cfg.env, 0)
        obs = env.reset()
        state = agent.get_initial_state(B=1)
        total = 0.0
        for step in range(500):
            img = obs['image'].copy()
            if dist_type == 'color_bg':
                rng = np.random.RandomState(42*10000 + ep*100 + step)
                bg = img.mean(-1) > 0.55
                c = rng.randint(0, 256, 3).astype(np.float32) / 255.0
                img[bg] = 0.6*img[bg] + 0.4*c
            elif dist_type == 'texture':
                rng = np.random.RandomState(42*10000 + ep*100 + step)
                img = np.clip(img + rng.randn(*img.shape).astype(np.float32)*0.05, 0, 1)
            elif dist_type == 'shift':
                rng = np.random.RandomState(42*10000 + ep*100 + step)
                img = np.roll(img, rng.randint(-5, 6), axis=1)
            ob = {'image': torch.as_tensor(img, dtype=torch.float32, device=device).unsqueeze(0)}
            ob['is_first'] = torch.tensor([step == 0], dtype=torch.bool, device=device)
            with torch.no_grad():
                a, state = agent.act(ob, state, eval=True)
            obs, reward, term, info = env.step(a.squeeze(0).cpu().numpy())
            total += float(reward)
            if term: break
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns))

name = "dreamerv3" if "dv3" in CKPT else "sf_rssm"
print(f'{name} OOD:')
res = {}
for d in ['none','color_bg','texture','shift']:
    m,s = eval_one(d)
    res[d] = {'mean':m,'std':s}
    print(f'  {d:12s}: {m:8.1f} +/- {s:.1f}')

base = res['none']['mean']
res['in_dist_return'] = base
res['method'] = name
for d in ['color_bg','texture','shift']:
    res[f'{d}_retention'] = res[d]['mean'] / (base + 1e-8)
    ret = res[f"{d}_retention"]; print(f"    {d} retention: {ret:.3f}")

print(f'In-dist: {base:.1f}')
with open(OUT, 'w') as f:
    json.dump(res, f, indent=2, default=float)
print(f'Saved {OUT}')
