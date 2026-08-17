import os, numpy as np
os.environ["MUJOCO_GL"] = "egl"
import stable_worldmodel as swm

w = swm.World(env_name="swm/PushT-v1", num_envs=1, image_shape=(224, 224), max_episode_steps=300)
raw = w.envs.envs[0]
obs, info = raw.reset()
print("Obs keys:", list(obs.keys()))
for k, v in obs.items():
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
print("Info keys:", list(info.keys()) if isinstance(info, dict) else type(info))
raw.close()
