"""Convert HDF5 dataset to Lance format for LeWorldModel training."""
import os, numpy as np, h5py, shutil
os.environ["MUJOCO_GL"] = "egl"
from stable_worldmodel.data.formats.lance import LanceWriter
import inspect

# Check signature
print("write_episode signature:", inspect.signature(LanceWriter.write_episode))

src = "/home/zhengkai/.stable_worldmodel/datasets/pusht_random_train.h5"
dst = "/home/zhengkai/.stable_worldmodel/datasets/pusht_random_train.lance"
if os.path.exists(dst):
    shutil.rmtree(dst)

print(f"Reading {src}...")
with h5py.File(src, "r") as f:
    n = len(f["pixels"])
    n_eps = int(f["episode_idx"][-1]) + 1
    print(f"  {n} rows, {n_eps} episodes")

    # Group by episode
    ep_data = {}
    for i in range(n):
        ep = int(f["episode_idx"][i])
        if ep not in ep_data:
            ep_data[ep] = {"pixels": [], "action": [], "state": [], "proprio": [], "step_idx": []}
        ep_data[ep]["pixels"].append(f["pixels"][i])
        ep_data[ep]["action"].append(f["action"][i])
        ep_data[ep]["state"].append(f["state"][i])
        ep_data[ep]["proprio"].append(f["proprio"][i])
        ep_data[ep]["step_idx"].append(f["step_idx"][i])

    # Write each episode using context manager
    with LanceWriter(dst) as writer:
        for ep in range(n_eps):
            d = ep_data[ep]
            episode = {
                "pixels": np.stack(d["pixels"]),
                "action": np.stack(d["action"]),
                "state": np.stack(d["state"]),
                "proprio": np.stack(d["proprio"]),
                "step_idx": np.array(d["step_idx"]),
                "episode_idx": np.full(len(d["step_idx"]), ep),
            }
            writer.write_episode(episode)
            if (ep + 1) % 50 == 0:
                print(f"  Ep {ep+1}/{n_eps}")

print(f"Done: {dst}")
