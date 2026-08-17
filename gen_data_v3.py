"""Generate Push-T HDF5 dataset with rendered pixels and ground-truth states."""
import os, numpy as np, h5py, time
os.environ["MUJOCO_GL"] = "egl"
import stable_worldmodel as swm

w = swm.World(env_name="swm/PushT-v1", num_envs=1, image_shape=(224, 224), max_episode_steps=300)
raw = w.envs.envs[0]

pixels_list, action_list, state_list = [], [], []
ep_idx_list, step_idx_list = [], []
total, n_episodes = 0, 200
t0 = time.time()

for ep in range(n_episodes):
    obs, info = raw.reset()
    for t in range(300):
        act = raw.action_space.sample()
        obs, rew, term, trunc, info = raw.step(act)

        # Render pixels
        rgb = raw.render()
        if rgb is None and "pixels" in info:
            rgb = info["pixels"]
        if rgb is None:
            rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        pixels_list.append(rgb.astype(np.uint8))

        action_list.append(act.astype(np.float32))

        # Ground-truth state: agent_pos + block_pose + goal_pose from info
        pos_agent = info.get("pos_agent", np.zeros(2))
        block_pose = info.get("block_pose", np.zeros(3))
        goal_pose = info.get("goal_pose", np.zeros(2))
        st = np.concatenate([
            pos_agent.flatten()[:2],
            block_pose.flatten()[:3],
            goal_pose.flatten()[:2],
        ]).astype(np.float32)
        if len(st) < 7:
            st = np.pad(st, (0, max(0, 7 - len(st))), mode='constant')
        st = st[:7]
        state_list.append(st)

        ep_idx_list.append(ep)
        step_idx_list.append(t)
        total += 1

        if term or trunc:
            break

    if (ep + 1) % 25 == 0:
        elapsed = time.time() - t0
        print(f"Ep {ep+1}/{n_episodes}, {total} steps, {elapsed:.0f}s")

raw.close()

# Save HDF5
path = "/home/zhengkai/.stable-wm/pusht_random_train.h5"
size_mb_total = 0
for arr in pixels_list:
    size_mb_total += arr.nbytes / 1024**2
print(f"Array size estimate: {size_mb_total:.0f} MB")

pixels_arr = np.stack(pixels_list)
action_arr = np.stack(action_list)
state_arr = np.stack(state_list)

with h5py.File(path, "w") as f:
    f.create_dataset("pixels", data=pixels_arr,
                     compression="gzip", compression_opts=4,
                     chunks=(1, 224, 224, 3))
    f.create_dataset("action", data=action_arr,
                     compression="gzip", compression_opts=4)
    f.create_dataset("state", data=state_arr,
                     compression="gzip", compression_opts=4)
    f.create_dataset("proprio", data=state_arr[:, :4],
                     compression="gzip", compression_opts=4)
    f.create_dataset("episode_idx", data=np.array(ep_idx_list, dtype=np.int32))
    f.create_dataset("step_idx", data=np.array(step_idx_list, dtype=np.int32))

size_mb = os.path.getsize(path) / 1024**2
print(f"Done! {total} steps, {size_mb:.1f} MB, {time.time()-t0:.0f}s")
print(f"Saved: {path}")
