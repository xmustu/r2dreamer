"""Generate Push-T HDF5 dataset via gymnasium directly (bypasses stable_worldmodel.World).

Format matches what LeWorldModel expects for training.
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import gymnasium as gym
import numpy as np
import h5py
import stable_worldmodel  # registers swm/PushT-v1


def main():
    output = "/home/zhengkai/.stable-wm/pusht_random_train.h5"
    n_episodes = 200
    max_steps = 300

    env = gym.make("swm/PushT-v1", max_episode_steps=max_steps)

    pixels_list = []
    action_list = []
    state_list = []
    ep_idx_list = []
    step_idx_list = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        for t in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            pixels_list.append(obs["pixels"].astype(np.uint8))

            action_list.append(action.astype(np.float32))

            # ground-truth state
            st = np.concatenate([
                obs["agent_pos"].flatten(),
                obs["block_pos"].flatten(),
                obs["goal_pos"].flatten(),
            ]).astype(np.float32)
            state_list.append(st)

            ep_idx_list.append(ep)
            step_idx_list.append(t)

            if terminated or truncated:
                break
        if (ep + 1) % 25 == 0:
            print(f"Ep {ep+1}/{n_episodes}")

    env.close()

    total = len(pixels_list)
    print(f"Total steps: {total}, Avg ep: {total/n_episodes:.1f}")

    pixels_arr = np.stack(pixels_list)
    action_arr = np.stack(action_list)
    state_arr = np.stack(state_list)

    with h5py.File(output, "w") as f:
        f.create_dataset("pixels", data=pixels_arr,
                         compression="gzip", compression_opts=2)
        f.create_dataset("action", data=action_arr,
                         compression="gzip", compression_opts=2)
        f.create_dataset("state", data=state_arr,
                         compression="gzip", compression_opts=2)
        f.create_dataset("proprio", data=state_arr[:, :4],
                         compression="gzip", compression_opts=2)
        f.create_dataset("episode_idx", data=np.array(ep_idx_list, dtype=np.int32))
        f.create_dataset("step_idx", data=np.array(step_idx_list, dtype=np.int32))

    print(f"Saved: {output} ({os.path.getsize(output)/1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
