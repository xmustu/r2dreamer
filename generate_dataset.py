"""Generate Push-T dataset for LeWorldModel training using MuJoCo environment."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import h5py
import stable_worldmodel as swm


def generate_dataset(
    output_path: str,
    n_episodes: int = 200,
    max_steps: int = 300,
):
    world = swm.World(
        env_name="swm/PushT-v1",
        num_envs=1,
        max_episode_steps=max_steps,
        image_shape=(224, 224),
    )

    pixels_list = []
    action_list = []
    state_list = []
    ep_idx_list = []
    step_idx_list = []

    for ep in range(n_episodes):
        obs, info = world.reset()
        ep_steps = 0
        while ep_steps < max_steps:
            action = world.action_space.sample()
            obs, reward, terminated, truncated, info = world.step(action)
            done = terminated or truncated

            if isinstance(obs, dict):
                img = obs.get("pixels", obs.get("image"))
                if img.ndim == 4:
                    img = img[0]  # remove batch dim
                pixels_list.append(img.astype(np.uint8))
                act = action
                if act.ndim == 2:
                    act = act[0]
                action_list.append(act.astype(np.float32))
                # Use dummy state if not available
                st = obs.get("state", np.zeros(6, dtype=np.float32))
                if st.ndim == 2:
                    st = st[0]
                state_list.append(st.astype(np.float32))
            else:
                pixels_list.append(obs[0].astype(np.uint8))
                action_list.append(action[0].astype(np.float32))
                state_list.append(np.zeros(6, dtype=np.float32))

            ep_idx_list.append(ep)
            step_idx_list.append(ep_steps)
            ep_steps += 1

            if done.any() if hasattr(done, 'any') else done:
                break

        if (ep + 1) % 20 == 0:
            print(f"  Episode {ep+1}/{n_episodes}: {ep_steps} steps")

    total = len(pixels_list)
    print(f"Total: {total} steps, Avg ep: {total/n_episodes:.1f}")

    # Use uint8 to save space
    pixels_arr = np.array(pixels_list, dtype=np.uint8)
    action_arr = np.array(action_list, dtype=np.float32)
    state_arr = np.array(state_list, dtype=np.float32)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("pixels", data=pixels_arr, compression="gzip", compression_opts=2)
        f.create_dataset("action", data=action_arr, compression="gzip", compression_opts=2)
        f.create_dataset("state", data=state_arr, compression="gzip", compression_opts=2)
        f.create_dataset("proprio", data=state_arr[:, :4], compression="gzip", compression_opts=2)
        f.create_dataset("episode_idx", data=np.array(ep_idx_list, dtype=np.int32))
        f.create_dataset("step_idx", data=np.array(step_idx_list, dtype=np.int32))

    size_mb = os.path.getsize(output_path) / 1024**2
    print(f"Saved: {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="/home/zhengkai/.stable-wm/pusht_random_train.h5")
    p.add_argument("--n_episodes", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=300)
    args = p.parse_args()
    generate_dataset(args.output, args.n_episodes, args.max_steps)
