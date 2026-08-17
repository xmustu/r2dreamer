"""Serial environment wrapper — drop-in replacement for ParallelEnv without multiprocessing."""
import numpy as np
import torch
from tensordict import TensorDict

class SerialEnv:
    def __init__(self, constructor, env_num, device):
        assert env_num == 1, "SerialEnv only supports env_num=1"
        self._env = constructor(0)()
        self.device = device
        # Regular attributes (not properties) for compatibility
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.env_num = 1

    def step(self, action, done):
        if done[0]:
            obs = self._env.reset()
            reward = 0.0
            new_done = False
        else:
            act_np = action.cpu().numpy() if hasattr(action, "cpu") else action
            obs, reward, new_done, _ = self._env.step(act_np[0])
        obs = {k: np.expand_dims(v, 0) if isinstance(v, np.ndarray) else np.array([[v]]) for k, v in obs.items()}
        obs_tensors = {k: torch.as_tensor(v, device="cpu") for k, v in obs.items()}
        td = TensorDict({**obs_tensors, "reward": torch.tensor([reward], dtype=torch.float32, device="cpu")}, batch_size=(1,), device="cpu").pin_memory()
        d = torch.tensor([new_done], device="cpu")
        return self._lift_dim(td), d

    def _lift_dim(self, td):
        for key in td.keys():
            if td[key].ndim == 1:
                td[key] = td[key].unsqueeze(-1)
        return td

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()
