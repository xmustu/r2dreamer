"""
DepthRenderWrapper: 为 DMControl 环境增加深度图观测（新文件，不影响现有 envs/）。

在每步 step/reset 时通过 MuJoCo 离屏渲染深度图（physics.render(depth=True)），
加入 obs["depth"]。深度值 = 距相机平面的距离（米），float32 (H, W)。

用于 SPER v2 的深度预测头目标（sper_v2.compute_depth_target）。

代价：每步多一次渲染（RGB + depth），环境步进时间约翻倍。仅在深度头实验
（M3 消融/主实验）时启用。

约定：本代码库使用旧 Gym API（4-tuple step: obs, reward, done, info；
reset 返回 obs）— 与 envs/wrappers.py 的 TimeLimit/NormalizeActions 一致。

Codex 评审修复（2026-08-15）：
- [CRITICAL→FIXED] _find_dmc：按类名精确匹配 DeepMindControl 并验证
  cur._env.physics 存在（DeepMindControl 本身没有 .physics，physics 在
  其 _env（dm_control 实例）上）
- [MAJOR→FIXED] obs dict 原地修改 → dict(obs) 拷贝后加入 depth
"""

import numpy as np
import gymnasium as gym


class DepthRenderWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # 找到底层 DeepMindControl 实例（envs/dmc.py）
        dmc_env = self._find_dmc(env)
        self._dmc_env = dmc_env
        self._camera = getattr(dmc_env, "_camera", 0)
        self._size = getattr(dmc_env, "_size", (64, 64))

        spaces = dict(self.env.observation_space.spaces)
        h, w = self._size
        spaces["depth"] = gym.spaces.Box(
            low=0.0, high=np.inf, shape=(h, w), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(spaces)

    @staticmethod
    def _find_dmc(env):
        """Walk the wrapper chain to find the DeepMindControl instance.

        链结构（_make_env）：DepthRenderWrapper → NormalizeActions → DeepMindControl。
        DeepMindControl 持有 self._env = dm_control 环境实例；
        physics 在 self._env.physics 上，而非 DeepMindControl 本身。
        """
        cur = env
        while cur is not None:
            if type(cur).__name__ == "DeepMindControl":
                if hasattr(cur, "_env") and hasattr(cur._env, "physics"):
                    return cur
            cur = getattr(cur, "env", None)
        raise ValueError("DepthRenderWrapper must wrap a DeepMindControl env")

    def _render_depth(self):
        # 与 envs/dmc.py DeepMindControl.render 相同的底层调用 + depth=True
        return self._dmc_env._env.physics.render(
            *self._size, camera_id=self._camera, depth=True
        )

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = dict(obs)  # 拷贝，不修改上游观测字典
        obs["depth"] = self._render_depth().astype(np.float32)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = dict(obs)
        obs["depth"] = self._render_depth().astype(np.float32)
        return obs
