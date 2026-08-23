"""
DepthRenderWrapper: 为 DMControl 环境增加深度图 + 接触掩码观测（新文件）。

每步 step/reset 时：
- obs["depth"]：MuJoCo 离屏渲染深度图（米），float32 (H, W)
- obs["contact"]：二值接触掩码（nbody,）— 来自 physics.data.contact 真值，
  某 body 的任一 geom 处于接触（dist < 0）即为 1

用于 SPER v2 的深度/接触预测头目标（sper_v2）。

代价：每步多一次渲染（RGB + depth），环境步进时间约翻倍。仅在空间感知
实验时启用（use_depth_obs=True）。

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
        nbody = self._dmc_env._env.physics.model.nbody
        spaces["contact"] = gym.spaces.Box(
            low=0.0, high=1.0, shape=(nbody,), dtype=np.float32
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

    def _collect_contact(self):
        """从 physics.data.contact 构造 per-body 二值接触掩码（nbody,）float32。"""
        ph = self._dmc_env._env.physics
        contact = np.zeros(ph.model.nbody, dtype=np.float32)
        ncon = ph.data.ncon
        if ncon > 0:
            g1 = ph.data.contact.geom1[:ncon]
            g2 = ph.data.contact.geom2[:ncon]
            dist = ph.data.contact.dist[:ncon]
            active = dist < 0.0  # 穿透 = 接触中
            bodies = []
            for g in g1[active]:
                bodies.append(ph.model.geom_bodyid[g])
            for g in g2[active]:
                bodies.append(ph.model.geom_bodyid[g])
            for b in bodies:
                contact[int(b)] = 1.0
        return contact

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = dict(obs)  # 拷贝，不修改上游观测字典
        obs["depth"] = self._render_depth().astype(np.float32)
        obs["contact"] = self._collect_contact()
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = dict(obs)
        obs["depth"] = self._render_depth().astype(np.float32)
        obs["contact"] = self._collect_contact()
        return obs
