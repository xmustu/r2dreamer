"""Distracting Control Suite (Stone et al., 2021) environment wrapper.

Integrates DCS with R2-Dreamer's environment interface.
Supports: dynamic background, color, and camera distractions.
"""

import os

import gymnasium as gym
import numpy as np


class DistractingControl(gym.Env):
    """Wrapper around distracting_control.suite to match R2-Dreamer's DMC interface."""

    metadata = {}

    def __init__(
        self,
        name,
        action_repeat=1,
        size=(64, 64),
        camera=None,
        seed=0,
        difficulty="easy",
        distraction_types=("background",),
        dynamic=True,
        background_dataset_path=None,
    ):
        # Parse name format: "domain_task" e.g., "cheetah_run"
        domain, task = name.rsplit("_", 1)

        # Resolve background dataset path
        if background_dataset_path is None:
            # Default DAVIS paths; uses environment variable as override
            background_dataset_path = os.environ.get(
                "DAVIS_PATH",
                os.path.expanduser("~/datasets/DAVIS/JPEGImages/480p"),
            )

        # Build distraction kwargs from difficulty
        self._domain = domain
        self._task = task
        self._difficulty = difficulty
        self._dynamic = dynamic

        # Import DCS suite (lazy import to avoid dependency errors if not installed)
        from distracting_control import suite

        # Configure which distractions are active
        bg_kwargs = None
        color_kwargs = None
        camera_kwargs = None

        if "background" not in distraction_types:
            bg_kwargs = {"scale": 0.0}
        if "color" not in distraction_types:
            color_kwargs = {"scale": 0.0}
        if "camera" not in distraction_types:
            camera_kwargs = {"scale": 0.0}

        self._env = suite.load(
            domain_name=domain,
            task_name=task,
            difficulty=difficulty if difficulty != "none" else None,
            dynamic=dynamic,
            background_dataset_path=background_dataset_path,
            background_dataset_videos="train",
            background_kwargs=bg_kwargs,
            color_kwargs=color_kwargs,
            camera_kwargs=camera_kwargs,
            task_kwargs={"random": seed},
            pixels_only=True,
        )

        self._action_repeat = action_repeat
        self._size = size
        if camera is None:
            camera = dict(quadruped=2, fish=3).get(domain, 0)
        self._camera = camera
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            if len(value.shape) == 0:
                shape = (1,)
            else:
                shape = value.shape
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)
        spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            if time_step.last():
                break
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        obs["image"] = self._render()
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        obs["is_last"] = time_step.last()
        done = time_step.last()
        info = {"discount": np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self, **kwargs):
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        obs["image"] = self._render()
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        obs["is_last"] = time_step.last()
        return obs

    def _render(self):
        return self._env.physics.render(*self._size, camera_id=self._camera)
