"""
Randomized Dynamics Wrapper for DreamerV3.

Wraps a DMC environment and randomly perturbs physics parameters
at episode reset for domain randomization. Tracks episode mode
via "dynamics_mode" observation key.

Default: 70% nominal episodes, 30% with 1-3 randomized physics parameters.
"""

import numpy as np
import gymnasium as gym
from envs.dmc import DeepMindControl


class RandomizedDynamicsWrapper(gym.Wrapper):
    """Env wrapper that randomly perturbs dynamics parameters on reset.

    Adds "dynamics_mode" (1.0 = randomized, 0.0 = nominal) to each observation.
    Inherits from gymnasium.Wrapper for compatibility with other wrappers.
    """

    def __init__(self, env, nominal_prob=0.7):
        super().__init__(env)
        self._nominal_prob = nominal_prob
        self._randomized = False
        self._rng = np.random.RandomState()
        self._baseline_physics = None  # Saved on first reset

        # Add dynamics_mode to observation space
        orig_space = env.observation_space
        new_spaces = dict(orig_space.spaces)
        new_spaces["dynamics_mode"] = gym.spaces.Box(0, 1, (1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(new_spaces)

    def _save_baseline_physics(self):
        """Save a deep copy of current physics parameters as the nominal baseline."""
        try:
            physics = self._physics
        except AttributeError:
            return
        import copy
        self._baseline_physics = {
            "geom_friction": physics.model.geom_friction.copy(),
            "body_mass": physics.model.body_mass.copy(),
            "dof_damping": physics.model.dof_damping.copy(),
            "actuator_gainprm": physics.model.actuator_gainprm.copy(),
            "gravity": physics.model.opt.gravity.copy(),
        }

    def _restore_baseline_physics(self):
        """Restore all physics parameters to the saved nominal baseline."""
        if self._baseline_physics is None:
            return
        try:
            physics = self._physics
        except AttributeError:
            return
        physics.model.geom_friction[:] = self._baseline_physics["geom_friction"]
        physics.model.body_mass[:] = self._baseline_physics["body_mass"]
        physics.model.dof_damping[:] = self._baseline_physics["dof_damping"]
        physics.model.actuator_gainprm[:] = self._baseline_physics["actuator_gainprm"]
        physics.model.opt.gravity[:] = self._baseline_physics["gravity"]

    @property
    def _physics(self):
        """Find MuJoCo physics through the wrapper chain (self.env points to DMC)."""
        # self.env is the inner env; follow chain to find DeepMindControl
        obj = self.env
        while hasattr(obj, 'env') and not isinstance(obj, DeepMindControl):
            obj = obj.env
        if isinstance(obj, DeepMindControl):
            return obj._env.physics
        raise AttributeError("Could not find MuJoCo physics in wrapper chain")

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # Save baseline physics on first call
        if self._baseline_physics is None:
            self._save_baseline_physics()
        # Restore nominal physics BEFORE deciding whether to perturb
        self._restore_baseline_physics()
        self._randomized = self._rng.random_sample() > self._nominal_prob
        if self._randomized:
            self._apply_random_dynamics()
        obs["dynamics_mode"] = np.float32([1.0 if self._randomized else 0.0])
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs["dynamics_mode"] = np.float32([1.0 if self._randomized else 0.0])
        return obs, reward, done, info

    def _apply_random_dynamics(self):
        """Randomly perturb 1-3 physics parameter types."""
        try:
            physics = self._physics
        except AttributeError:
            return  # Silently skip if physics not accessible
        pert_fns = [
            lambda: self._perturb_friction(physics),
            lambda: self._perturb_mass(physics),
            lambda: self._perturb_damping(physics),
            lambda: self._perturb_actuator(physics),
            lambda: self._perturb_gravity(physics),
        ]
        n_pert = self._rng.randint(1, 4)
        indices = self._rng.choice(len(pert_fns), n_pert, replace=False)
        for idx in indices:
            pert_fns[idx]()

    def _perturb_friction(self, physics):
        scale = np.exp(self._rng.uniform(np.log(0.5), np.log(1.5)))
        friction = physics.model.geom_friction.copy()
        friction[:, 0] *= scale
        physics.model.geom_friction[:] = friction

    def _perturb_mass(self, physics):
        scale = np.exp(self._rng.uniform(np.log(0.5), np.log(1.5)))
        mass = physics.model.body_mass.copy()
        mass[1:] *= scale  # Skip world body (index 0)
        physics.model.body_mass[:] = mass

    def _perturb_damping(self, physics):
        scale = np.exp(self._rng.uniform(np.log(0.5), np.log(2.0)))
        damping = physics.model.dof_damping.copy()
        damping *= scale
        physics.model.dof_damping[:] = damping

    def _perturb_actuator(self, physics):
        scale = np.exp(self._rng.uniform(np.log(0.7), np.log(1.3)))
        gain = physics.model.actuator_gainprm.copy()
        gain[:, 0] *= scale
        physics.model.actuator_gainprm[:] = gain

    def _perturb_gravity(self, physics):
        scale = np.exp(self._rng.uniform(np.log(0.7), np.log(1.3)))
        gravity = physics.model.opt.gravity.copy()
        gravity[2] *= scale
        physics.model.opt.gravity[:] = gravity
