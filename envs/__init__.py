from . import parallel, wrappers
import os


def _egl_init():
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["EGL_DEVICE_ID"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    try:
        from dm_control import _render
    except ImportError:
        pass


def make_envs(config):
    def env_constructor(idx):
        return lambda: _make_env(config, idx)

    if config.env_num == 1:
        from .serial_env import SerialEnv
        train_envs = SerialEnv(env_constructor, 1, config.device)
        eval_envs = SerialEnv(env_constructor, 1, config.device)
    else:
        parallel.Worker.initializers = [_egl_init]
        train_envs = parallel.ParallelEnv(env_constructor, config.env_num, config.device)
        eval_envs = parallel.ParallelEnv(env_constructor, config.eval_episode_num, config.device)
    obs_space = train_envs.observation_space
    act_space = train_envs.action_space
    return train_envs, eval_envs, obs_space, act_space


def _make_env(config, id):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc
        env = dmc.DeepMindControl(task, config.action_repeat, config.size, seed=config.seed + id)
        env = wrappers.NormalizeActions(env)
        if bool(getattr(config, "use_depth_obs", False)):
            from depth_obs import DepthRenderWrapper
            env = DepthRenderWrapper(env)
        use_domain_rand = bool(getattr(config, "use_domain_rand", False))
        if use_domain_rand:
            from envs.randomized_dynamics import RandomizedDynamicsWrapper
            nominal_prob = float(getattr(config, "nominal_prob", getattr(config, "randomize_dynamics_prob", 0.7)))
            env = RandomizedDynamicsWrapper(env, nominal_prob=nominal_prob)
    elif suite == "atari":
        import envs.atari as atari
        env = atari.Atari(task, config.action_repeat, config.size, gray=config.gray, noops=config.noops, lives=config.lives, sticky=config.sticky, actions=config.actions, length=config.time_limit, pooling=config.pooling, aggregate=config.aggregate, resize=config.resize, autostart=config.autostart, clip_reward=config.clip_reward, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze
        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter
        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "metaworld":
        import envs.metaworld as metaworld
        env = metaworld.MetaWorld(task, config.action_repeat, config.size, config.camera, config.seed + id)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit // config.action_repeat)
    return wrappers.Dtype(env)
