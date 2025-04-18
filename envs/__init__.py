import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register

register(id="ant-dir", entry_point="envs.ant_dir:AntDirEnv")

register(id="point-robot", entry_point="envs.point_robot:PointEnv")


def make_env(env_name, task, *args, **kwargs):
    def thunck():
        if env_name == "ant-dir":
            kwargs.pop("include_cfrc_ext_in_observation", None)
            kwargs.pop("terminate_when_unhealthy", None)
            env = gym.make(
                "ant-dir",
                task=task,
                include_cfrc_ext_in_observation=False,
                terminate_when_unhealthy=False,
                *args,
                **kwargs
            )
        elif env_name == "point-robot":
            env = gym.make("point-robot", task=task, *args, **kwargs)
        else:
            env = gym.make(env_name, *args, **kwargs)
        env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)
        env = gym.wrappers.FlattenObservation(
            env
        )  # deal with dm_control's Dict observation space
        # env = gym.wrappers.NormalizeObservation(env)
        return env

    return thunck
