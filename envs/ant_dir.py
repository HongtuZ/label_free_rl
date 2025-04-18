import numpy as np
from gymnasium.envs.mujoco.ant_v5 import AntEnv


class AntDirEnv(AntEnv):

    def __init__(self, task, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = task
        self.direction = np.array((np.cos(task), np.sin(task)), dtype=np.float32)

    def step(self, action):
        xy_position_before = self.data.body(self._main_body).xpos[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.body(self._main_body).xpos[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, y_velocity, action)
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            **reward_info,
        }
        info["task"] = self.task

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, False, False, info

    def _get_rew(self, x_velocity: float, y_velocity: float, action):
        forward_reward = (
            np.dot(self.direction, np.array((x_velocity, y_velocity)))
            * self._forward_reward_weight
        )
        healthy_reward = self.healthy_reward
        rewards = forward_reward + healthy_reward

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        costs = ctrl_cost + contact_cost

        reward = rewards - costs

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_survive": healthy_reward,
        }

        return reward, reward_info

    def reset(self, *, seed=None, options=None):
        task = options.get("task", None) if options else None
        obs, info = super().reset(seed=seed, options=options)
        if task is not None:
            self.task = task
            self.direction = np.array((np.cos(task), np.sin(task)), dtype=np.float32)
        info["task"] = self.task
        return obs, info
