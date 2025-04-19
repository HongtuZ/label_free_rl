import os

import gymnasium as gym
import numpy as np
import torch
import yaml
from gymnasium.wrappers import Autoreset, RecordVideo

import envs
import rl.pytorch_utils as ptu

os.environ["MUJOCO_GL"] = "osmesa"  # 使用 osmesa 渲染
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

if __name__ == "__main__":
    env = Autoreset(envs.make_env("point-robot", 0.0)())
    obs, _ = env.reset()
    for i in range(22):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print("terminated:", terminated, "truncated:", truncated)
        print(info)
