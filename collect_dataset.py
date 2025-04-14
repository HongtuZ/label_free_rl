import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import RecordVideo

import envs
import rl.pytorch_utils as ptu
from rl.algorithm.ppo import PPOAgent, PPOArgs, PPORunner

os.environ["MUJOCO_GL"] = "osmesa"  # 使用 osmesa 渲染
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


def train():
    ptu.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up the environment
    env_nums = 2
    env_id = "point-robot"
    task = np.pi / 2
    vec_envs = gym.vector.AsyncVectorEnv(
        [envs.make_env(env_id, task=task) for i in range(env_nums)]
    )
    eval_env = envs.make_env(env_id, task=task)()

    # Set up PPO args
    log_dir = f"logs/{env_id}/task_{task}"
    args = PPOArgs(
        log_dir=log_dir,
        seed=1,
        num_steps=20,
        num_iterations=500,
        eval_interval=5,
        save_interval=50,
    )
    runner = PPORunner(vec_envs, eval_env, args)
    runner.train()


def visualize_policy():
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    ptu.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = envs.make_env("point-robot", render_mode="rgb_array")()

    agent_path = Path(
        "runs/point-robot__ppo_continuous_action__1__1744598594/ppo_continuous_action_10000.cleanrl_model"
    )

    agent = PPOAgent.load_from(str(agent_path)).to(ptu.device)

    env = RecordVideo(
        env,
        str(agent_path.parent),
        episode_trigger=lambda x: True,
        name_prefix=f"{agent_path.stem}_video",
    )
    obs, _ = env.reset()
    for _ in range(20):
        action = agent.get_action(
            ptu.from_numpy(obs).unsqueeze(0).to(ptu.device), deterministic=True
        )
        obs, _, _, _, _ = env.step(action.cpu().numpy().flatten())
    env.close()


def test_env():
    vec_envs = gym.vector.SyncVectorEnv(
        [envs.make_env("point-robot") for i in range(2)]
    )
    obs, info = vec_envs.reset(options={"task": np.pi / 3})
    print(vec_envs.get_attr("task"))
    for _ in range(20):
        action = vec_envs.action_space.sample()
        obs, r, _, _, info = vec_envs.step(action)
        print(obs, r, info)
    vec_envs.close()


if __name__ == "__main__":
    train()
    # visualize_policy()
    # test_env()
