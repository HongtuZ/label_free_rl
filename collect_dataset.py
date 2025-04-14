import os
from rl.algorithm.ppo import PPOArgs, PPOAgent, PPORunner
import torch
import rl.pytorch_utils as ptu
import numpy as np
from gymnasium.wrappers import RecordVideo
from pathlib import Path
import gymnasium as gym
import envs

os.environ["MUJOCO_GL"] = "osmesa"  # 使用 osmesa 渲染

def train():
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    ptu.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set up the environment and PPO agent
    env = envs.make_env('ant-dir')
    env.set_task_idx(0)
    args = PPOArgs()
    args.seed = 42
    runner = PPORunner(env, args)
    runner.train()

def visualize_policy():
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    ptu.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = envs.make_env('point-robot', render_mode='rgb_array')()

    agent_path = Path('runs/ant-dir__ppo_continuous_action__1__1744190946/ppo_continuous_action_999936.cleanrl_model')

    agent = PPOAgent.load_from(str(agent_path)).to(ptu.device)

    env = RecordVideo(env, str(agent_path.parent), episode_trigger=lambda x: True, name_prefix=f'{agent_path.stem}_video')
    obs, _ = env.reset()
    for _ in range(256):
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(ptu.from_numpy(obs))
        obs, _, _, _, _ = env.step(action.cpu().numpy().flatten())
    env.close()

def test_env():
    gym.pprint_registry()
    env = envs.make_env('ant-dir', render_mode='rgb_array')()
    env = RecordVideo(env, 'videos', episode_trigger=lambda x: True, name_prefix=f'test_video')
    obs, info = env.reset()
    for _ in range(20):
        action = env.action_space.sample()
        obs, r, _, _, info = env.step(action)
        print(obs, r, info)
    env.close()

if __name__ == "__main__":
    # train()
    # visualize_policy()
    test_env()
