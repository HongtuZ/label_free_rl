import multiprocessing
import os
from pathlib import Path

import click
import gymnasium as gym
import numpy as np
import torch
import yaml
from gymnasium.wrappers import RecordVideo

import envs
import rl.pytorch_utils as ptu
from rl.algorithm.ppo import PPOAgent, PPOArgs, PPORunner

os.environ["MUJOCO_GL"] = "osmesa"  # 使用 osmesa 渲染
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"


def train_task(config, task_idx):
    ptu.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up the environment
    env_args = config["EnvArgs"]
    env_id = env_args["env_id"]
    task = env_args["tasks"][task_idx]
    num_train_envs = env_args["num_train_envs"]
    num_eval_envs = env_args["num_eval_envs"]
    vec_envs = gym.vector.SyncVectorEnv(
        [envs.make_env(env_id, task) for i in range(num_train_envs)]
    )
    eval_envs = gym.vector.SyncVectorEnv(
        [envs.make_env(env_id, task) for i in range(num_eval_envs)]
    )

    # Set up PPO args
    ppo_args = PPOArgs(**config["PPOArgs"])
    ppo_args.log_dir = f"{ppo_args.log_dir}/task_{task}"
    PPORunner(vec_envs, eval_envs, ppo_args, task_idx).train()


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--config_path", required=True, type=click.Path(exists=True, dir_okay=False)
)
@click.option(
    "--checkpoint_path", required=True, type=click.Path(exists=True, dir_okay=False)
)
def show_policy(config_path, checkpoint_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    checkpoint_path = Path(checkpoint_path)
    ptu.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_id = checkpoint_path.parent.parent.parent.name
    task = float(checkpoint_path.parent.parent.name.split("_")[-1])
    env = RecordVideo(
        envs.make_env(env_id, task=task, render_mode="rgb_array")(),
        str(checkpoint_path.parent),
        episode_trigger=lambda x: True,
        name_prefix=checkpoint_path.stem,
    )

    agent = PPOAgent.load_from(
        str(checkpoint_path),
        config["PPOArgs"]["actor_hidden_dims"],
        config["PPOArgs"]["critic_hidden_dims"],
    ).to(ptu.device)

    obs, _ = env.reset()
    for _ in range(config["PPOArgs"]["num_steps"]):
        action = agent.get_action(ptu.from_numpy(obs), deterministic=True)
        obs, _, _, _, _ = env.step(ptu.get_numpy(action).flatten())
    env.close()
    print(f"Video saved to {checkpoint_path.parent}")


@cli.command()
@click.option(
    "--config_path", required=True, type=click.Path(exists=True, dir_okay=False)
)
def train(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    tasks = list(eval(str(config["EnvArgs"]["tasks"])))
    config["EnvArgs"]["tasks"] = tasks
    with multiprocessing.Pool(processes=3) as pool:
        results = pool.starmap(
            train_task, [(config, task_idx) for task_idx in range(len(tasks))]
        )


if __name__ == "__main__":
    cli()
