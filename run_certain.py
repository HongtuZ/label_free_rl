import multiprocessing
import os
import time
from pathlib import Path

import click
import gymnasium as gym
import numpy as np
import torch
import yaml
from gymnasium.wrappers import RecordVideo
from torch.utils.data import DataLoader

import envs
import rl.pytorch_utils as ptu
from rl.algorithm.certain import CERTAINArgs, CERTAINRunner

os.environ["MUJOCO_GL"] = "osmesa"  # 使用 osmesa 渲染

@click.group()
def cli():
    pass


@cli.command()
@click.option("--config_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--checkpoint_path", required=True, type=click.Path(exists=True, dir_okay=False))
def show_policy(config_path, checkpoint_path):
    pass


@cli.command()
@click.option("--config_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--gpu", required=False, default=0, type=int, help="GPU ID to use")
@click.option("--baseline", required=True, help="focal, classifier, unicorn")
@click.option("--certain_beta", required=False, default=0.1, type=float, help="certain beta")
def train(config_path, gpu, baseline, certain_beta):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    ptu.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    # Set up the environment
    env_args = config["EnvArgs"]
    env_id = env_args["env_id"]
    # This is just to make the env, the task will be set in the eval
    tasks = list(eval(str(config["EnvArgs"]["tasks"])))
    num_eval_envs = env_args["num_eval_envs"]
    eval_envs = gym.vector.SyncVectorEnv(
        [envs.make_env(env_id, tasks[0]) for i in range(num_eval_envs)]
    )

    # Set up CERTAIN args
    certain_args = CERTAINArgs(**config["CERTAINArgs"])
    certain_args.context_agent_type = baseline
    certain_args.certain_beta = certain_beta
    method_name = certain_args.context_agent_type
    method_name += f"-certain" if certain_args.enable_certain else ""
    certain_args.log_dir = str(
        Path(certain_args.log_dir)
        / method_name
        / f"{certain_beta}"
        / f'{time.strftime("%Y%m%d_%H%M%S", time.localtime())}'
    )
    CERTAINRunner(eval_envs, certain_args).train()


if __name__ == "__main__":
    cli()
