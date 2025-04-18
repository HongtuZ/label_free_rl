import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import rl.pytorch_utils as ptu
from rl.net.agent import (
    CertainAgent,
    ClassifierAgent,
    FocalAgent,
    TD3BCAgent,
    UnicornAgent,
)
from rl.net.mlp import MLP


@dataclass
class CERTAINArgs:
    # --------------------------------------------------------
    #                      Training setting
    # --------------------------------------------------------
    log_dir: str = "logs/certain"
    dataset_dir: str = None
    seed: int = 1
    batch_size: int = 256  # batch size for training
    num_iterations: int = 100  # total training iterations
    eval_num_steps: int = 256  # number of eval env rollout steps
    eval_interval: int = 1  # interval iteratoins to eval the policy
    save_interval: int = 10  # interval iterations to save the policy
    # --------------------------------------------------------
    #                      Context agent
    # --------------------------------------------------------
    use_next_observation: bool = True  # use next observation in context
    latent_dim: int = 20
    encoder_hidden_dims: list = None
    encoder_learning_rate: float = 1e-3
    context_agent_type: str = "focal"  # context agent type
    # ----------------------FOCAL-----------------------------
    # ----------------------Classifier------------------------
    num_classes: int = 10
    classifier_hidden_dims: list = None
    # ----------------------Unicorn---------------------------
    decoder_hidden_dims: list = None
    unicorn_alpha: float = 1.0
    # ----------------------CERTAIN---------------------------
    enable_certain: bool = True
    loss_predictor_hidden_dims: list = None
    recon_decoder_hidden_dims: list = None
    # --------------------------------------------------------
    #                      TD3+BC agent
    # --------------------------------------------------------
    actor_hidden_dims: list = None
    critic_hidden_dims: list = None
    actor_learning_rate: float = 1e-3
    critic_learning_rate: float = 1e-3
    discount: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    alpha: float = 2.5
    # --------------------------------------------------------


class CERTAINRunner:

    def __init__(self, eval_envs, args):
        self.log_dir = Path(args.log_dir)
        self.writer = SummaryWriter(str(self.log_dir))
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        # TRY NOT TO MODIFY: seeding
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        self.eval_envs = eval_envs
        self.state_dim = np.prod(self.eval_envs.single_observation_space.shape)
        self.action_dim = np.prod(self.eval_envs.single_action_space.shape)
        self.args = args
        self.context_dim = (
            2 * self.state_dim + self.action_dim + 1
            if args.use_next_observation
            else self.state_dim + self.action_dim + 1
        )
        # Certain agent to learn the latent restraint stratage
        if args.enable_certain:
            self.certain_agent = CertainAgent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                context_dim=self.context_dim,
                latent_dim=args.latent_dim,
                loss_predictor_hidden_dims=args.loss_predictor_hidden_dims,
                recon_decoder_hidden_dims=args.recon_decoder_hidden_dims,
            ).to(ptu.device)
            self.certain_optimizer = optim.Adam(
                self.certain_agent.parameters(), lr=args.encoder_learning_rate, eps=1e-5
            )
        # Context agent to infer the latent variable
        if args.context_agent_type == "focal":
            self.context_agent = FocalAgent(
                context_dim=self.context_dim,
                latent_dim=args.latent_dim,
                encoder_hidden_dims=args.encoder_hidden_dims,
            ).to(ptu.device)
        elif args.context_agent_type == "classifier":
            self.context_agent = ClassifierAgent(
                context_dim=self.context_dim,
                latent_dim=args.latent_dim,
                encoder_hidden_dims=args.encoder_hidden_dims,
                num_classes=args.num_classes,
                classifier_hidden_dims=args.classifier_hidden_dims,
            ).to(ptu.device)
        elif args.context_agent_type == "unicorn":
            self.context_agent = UnicornAgent(
                context_dim=self.context_dim,
                latent_dim=args.latent_dim,
                encoder_hidden_dims=args.encoder_hidden_dims,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                decoder_hidden_dims=args.decoder_hidden_dims,
                unicorn_alpha=args.unicorn_alpha,
            ).to(ptu.device)
        else:
            raise NotImplementedError(
                f"Context agent type {args.context_agent_type} is not implemented."
            )
        self.context_optimizer = optim.Adam(
            self.context_agent.parameters(), lr=args.encoder_learning_rate, eps=1e-5
        )
        # TD3+BC agent to learn the policy
        self.policy_agent = TD3BCAgent(
            state_dim=self.state_dim + args.latent_dim,
            action_dim=self.action_dim,
            actor_hidden_dims=args.actor_hidden_dims,
            critic_hidden_dims=args.critic_hidden_dims,
            discount=args.discount,
            tau=args.tau,
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip,
            policy_freq=args.policy_freq,
            alpha=args.alpha,
        ).to(ptu.device)
        self.actor_optimizer = optim.Adam(
            self.policy_agent.actor.parameters(), lr=args.actor_learning_rate, eps=1e-5
        )
        self.critic1_optimizer = optim.Adam(
            self.policy_agent.critic1.parameters(),
            lr=args.critic_learning_rate,
            eps=1e-5,
        )
        self.critic2_optimizer = optim.Adam(
            self.policy_agent.critic2.parameters(),
            lr=args.critic_learning_rate,
            eps=1e-5,
        )

    def train(self):
        torch.autograd.set_detect_anomaly(True)
        # ALGO Logic: Load the offline dataset
        dataloader = DataLoader(
            ptu.OfflineMetaDataset(str(self.args.dataset_dir)),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=8,
            collate_fn=ptu.OfflineMetaDataset.collate_fn,
        )
        tasks = [float(info["task"]) for info in dataloader.dataset[0]["infos"]]

        # ALGO Logic: training setup
        global_step = 0
        for iteration in tqdm(
            range(self.args.num_iterations),
            desc=f"Iteration",
            unit="it",
        ):
            for batch in tqdm(
                dataloader, desc="Training batch", unit="batch", leave=False
            ):
                observations = batch["observations"].to(ptu.device)
                actions = batch["actions"].to(ptu.device)
                rewards = batch["rewards"].to(ptu.device)
                next_observations = batch["next_observations"].to(ptu.device)
                dones = batch["dones"].to(ptu.device)
                global_step += 1

                # Update context encoder
                context = torch.cat(
                    [observations, actions, rewards, next_observations], dim=-1
                )[..., : self.context_dim]
                context_loss, latent_zs = self.context_agent.compute_loss(context)
                self.context_optimizer.zero_grad()
                context_loss.backward()
                self.context_optimizer.step()

                if self.args.enable_certain:
                    # Update certain agent
                    certain_loss = self.certain_agent.compute_loss(
                        context, latent_zs.detach(), context_loss.detach()
                    )
                    self.certain_optimizer.zero_grad()
                    certain_loss.backward()
                    self.certain_optimizer.step()

                # Update TD3+BC
                latent_z = (
                    torch.mean(latent_zs, dim=1, keepdim=True)
                    .expand_as(latent_zs)
                    .detach()
                )

                critic_loss, actor_loss = self.policy_agent.update(
                    torch.cat([observations, latent_z], dim=-1),
                    actions,
                    rewards,
                    torch.cat([next_observations, latent_z], dim=-1),
                    dones,
                    self.actor_optimizer,
                    self.critic1_optimizer,
                    self.critic2_optimizer,
                )

                # Logging
                self.writer.add_scalar(
                    "Loss/context_loss", context_loss.item(), global_step
                )
                self.writer.add_scalar(
                    "Loss/critic_loss", critic_loss.item(), global_step
                )
                if actor_loss is not None:
                    self.writer.add_scalar(
                        "Loss/actor_loss", actor_loss.item(), global_step
                    )
                if self.args.enable_certain:
                    self.writer.add_scalar(
                        "Loss/certain_loss", context_loss.item(), global_step
                    )
            # ALGO Logic: eval
            if (
                iteration % self.args.eval_interval == 0
                or iteration == self.args.num_iterations - 1
            ):
                self.eval(tasks, dataloader.dataset, global_step)
            # ALGO Logic: save the model
            if (
                iteration % self.args.save_interval == 0
                or iteration == self.args.num_iterations - 1
            ):
                context_agent_path = (
                    self.log_dir / f"checkpoints/context_agent_{global_step}.pth"
                )
                policy_agent_path = (
                    self.log_dir / f"checkpoints/policy_agent_{global_step}.pth"
                )
                context_agent_path.parent.mkdir(parents=True, exist_ok=True)
                policy_agent_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.context_agent.state_dict(), str(context_agent_path))
                torch.save(self.policy_agent.state_dict(), str(policy_agent_path))
                if self.args.enable_certain:
                    certain_agent_path = (
                        self.log_dir / f"checkpoints/certain_agent_{global_step}.pth"
                    )
                    torch.save(self.certain_agent.state_dict(), str(certain_agent_path))

    def online_collect_traj(self, task, latent_z, episode_steps):
        observations = []
        actions = []
        rewards = []
        next_observations = []
        latent_z = latent_z.expand(len(self.eval_envs.envs), -1)
        obs, _ = self.eval_envs.reset(options={"task": task})
        for step in range(episode_steps):
            state = torch.cat([ptu.from_numpy(obs), latent_z], dim=-1)
            action = ptu.get_numpy(self.policy_agent.get_action(state))
            next_obs, reward, terminated, truncated, info = self.eval_envs.step(action)
            observations.append(obs)
            actions.append(action)
            rewards.append(reward.reshape(-1, 1))
            next_observations.append(obs)
            obs = next_obs
        return (
            np.array(observations).transpose(1, 0, 2),
            np.array(actions).transpose(1, 0, 2),
            np.array(rewards).transpose(1, 0, 2),
            np.array(next_observations).transpose(1, 0, 2),
        )

    def eval_task_with_context(self, task, context, context_type="online"):
        context = context[..., : self.context_dim].reshape(-1, self.context_dim)
        latent_zs = self.context_agent.infer_latent(context)
        if self.args.enable_certain and context_type == "online":
            latent_z = self.certain_agent.get_restraint_latent(context, latent_zs)
        else:
            latent_z = latent_zs.mean(dim=0, keepdim=True)
        first_traj = self.online_collect_traj(task, latent_z, self.args.eval_num_steps)
        observations, actions, rewards, next_observations = first_traj
        return np.mean(np.sum(rewards, axis=1))

    def eval_task(self, task, dataset=ptu.OfflineMetaDataset):
        # given offline context:
        offline_batch = dataset.sample(task, batch_size=self.args.batch_size)
        offline_context = torch.cat(
            [
                offline_batch["observations"],
                offline_batch["actions"],
                offline_batch["rewards"],
                offline_batch["next_observations"],
            ],
            dim=-1,
        ).to(ptu.device)
        offline_return = self.eval_task_with_context(
            task, offline_context, context_type="offline"
        )

        # eval zero-shot
        prior_latent_z = ptu.zeros(1, self.args.latent_dim)
        first_traj = self.online_collect_traj(
            task, prior_latent_z, self.args.eval_num_steps
        )
        observations, actions, rewards, next_observations = first_traj
        zero_shot_return = np.mean(np.sum(rewards, axis=1))
        # eval one-shot
        multi_context = torch.cat(
            [
                ptu.from_numpy(observations),
                ptu.from_numpy(actions),
                ptu.from_numpy(rewards),
                ptu.from_numpy(next_observations),
            ],
            dim=-1,
        )
        one_shot_returns = []
        for context in multi_context:
            one_shot_returns.append(self.eval_task_with_context(task, context))
        one_shot_return = np.mean(one_shot_returns)
        return offline_return, zero_shot_return, one_shot_return

    def eval(self, tasks: list, dataset: ptu.OfflineMetaDataset, global_step: int):
        offline_returns, zero_shot_returns, one_shot_returns = [], [], []
        for task in tasks:
            offline_return, zero_shot_return, one_shot_return = self.eval_task(
                task, dataset
            )
            self.writer.add_scalar(
                f"Eval_Task_{task}/offline_return", offline_return, global_step
            )
            self.writer.add_scalar(
                f"Eval_Task_{task}/zero_shot_return", zero_shot_return, global_step
            )
            self.writer.add_scalar(
                f"Eval_Task_{task}/one_shot_return", one_shot_return, global_step
            )
            offline_returns.append(offline_return)
            zero_shot_returns.append(zero_shot_return)
            one_shot_returns.append(one_shot_return)
        self.writer.add_scalar(
            f"Eval_All_task/offline_return", np.mean(offline_returns), global_step
        )
        self.writer.add_scalar(
            f"Eval_All_task/zero_shot_return", np.mean(zero_shot_returns), global_step
        )
        self.writer.add_scalar(
            f"Eval_All_task/one_shot_return", np.mean(one_shot_returns), global_step
        )
        tqdm.write(
            f"Step {global_step}: offline_return={np.mean(offline_returns):.3f} zero_shot_return={np.mean(zero_shot_returns):.3f} one_shot_return={np.mean(one_shot_returns):.3f}"
        )
