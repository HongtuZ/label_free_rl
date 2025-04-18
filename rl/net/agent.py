import copy
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import rl.pytorch_utils as ptu
from rl.net.mlp import MLP


class FOCALAgent(nn.Module):
    def __init__(
        self,
        context_dim,
        latent_dim,
        encoder_hidden_dims,
    ):
        super().__init__()
        self.register_buffer("context_dim", torch.tensor(context_dim))
        self.register_buffer("latent_dim", torch.tensor(latent_dim))
        self.register_buffer("encoder_hidden_dims", torch.tensor(encoder_hidden_dims))
        self.encoder = MLP(
            input_dim=context_dim,
            hidden_dims=encoder_hidden_dims,
            output_dim=latent_dim,
            output_activation=nn.Tanh,
        )

    def compute_loss(self, context):
        # Compute the encoder loss
        latent_zs = self.encoder(context)  # shape (task, batch, dim)
        latent_z = torch.mean(latent_zs, dim=1)  # shape (task, dim)
        tasks = list(range(latent_z.shape[0]))
        return self.metric_loss(latent_z, tasks), latent_zs

    # FOCAL-latest: https://github.com/LanqingLi1993/FOCAL-latest/blob/main/train_offline_FOCAL.py#L244
    def metric_loss(self, z, tasks, epsilon=1e-3):
        # z shape is (task, dim)
        pos_z_loss = 0.0
        neg_z_loss = 0.0
        pos_cnt = 0
        neg_cnt = 0
        for i in range(len(tasks)):
            for j in range(i + 1, len(tasks)):
                # positive pair
                if tasks[i] == tasks[j]:
                    pos_z_loss += torch.sqrt(torch.mean((z[i] - z[j]) ** 2) + epsilon)
                    pos_cnt += 1
                else:
                    neg_z_loss += 1 / (torch.mean((z[i] - z[j]) ** 2) + epsilon * 100)
                    neg_cnt += 1
        # print(pos_z_loss, pos_cnt, neg_z_loss, neg_cnt)
        return pos_z_loss / (pos_cnt + epsilon) + neg_z_loss / (neg_cnt + epsilon)

    @torch.no_grad()
    def infer_latent(self, context):
        # Compute the encoder loss
        latent_zs = self.encoder(context)
        return latent_zs


class TD3BCAgent(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_hidden_dims: list = None,
        critic_hidden_dims: list = None,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        alpha: float = 2.5,
    ):
        super().__init__()
        self.register_buffer("state_dim", torch.tensor(state_dim))
        self.register_buffer("action_dim", torch.tensor(action_dim))
        self.actor = MLP(
            input_dim=state_dim,
            hidden_dims=actor_hidden_dims,
            output_dim=action_dim,
            output_activation=nn.Tanh,
        )
        self.critic1 = MLP(
            input_dim=state_dim + action_dim,
            hidden_dims=critic_hidden_dims,
            output_dim=1,
        )
        self.critic2 = MLP(
            input_dim=state_dim + action_dim,
            hidden_dims=critic_hidden_dims,
            output_dim=1,
        )

        self.actor_target = copy.deepcopy(self.actor)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.total_it = 0

    def update(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        dones,
        actor_optimizer,
        critic1_optimizer,
        critic2_optimizer,
    ):
        # Compute the critic
        self.total_it += 1
        with torch.no_grad():
            # Select action according to actor and add clipped noise
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_observations) + noise).clamp(-1, 1)
            # Compute the target Q value
            target_q1 = self.critic1_target(
                torch.cat([next_observations, next_action], dim=-1)
            )
            target_q2 = self.critic2_target(
                torch.cat([next_observations, next_action], dim=-1)
            )
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.discount * target_q

        # Get current Q estimates
        current_q1 = self.critic1(torch.cat([observations, actions], dim=-1))
        current_q2 = self.critic2(torch.cat([observations, actions], dim=-1))

        # Compute critic loss
        critic_loss = torch.mean((current_q1 - target_q) ** 2) + torch.mean(
            (current_q2 - target_q) ** 2
        )
        critic1_optimizer.zero_grad()
        critic2_optimizer.zero_grad()
        critic_loss.backward()
        critic1_optimizer.step()
        critic2_optimizer.step()

        # Delayed actor updates
        actor_loss = None
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            pi = self.actor(observations)
            q = self.critic1(torch.cat([observations, pi], dim=-1))
            lmbda = self.alpha / q.abs().mean().detach()
            actor_loss = -lmbda * q.mean() + torch.mean((pi - actions) ** 2)
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            ptu.soft_update(self.critic1_target, self.critic1, self.tau)
            ptu.soft_update(self.critic2_target, self.critic2, self.tau)
            ptu.soft_update(self.actor_target, self.actor, self.tau)
        return critic_loss, actor_loss

    @torch.no_grad()
    def get_action(self, observation):
        action = self.actor(observation)
        return action
