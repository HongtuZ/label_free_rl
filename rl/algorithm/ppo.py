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


@dataclass
class PPOArgs:
    # --------------------------------------------------------
    #                      Training setting
    # --------------------------------------------------------
    log_dir: str = "logs/ppo"
    seed: int = 1
    num_steps: int = 256  # number of env rollout steps per iteration
    num_iterations: int = 100  # total training iterations
    eval_interval: int = 1  # interval iteratoins to eval the policy
    save_interval: int = 10  # interval iterations to save the policy
    update_epochs: int = 10  # number of epochs to update the policy
    num_minibatches: int = 4  # number of minibatches to cal the loss
    eval_nums: int = 3  # number of episodes to eval the policy
    # --------------------------------------------------------
    #                      PPO hyperparameters
    # --------------------------------------------------------
    learning_rate: float = 3e-4  # learning rate
    anneal_lr: float = True  # whether to anneal the learning rate
    gamma: float = 0.99  # discount factor
    gae_lambda: float = 0.95  # lambda for GAE
    clip_coef: float = 0.2  # clip range for the PPO
    ent_coef: float = 0.0  # 0.01  # entropy loss coefficient
    vf_coef: float = 0.5  # value loss coefficient
    clip_vloss: bool = True  # whether to clip the value loss
    max_grad_norm: float = 0.5  # max norm for the gradient
    # --------------------------------------------------------


class PPOAgent(nn.Module):
    LOG_SIG_MAX = 2
    LOG_SIG_MIN = -20
    EPS = 1e-6

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.register_buffer("state_dim", torch.tensor(state_dim))
        self.register_buffer("action_dim", torch.tensor(action_dim))

        self.critic = MLP(input_dim=self.state_dim, hidden_dims=[64, 64], output_dim=1)
        self.actor_mean = MLP(
            input_dim=self.state_dim,
            hidden_dims=[64, 64],
            output_dim=self.action_dim,
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_dim))

    def get_value(self, state):
        return self.critic(state)

    @torch.no_grad()
    def get_action(self, state, deterministic=False):
        action_mean = self.actor_mean(state)
        if deterministic:
            action = torch.tanh(action_mean)
        else:
            action_std = torch.exp(
                torch.clamp(self.actor_logstd, self.LOG_SIG_MIN, self.LOG_SIG_MAX)
            )
            dist = ptu.TanhNormal(action_mean, action_std)
            action = dist.sample()
        return action

    def get_action_and_value(self, state, action=None):
        action_mean = self.actor_mean(state)
        action_std = torch.exp(
            torch.clamp(self.actor_logstd, self.LOG_SIG_MIN, self.LOG_SIG_MAX)
        )
        dist = ptu.TanhNormal(action_mean, action_std)
        if action is None:
            action = dist.rsample()
        log_prob = dist.log_prob(action)
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # sum over action dims

        return (
            action,
            log_prob,
            dist.entropy().sum(dim=-1, keepdim=True),
            self.critic(state),
        )

    @classmethod
    def load_from(cls, path):
        state_dict = torch.load(path, map_location=ptu.device)
        state_dim = state_dict["state_dim"].detach().cpu().item()
        action_dim = state_dict["action_dim"].detach().cpu().item()
        agent = cls(state_dim, action_dim)
        agent.load_state_dict(state_dict)
        return agent


class PPORunner:

    def __init__(self, vec_envs, eval_env, args):
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

        self.eval_env = eval_env
        self.vec_envs = vec_envs
        self.num_envs = vec_envs.num_envs
        self.state_dim = np.prod(self.vec_envs.single_observation_space.shape)
        self.action_dim = np.prod(self.vec_envs.single_action_space.shape)
        self.args = args
        self.agent = PPOAgent(self.state_dim, self.action_dim).to(ptu.device)
        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=args.learning_rate, eps=1e-5
        )

    def train(self):
        # ALGO Logic: Storage setup
        observations = torch.zeros(
            (self.args.num_steps, self.num_envs)
            + self.vec_envs.single_observation_space.shape
        ).to(ptu.device)
        actions = torch.zeros(
            (self.args.num_steps, self.num_envs)
            + self.vec_envs.single_action_space.shape
        ).to(ptu.device)
        logprobs = torch.zeros((self.args.num_steps, self.num_envs, 1)).to(ptu.device)
        rewards = torch.zeros((self.args.num_steps, self.num_envs, 1)).to(ptu.device)
        dones = torch.zeros((self.args.num_steps, self.num_envs, 1)).to(ptu.device)
        values = torch.zeros((self.args.num_steps, self.num_envs, 1)).to(ptu.device)

        # ALGO Logic: training setup
        global_step = 0
        obs, _ = self.vec_envs.reset(seed=self.args.seed)
        done = np.zeros(self.num_envs)
        for iteration in tqdm(
            range(self.args.num_iterations), desc="Training iter", unit="it"
        ):
            # Annealing the rate if instructed to do so.
            if self.args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.args.num_iterations
                lrnow = frac * self.args.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow
            # ALGO Logic: collect data
            for step in range(self.args.num_steps):
                global_step += self.num_envs
                obs, done = ptu.from_numpy(obs), ptu.from_numpy(done.reshape(-1, 1))
                observations[step], dones[step] = obs, done
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(obs)
                obs, reward, terminated, truncated, info = self.vec_envs.step(
                    ptu.get_numpy(action)
                )
                done = np.logical_or(terminated, truncated)

                # store data
                actions[step] = action
                logprobs[step] = logprob
                rewards[step] = ptu.from_numpy(reward.reshape(-1, 1))
                values[step] = value

            # ALGO Logic: update agent
            # Compute the target value and advantage
            with torch.no_grad():
                # bootstrap value if not done
                final_next_value = self.agent.get_value(ptu.from_numpy(obs))
                final_done = ptu.from_numpy(done.reshape(-1, 1))
                # cal the advantage
                advantages = ptu.zeros_like(rewards)
                gae = ptu.zeros(self.num_envs, 1)
                for t in reversed(range(self.args.num_steps)):
                    if t == self.args.num_steps - 1:
                        # bootstrap value if not done
                        next_value = final_next_value
                        notdone = 1.0 - final_done
                    else:
                        next_value = values[t + 1]
                        notdone = 1.0 - dones[t + 1]
                    delta = (
                        rewards[t] + self.args.gamma * next_value * notdone - values[t]
                    )
                    advantages[t] = gae = (
                        delta + self.args.gamma * self.args.gae_lambda * notdone * gae
                    )
                    # Target value
                    returns = advantages + values
            # update the policy and value networks with mini-batch
            b_observations = observations.reshape(-1, self.state_dim)
            b_actions = actions.reshape(-1, self.action_dim)
            b_logprobs = logprobs.reshape(-1, 1)
            b_values = values.reshape(-1, 1)
            b_advantages = advantages.reshape(-1, 1)
            b_returns = returns.reshape(-1, 1)

            batch_size = b_observations.shape[0]
            minibatch_size = batch_size // self.args.num_minibatches
            b_idxs = np.arange(batch_size)

            for _ in range(self.args.update_epochs):
                random.shuffle(b_idxs)
                for start in range(0, len(b_idxs), minibatch_size):
                    # update with mini-batch
                    end = start + minibatch_size
                    mb_idxs = b_idxs[start:end]
                    _, new_logprobs, entropy, new_values = (
                        self.agent.get_action_and_value(
                            b_observations[mb_idxs], b_actions[mb_idxs]
                        )
                    )
                    # cal policy gradient loss
                    ratio = torch.exp(new_logprobs - b_logprobs[mb_idxs])
                    pg_loss1 = -b_advantages[mb_idxs] * ratio
                    pg_loss2 = -b_advantages[mb_idxs] * torch.clamp(
                        ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # cal value loss
                    if self.args.clip_vloss:
                        v_loss_unclipped = (new_values - b_returns[mb_idxs]) ** 2
                        v_clipped = b_values[mb_idxs] + torch.clamp(
                            new_values - b_values[mb_idxs],
                            -self.args.clip_coef,
                            self.args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_idxs]) ** 2
                        v_loss = torch.mean(torch.max(v_loss_unclipped, v_loss_clipped))
                    else:
                        v_loss = torch.mean(
                            0.5 * (new_values - b_returns[mb_idxs]) ** 2
                        )

                    # cal entropy loss
                    entropy_loss = entropy.mean()

                    # total_loss
                    loss = (
                        pg_loss
                        - self.args.ent_coef * entropy_loss
                        + v_loss * self.args.vf_coef
                    )

                    # update the policy and value networks
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.agent.parameters(), self.args.max_grad_norm
                    )
                    self.optimizer.step()
            self.writer.add_scalar("Loss/pg_loss", pg_loss.item(), global_step)
            self.writer.add_scalar("Loss/v_loss", v_loss.item(), global_step)
            self.writer.add_scalar(
                "Loss/entropy_loss", entropy_loss.item(), global_step
            )
            self.writer.add_scalar("Loss/total_loss", loss.item(), global_step)
            # ALGO Logic: eval
            if iteration % self.args.eval_interval == 0:
                self.eval(global_step)
            if iteration % self.args.save_interval == 0:
                save_path = self.log_dir / f"checkpoints/agent_{iteration}.pth"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.agent.state_dict(), str(save_path))

            # Rest the environment
            obs, _ = self.vec_envs.reset()
        self.vec_envs.close()

    def eval(self, training_step):
        obs, _ = self.eval_env.reset()
        eval_nums, total_rewards = self.args.eval_nums, 0
        for _ in range(eval_nums):
            for step in range(self.args.num_steps):
                action = self.agent.get_action(ptu.from_numpy(obs), deterministic=False)
                obs, reward, terminated, truncated, info = self.eval_env.step(
                    ptu.get_numpy(action).flatten()
                )
                total_rewards += reward
        self.writer.add_scalar("Eval/return", total_rewards / eval_nums, training_step)
        tqdm.write(
            f"Training step: {training_step} Eval return: {total_rewards / eval_nums:.6f}"
        )
