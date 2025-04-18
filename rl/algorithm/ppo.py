import pickle
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
    save_training_trajectory: bool = False  # whether to save training transitions
    save_training_trajectory_interval: int = 2  # interval iterations to save
    # --------------------------------------------------------
    #                      PPO hyperparameters
    # --------------------------------------------------------
    learning_rate: float = 3e-4  # learning rate
    anneal_lr: float = True  # whether to anneal the learning rate
    gamma: float = 0.99  # discount factor
    gae_lambda: float = 0.95  # lambda for GAE
    clip_coef: float = 0.2  # clip range for the PPO
    ent_coef: float = 0.01  # 0.01  # entropy loss coefficient
    vf_coef: float = 0.5  # value loss coefficient
    clip_vloss: bool = True  # whether to clip the value loss
    max_grad_norm: float = 0.5  # max norm for the gradient
    # --------------------------------------------------------
    #                     Agent hyperparameters
    # --------------------------------------------------------
    actor_hidden_dims: list = None  # hidden dims for the actor
    critic_hidden_dims: list = None  # hidden dims for the critic
    # --------------------------------------------------------


class PPOAgent(nn.Module):
    LOG_SIG_MAX = 2
    LOG_SIG_MIN = -20
    EPS = 1e-6

    def __init__(self, state_dim, action_dim, actor_hidden_dims, critic_hidden_dims):
        super().__init__()
        self.register_buffer("state_dim", torch.tensor(state_dim))
        self.register_buffer("action_dim", torch.tensor(action_dim))

        self.critic = MLP(
            input_dim=self.state_dim, hidden_dims=critic_hidden_dims, output_dim=1
        )
        self.actor_mean = MLP(
            input_dim=self.state_dim,
            hidden_dims=actor_hidden_dims,
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
    def load_from(cls, path, actor_hidden_dims, critic_hidden_dims):
        state_dict = torch.load(path, map_location=ptu.device)
        state_dim = state_dict["state_dim"].detach().cpu().item()
        action_dim = state_dict["action_dim"].detach().cpu().item()
        agent = cls(state_dim, action_dim, actor_hidden_dims, critic_hidden_dims)
        agent.load_state_dict(state_dict)
        return agent


class PPORunner:

    def __init__(self, vec_envs, eval_envs, args, task_idx=0):
        self.task_idx = task_idx
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
        self.vec_envs = vec_envs
        self.num_envs = vec_envs.num_envs
        self.state_dim = np.prod(self.vec_envs.single_observation_space.shape)
        self.action_dim = np.prod(self.vec_envs.single_action_space.shape)
        self.args = args
        self.agent = PPOAgent(
            self.state_dim,
            self.action_dim,
            args.actor_hidden_dims,
            args.critic_hidden_dims,
        ).to(ptu.device)
        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=args.learning_rate, eps=1e-5
        )

    def train(self):
        # ALGO Logic: Storage setup
        observations = ptu.zeros((self.args.num_steps, self.num_envs, self.state_dim))
        actions = ptu.zeros((self.args.num_steps, self.num_envs, self.action_dim))
        logprobs = ptu.zeros((self.args.num_steps, self.num_envs, 1))
        rewards = ptu.zeros((self.args.num_steps, self.num_envs, 1))
        dones = ptu.zeros((self.args.num_steps, self.num_envs, 1))
        values = ptu.zeros((self.args.num_steps, self.num_envs, 1))
        next_observations = ptu.zeros_like(observations)
        infos = []

        # ALGO Logic: training setup
        global_step = 0
        obs, _ = self.vec_envs.reset(seed=self.args.seed)
        for iteration in tqdm(
            range(self.args.num_iterations),
            desc=f"Task {self.task_idx} Training iter",
            unit="it",
            position=self.task_idx,
            leave=True,
        ):
            # Annealing the rate if instructed to do so.
            if self.args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.args.num_iterations
                lrnow = frac * self.args.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow
            # ALGO Logic: collect data
            infos.clear()
            for step in range(self.args.num_steps):
                global_step += self.num_envs
                obs = ptu.from_numpy(obs)
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(obs)
                next_obs, reward, terminated, truncated, info = self.vec_envs.step(
                    ptu.get_numpy(action)
                )
                done = np.logical_or(terminated, truncated)

                # store data
                observations[step] = obs
                actions[step] = action
                logprobs[step] = logprob
                values[step] = value
                next_observations[step] = ptu.from_numpy(next_obs)
                rewards[step] = ptu.from_numpy(reward.reshape(-1, 1))
                dones[step] = ptu.from_numpy(done.reshape(-1, 1))
                infos.append(info)

                obs = next_obs

            # ALGO Logic: update agent
            # Compute the target value and advantage
            with torch.no_grad():
                # bootstrap value if not done
                final_next_value = self.agent.get_value(ptu.from_numpy(next_obs))
                advantages = ptu.zeros_like(rewards)
                gae = ptu.zeros(self.num_envs, 1)
                for t in reversed(range(self.args.num_steps)):
                    next_value = (
                        final_next_value
                        if t == self.args.num_steps - 1
                        else values[t + 1]
                    )
                    notdone = 1.0 - dones[t]
                    # PPO Advantage
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
            self.writer.add_scalar(
                "Return/training_return", rewards.sum(0).mean(), global_step
            )
            # ALGO Logic: eval
            if (
                iteration % self.args.eval_interval == 0
                or iteration == self.args.num_iterations - 1
            ):
                self.eval(global_step)
            if (
                iteration % self.args.save_interval == 0
                or iteration == self.args.num_iterations - 1
            ):
                save_path = self.log_dir / f"checkpoints/agent_{iteration}.pth"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.agent.state_dict(), str(save_path))
            if (
                self.args.save_training_trajectory
                and iteration % self.args.save_training_trajectory_interval == 0
                or iteration == self.args.num_iterations - 1
            ):
                for i in range(self.num_envs):
                    save_path = (
                        self.log_dir
                        / f"training_trajectory/itr_{iteration}_traj_{i}.pkl"
                    )
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    save_dict = {
                        "observations": ptu.get_numpy(observations[:, i]),
                        "actions": ptu.get_numpy(actions[:, i]),
                        "rewards": ptu.get_numpy(rewards[:, i]),
                        "next_observations": ptu.get_numpy(next_observations[:, i]),
                        "dones": ptu.get_numpy(dones[:, i]),
                        "infos": [{k: v[i]} for info in infos for k, v in info.items()],
                    }
                    # save with pickle
                    with open(save_path, "wb") as f:
                        pickle.dump(save_dict, f)
            # Rest the environment
            obs, _ = self.vec_envs.reset()
        self.vec_envs.close()

    def eval(self, training_step):
        obs, _ = self.eval_envs.reset()
        eval_return = 0
        for step in range(self.args.num_steps):
            action = self.agent.get_action(ptu.from_numpy(obs), deterministic=True)
            obs, reward, terminated, truncated, info = self.eval_envs.step(
                ptu.get_numpy(action)
            )
            eval_return += np.mean(reward)
        self.writer.add_scalar("Return/eval_return", eval_return, training_step)
        tqdm.write(f"Task {self.task_idx} Eval return: {eval_return}")
