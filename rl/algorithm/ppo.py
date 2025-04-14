import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter

import rl.pytorch_utils as ptu
from rl.net.mlp import MLP

@dataclass
class PPOArgs:
    log_dir: str = 'logs/ppo'
    seed: int = 1
    learning_rate: float = 2.5e-4
    num_iterations: int = 10000
    num_steps: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    minibatch_size: int = 256


class PPOAgent(nn.Module):
    LOG_SIG_MAX = 2
    LOG_SIG_MIN = -20
    EPS = 1e-6

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.register_buffer('state_dim', torch.tensor(state_dim))
        self.register_buffer('action_dim', torch.tensor(action_dim))

        self.critic = MLP(
            input_dim=self.state_dim,
            hidden_dims=[64,64],
            output_dim=1
        )
        self.actor_mean = MLP(
            input_dim=self.state_dim,
            hidden_dims=[64,64],
            output_dim=self.action_dim,
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_dim))


    def get_value(self, state):
        return self.critic(state)

    def get_action_and_value(self, state, action=None):
        action_mean = self.actor_mean(state)
        action_std = torch.exp(torch.clamp(self.actor_logstd, self.LOG_SIG_MIN, self.LOG_SIG_MAX))
        dist = ptu.TanhNormal(action_mean, action_std)
        if action is None:
            action = dist.rsample()
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # sum over action dims

        return action, log_prob, normal.entropy().sum(dim=-1, keepdim=True), self.critic(state)

    @classmethod
    def load_from(cls, path):
        agent = cls()
        agent.load_state_dict(torch.load(path))
        return agent

class PPORunner:
    def __init__(self, env, args):
        self.log_dir = Path(args.log_dir)/f'goal_{env.current_task_idx}'
        self.writer = SummaryWriter(str(self.log_dir))
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        # TRY NOT TO MODIFY: seeding
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        self.env = env
        self.state_dim = np.prod(env.observation_space.shape)
        self.action_dim = np.prod(env.action_space.shape)
        self.args = args
        self.agent = PPOAgent(self.state_dim, self.action_dim).to(ptu.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=args.learning_rate, eps=1e-5)

    def train(self):
        # ALGO Logic: Storage setup
        observations = np.zeros((self.args.num_steps, self.state_dim))
        actions = np.zeros((self.args.num_steps, self.action_dim))
        logprobs = np.zeros((self.args.num_steps,))
        rewards = np.zeros((self.args.num_steps,))
        dones = np.zeros((self.args.num_steps,))
        values = np.zeros((self.args.num_steps,))

        # ALGO Logic: training setup
        global_step = 0
        obs, _ = self.env.reset(seed=self.args.seed)
        done = False
        for iteration in tqdm(range(self.args.num_iterations), desc="Iter", unit="it"):
            # ALGO Logic: collect data
            for step in range(self.args.num_steps):
                global_step += 1
                observations[step], dones[step] = obs.flatten(), done
                with torch.no_grad():
                    if not isinstance(obs, np.ndarray):
                        obs = np.array(obs)
                    action, logprob, _, value = self.agent.get_action_and_value(ptu.from_numpy(obs))
                obs, reward, terminated, truncated, info = self.env.step(action.cpu().numpy().flatten())
                done = terminated or truncated

                # store data
                actions[step] = action.cpu().numpy()
                logprobs[step] = logprob.cpu().numpy()
                rewards[step] = reward
                values[step] = value.cpu().numpy()

                if terminated or truncated:
                    obs, _ = self.env.reset()
            # ALGO Logic: update agent
            with torch.no_grad():
                final_next_value = self.agent.get_value(ptu.from_numpy(obs)).cpu().numpy()
            final_done = True
            # cal the advantage
            advantages = np.zeros_like(rewards)
            gae = 0
            for t in reversed(range(self.args.num_steps)):
                next_value = final_next_value if t == self.args.num_steps - 1 else values[t + 1]
                next_done = final_done if t == self.args.num_steps - 1 else dones[t + 1]
                delta = rewards[t] + self.args.gamma * next_value * (1 - next_done) - values[t]
                advantages[t] = gae = delta + self.args.gamma * self.args.gae_lambda * (1 - next_done) * gae
            returns = advantages + values
            # update the policy and value networks with mini-batch
            b_idxs = np.arange(self.args.num_steps)
            for _ in range(self.args.update_epochs):
                random.shuffle(b_idxs)
                for start in range(0, len(b_idxs), self.args.minibatch_size):
                    # update with mini-batch
                    end = start + self.args.minibatch_size
                    mb_idxs = b_idxs[start:end]
                    _, new_logprobs, entropy, new_values = self.agent.get_action_and_value(ptu.from_numpy(observations[mb_idxs]), ptu.from_numpy(actions[mb_idxs]))
                    # cal policy gradient loss
                    ratio = torch.exp(new_logprobs - ptu.from_numpy(logprobs[mb_idxs]))
                    pg_loss1 = -ptu.from_numpy(advantages[mb_idxs]) * ratio
                    pg_loss2 = -ptu.from_numpy(advantages[mb_idxs]) * torch.clamp(ratio, 1-self.args.clip_coef, 1+self.args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    # cal value loss
                    v_loss = torch.mean(0.5 * (new_values - ptu.from_numpy(returns[mb_idxs])) ** 2)
                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef
                    # update the policy and value networks
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    # tqdm.write(f"pg_loss: {pg_loss.item():.6f}, v_loss: {v_loss.item():.6f}, entropy_loss: {entropy_loss.item():.6f}")
            self.writer.add_scalar('Loss/pg_loss', pg_loss.item(), global_step)
            self.writer.add_scalar('Loss/v_loss', v_loss.item(), global_step)
            self.writer.add_scalar('Loss/entropy_loss', entropy_loss.item(), global_step)
            self.writer.add_scalar('Loss/total_loss', loss.item(), global_step)
            # ALGO Logic: eval
            if iteration % 5 == 0:
                self.eval(global_step)
            if iteration % 100 == 0:
                save_path = self.log_dir / f"agent_{iteration}.pth"
                torch.save(self.agent.state_dict(), str(save_path))
        self.env.close()

    def eval(self, training_step):
        obs, _ = self.env.reset()
        eval_num, total_rewards = 3, 0
        for _ in range(eval_num):
            for step in range(self.args.num_steps):
                with torch.no_grad():
                    action, _, _, _ = self.agent.get_action_and_value(ptu.from_numpy(obs))
                obs, reward, terminated, truncated, info = self.env.step(action.cpu().numpy().flatten())
                total_rewards += reward
        self.writer.add_scalar('Eval/return', total_rewards / eval_num, training_step)
        tqdm.write(f"Training step: {training_step} Eval return: {total_rewards / eval_num:.6f}")