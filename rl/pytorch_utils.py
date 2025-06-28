import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils
import torch.utils.data
from torch.distributions import Distribution, Normal
from torch.distributions.utils import broadcast_all

device = "cpu"


def from_numpy(x):
    """Convert a numpy array to a PyTorch tensor."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.float().to(device)
    return torch.from_numpy(x).float().to(device) if x is not None else None


def get_numpy(x):
    """Convert a PyTorch tensor to a numpy array."""
    return x.detach().cpu().numpy() if x is not None else None


def zeros(*sizes, **kwargs):
    return torch.zeros(*sizes, **kwargs).to(device)


def ones(*sizes, **kwargs):
    return torch.ones(*sizes, **kwargs).to(device)


def randn(*args, **kwargs):
    return torch.randn(*args, **kwargs).to(device)


def zeros_like(*args, **kwargs):
    return torch.zeros_like(*args, **kwargs).to(device)


def ones_like(*args, **kwargs):
    return torch.ones_like(*args, **kwargs).to(device)


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


class TanhNormal(Distribution):
    arg_constraints = {
        "loc": torch.distributions.constraints.real,
        "scale": torch.distributions.constraints.positive,
    }
    has_rsample = True

    def __init__(self, loc, scale, eps=1e-6, validate_args=False):
        self.loc, self.scale = broadcast_all(loc, scale)
        self.normal = Normal(self.loc, self.scale)
        self.eps = eps
        super().__init__(self.normal.batch_shape, validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        z = self.normal.sample(sample_shape)
        return torch.tanh(z)

    def rsample(self, sample_shape=torch.Size()):
        z = self.normal.rsample(sample_shape)
        return torch.tanh(z)

    def log_prob(self, value):
        value = torch.clamp(value, -1 + self.eps, 1 - self.eps)
        atanh = 0.5 * (torch.log1p(value) - torch.log1p(-value))
        log_prob = self.normal.log_prob(atanh)
        log_det_jacobian = 2 * (
            torch.log(torch.tensor(2.0, device=value.device))
            - atanh
            - F.softplus(-2 * atanh)
        )
        return log_prob - log_det_jacobian

    def entropy(self, n_samples=1):
        """
        Estimate the entropy of TanhNormal using Monte Carlo sampling.
        """
        z = self.normal.rsample((n_samples,))  # shape: (n_samples, *batch_shape)
        log_det = 2 * (
            torch.log(torch.tensor(2.0, device=z.device)) - z - F.softplus(-2 * z)
        )
        # H[tanh(z)] = H[z] + E[log|det J|]
        entropy = self.normal.entropy() + log_det.mean(dim=0)
        return entropy

    @property
    def mean(self):
        return torch.tanh(self.loc)

    @property
    def stddev(self):
        return self.scale

    @property
    def mode(self):
        return self.mean

    @property
    def base_dist(self):
        return self.normal


class OfflineMetaDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = self.load_data(data_dir)
        self.max_size = max(
            [len(task_data["observations"]) for task_data in self.data.values()]
        )

    def __len__(self):
        return self.max_size

    def __getitem__(self, idx):
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []
        infos = []
        for task, task_data in self.data.items():
            actual_idx = idx % len(task_data["observations"])
            observations.append(task_data["observations"][actual_idx])
            actions.append(task_data["actions"][actual_idx])
            rewards.append(task_data["rewards"][actual_idx])
            next_observations.append(task_data["next_observations"][actual_idx])
            dones.append(task_data["dones"][actual_idx])
            infos.append(task_data["infos"][actual_idx])
        return {
            "observations": np.array(observations),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "next_observations": np.array(next_observations),
            "dones": np.array(dones),
            "infos": infos,
        }

    @property
    def tasks(self):
        return [float(task) for task in self.data.keys()]

    @property
    def all_data(self):
        tensor_keys = [
            "observations",
            "actions",
            "rewards",
            "next_observations",
            "dones",
        ]
        list_keys = ["infos", "tasks"]
        all_data = {key: [] for key in tensor_keys + list_keys}
        for task, task_data in self.data.items():
            for key in tensor_keys:
                all_data[key].append(task_data[key])
            all_data["infos"].extend(task_data["infos"])
            all_data["tasks"].extend([float(task)] * len(task_data["observations"]))
        for key in tensor_keys:
            all_data[key] = torch.tensor(np.concatenate(all_data[key], axis=0))
        return all_data

    def sample(self, task, batch_size):
        task_data = self.data[str(task)]
        indices = np.random.choice(len(task_data["observations"]), batch_size)
        observations = task_data["observations"][indices]
        actions = task_data["actions"][indices]
        rewards = task_data["rewards"][indices]
        next_observations = task_data["next_observations"][indices]
        dones = task_data["dones"][indices]
        infos = [task_data["infos"][i] for i in indices]
        return {
            "observations": torch.tensor(observations),
            "actions": torch.tensor(actions),
            "rewards": torch.tensor(rewards),
            "next_observations": torch.tensor(next_observations),
            "dones": torch.tensor(dones),
            "infos": infos,
        }

    def sample_all(self, batch_size):
        all_data = {
            key: []
            for key in [
                "observations",
                "actions",
                "rewards",
                "next_observations",
                "dones",
                "infos",
            ]
        }
        for task_data in self.data.values():
            indices = np.random.choice(len(task_data["observations"]), batch_size)
            for key in all_data:
                all_data[key].append(
                    task_data[key][indices]
                    if key != "infos"
                    else [task_data[key][i] for i in indices]
                )
        return {
            key: (
                torch.tensor(np.stack(all_data[key]))
                if key != "infos"
                else all_data[key]
            )
            for key in all_data
        }

    @staticmethod
    def collate_fn(batch):
        # Convert batch of data to tensors: tasks x batch_size x data_dim
        batch_data = {
            key: (
                torch.tensor(np.stack([b[key] for b in batch])).permute(1, 0, 2)
                if key != "infos"
                else list(zip(*[b[key] for b in batch]))
            )
            for key in batch[0]
        }
        return batch_data

    def load_data(self, data_dir):
        data_dir = Path(data_dir)
        data = {}
        if not data_dir.exists():
            raise ValueError(f"Data directory {str(data_dir)} does not exist.")
        for traj_path in data_dir.rglob("*.pkl"):
            task = traj_path.parent.parent.name.split("_")[-1]
            if task not in data:
                data[task] = {
                    "observations": [],
                    "actions": [],
                    "rewards": [],
                    "next_observations": [],
                    "dones": [],
                    "infos": [],
                }
            with open(str(traj_path), "rb") as f:
                traj_data = pickle.load(f)
            data[task]["observations"].extend(traj_data["observations"])
            data[task]["actions"].extend(traj_data["actions"])
            data[task]["rewards"].extend(traj_data["rewards"])
            data[task]["next_observations"].extend(traj_data["next_observations"])
            data[task]["dones"].extend(traj_data["dones"])
            data[task]["infos"].extend(traj_data["infos"])
        for task, task_data in data.items():
            for key, value in task_data.items():
                if key != "infos":
                    task_data[key] = np.array(value)
        return data
