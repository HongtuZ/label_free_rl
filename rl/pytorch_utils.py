import torch
import torch.nn.functional as F
from torch.distributions import Distribution, Normal
from torch.distributions.utils import broadcast_all

device = 'cpu'

def from_numpy(x):
    """Convert a numpy array to a PyTorch tensor."""
    return torch.from_numpy(x).float().to(device) if x is not None else None

def to_numpy(x):
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


class TanhNormal(Distribution):
    arg_constraints = {'loc': torch.distributions.constraints.real,
                       'scale': torch.distributions.constraints.positive}
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
        log_det_jacobian = 2 * (torch.log(torch.tensor(2.0, device=value.device)) 
                                - atanh - F.softplus(-2 * atanh))
        return log_prob - log_det_jacobian

    def entropy(self, n_samples=1):
        """
        Estimate the entropy of TanhNormal using Monte Carlo sampling.
        """
        z = self.normal.rsample((n_samples,))  # shape: (n_samples, *batch_shape)
        log_det = 2 * (torch.log(torch.tensor(2.0, device=z.device)) 
                       - z - F.softplus(-2 * z))
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