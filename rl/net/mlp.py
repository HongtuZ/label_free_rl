import torch
import torch.nn.functional as F
from torch import nn


def init_weights(m):
    """Initialize weights for the model."""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


class MLP(nn.Module):

    def __init__(
        self, hidden_dims, input_dim, output_dim, output_activation=nn.Identity
    ):
        super(MLP, self).__init__()
        self.hidden_dims = hidden_dims
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Create the layers of the MLP
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LeakyReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(output_activation())

        # Combine all layers into a single module
        self.model = nn.Sequential(*layers)
        self.model.apply(init_weights)

    def forward(self, x):
        return self.model(x)


class GaussianMLP(nn.Module):
    def __init__(self, hidden_dims, input_dim, output_dim):
        super(GaussianMLP, self).__init__()
        self.hidden_dims = hidden_dims
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Create the layers of the MLP
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LeakyReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim * 2))  # mu and log_std

        # Combine all layers into a single module
        self.model = nn.Sequential(*layers)
        self.model.apply(init_weights)

        # Initialize max_logvar and min_logvar for regularization
        self.max_logvar = nn.Parameter(torch.ones(output_dim) * 0.5)
        self.min_logvar = nn.Parameter(torch.ones(output_dim) * -10)

    def forward(self, x):
        mu_logvar = self.model(x)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mu, logvar

    def nll_loss(self, input, target):
        mu, logvar = self.forward(input)
        nll = torch.sum(logvar + ((target - mu) ** 2) / torch.exp(logvar), dim=-1)
        # Add regularization term
        reg_term = 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
        return nll.mean() + reg_term
