import torch
from torch import nn

def init_weights(m):
    """Initialize weights for the model."""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
        

class MLP(nn.Module):
    def __init__ (self, hidden_dims, input_dim, output_dim, output_activation=nn.Identity):
        super(MLP, self).__init__()
        self.hidden_dims = hidden_dims
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Create the layers of the MLP
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(output_activation())
        
        # Combine all layers into a single module
        self.model = nn.Sequential(*layers)
        self.model.apply(init_weights)

    def forward(self, x):
        return self.model(x)