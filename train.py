import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from helper import plot_tsne

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")


class LabelFreeDataset(Dataset):

    def __init__(self, dataset_dir):
        self.data, self.labels = self.load_dataset(dataset_dir)

    def load_dataset(self, dataset_dir):
        all_data, labels = [], []
        for npy_path in Path(dataset_dir).rglob("*.npy"):
            trj = np.load(str(npy_path), allow_pickle=True)
            all_data.append(trj)
            labels.append(int(npy_path.parent.name.split("idx")[-1]))
        return all_data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)
        self.model.apply(self.init_weights)

    def forward(self, x):
        x = self.model(x)
        return x

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)


def train(dataset_dir):

    def collate_fn(batch):
        data, labels = zip(*batch)
        max_len = max([len(trj) for trj in data])
        data = [
            np.pad(trj, ((0, max_len - len(trj)), (0, 0)), mode="edge")
            for trj in data
        ]
        data = torch.tensor(np.stack(data), dtype=torch.float32).to(device)
        labels = torch.tensor(labels, dtype=torch.long)
        return data, labels

    dataset = DataLoader(
        LabelFreeDataset(dataset_dir),
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn,
    )

    state_dim, action_dim, latent_dim = 27, 8, 20
    input_dim = 2 * state_dim + action_dim + 1
    output_dim = state_dim + 1

    Encoder = MLP(input_dim, latent_dim, [128, 128, 128])
    Decoder = MLP(state_dim + action_dim + latent_dim, output_dim,
                  [128, 128, 128])
    Encoder.to(device)
    Decoder.to(device)

    # Train the model
    optimizer = torch.optim.Adam(list(Encoder.parameters()) +
                                 list(Decoder.parameters()),
                                 lr=1e-3)
    criterion = nn.MSELoss()
    num_epochs = 100
    for epoch in tqdm(range(num_epochs), desc="Training"):
        all_zs = []
        for data, labels in tqdm(dataset, desc="Epoch", leave=False):
            sa = data[..., :state_dim + action_dim]
            r_ns = data[..., state_dim + action_dim:]

            b, t, d = data.shape

            # Forward pass
            z = Encoder(data)
            z = torch.mean(z, dim=-2, keepdim=True)
            z = z.repeat(1, t, 1)
            pre_r_ns = Decoder(torch.cat([sa, z], dim=-1))

            # Compute loss
            loss = criterion(pre_r_ns, r_ns)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record
            record_zs = z[:, 0].cpu().detach().numpy()
            for z_i, l_i in zip(record_zs, labels):
                all_zs.append((z_i, l_i))
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            plot_tsne(all_zs, save_path=f"tsne_{epoch}.png")

        tqdm.write(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


if __name__ == "__main__":
    dataset_dir = "/home/autolab/zht/CSRO_new/offline_dataset/ant-dir"
    # set cuda visible devices
    train(dataset_dir)
