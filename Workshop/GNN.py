import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph, EdgeConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import pickle as pk
import h5py
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class RechitGNN(torch.nn.Module):
    def __init__(self, k=16):
        super().__init__()
        self.k = k

        # Step 1: Per-rechit encoder (shared across rechits)
        self.encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # Step 2-4: EdgeConv + Pool (repeat 3 times)
        self.edge_conv1 = EdgeConv(MLP(2 * 128, 128))
        self.edge_conv2 = EdgeConv(MLP(2 * 128, 128))
        self.edge_conv3 = EdgeConv(MLP(2 * 128, 128))

        # Output block
        self.output_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Regression output
        )

    def forward(self, data: Batch):
        x, pos, batch = data.x, data.pos, data.batch

        # Encode input (x, y, z, E)
        x = self.encoder(x)

        # Repeated EdgeConv + KNN graph building
        for edge_conv in [self.edge_conv1, self.edge_conv2, self.edge_conv3]:
            edge_index = knn_graph(x, k=self.k, batch=batch)
            x = edge_conv(x, edge_index)

        # Global pooling
        x = global_mean_pool(x, batch)

        # Final MLP
        out = self.output_mlp(x)
        return out.view(-1)

model = RechitGNN()

print(model)

with open('events.pkl', 'rb') as f:
    data = pk.load(f)

print(data)


with h5py.File("hgcal_electron_data_0001.h5", "r") as f:
    target_energy = f["target"][:]  # shape: (num_events,)

print(target_energy)

class RechitEventDataset(Dataset):
    def __init__(self, events, target_energy):
        assert len(events) == len(target_energy)
        self.events = events
        self.targets = target_energy

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        e = self.events[idx]
        y = self.targets[idx]

        x = torch.tensor(e['x'], dtype=torch.float32)
        y_ = torch.tensor(e['y'], dtype=torch.float32)
        z = torch.tensor(e['z'], dtype=torch.float32)
        E = torch.tensor(e['E'], dtype=torch.float32)

        features = torch.stack([x, y_, z, E], dim=-1)
        pos = features[:, :3]
        y_target = torch.tensor([y], dtype=torch.float32)

        return Data(x=features, pos=pos, y=y_target)

dataset = RechitEventDataset(data, target_energy)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# model = RechitGNN(k=12)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

# Training loop
for epoch in range(1, 101):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to("cuda" if torch.cuda.is_available() else "cpu")

        optimizer.zero_grad()
        out = model(batch)            # shape: (batch_size,)
        loss = loss_fn(out, batch.y)  # batch.y shape: (batch_size, 1)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

    print(f"Epoch {epoch}, Loss: {total_loss / len(dataset):.4f}")

