
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph, EdgeConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset, random_split
import numpy as np
import pickle as pk
from tqdm import tqdm

# ------------------------------
# Model Components
# ------------------------------

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

class RechitGNN(nn.Module):
    def __init__(self, k=16):
        super().__init__()
        self.k = k

        # Encoder with reduced dimensions
        self.encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        # Only two EdgeConv layers now
        self.edge_conv1 = EdgeConv(MLP(2 * 64, 64))
        self.edge_conv2 = EdgeConv(MLP(2 * 64, 64))

        # Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, data: Batch):
        x, pos, batch = data.x, data.pos, data.batch
        x = self.encoder(x)

        # First EdgeConv
        edge_index = knn_graph(x, k=self.k, batch=batch.to(x.device))
        x = self.edge_conv1(x, edge_index)

        # Second EdgeConv
        edge_index = knn_graph(x, k=self.k, batch=batch.to(x.device))
        x = self.edge_conv2(x, edge_index)

        x = global_mean_pool(x, batch)
        out = self.output_mlp(x)
        return out.view(-1)


# ------------------------------
# Dataset Class
# ------------------------------
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

# ------------------------------
# Load Data
# ------------------------------
with open('events.pkl', 'rb') as f:
    events = pk.load(f)

target_energy = np.load("target_energy.npy")

# ------------------------------
# Prepare Dataset and Split
# ------------------------------
dataset = RechitEventDataset(events, target_energy)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=300, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=300, shuffle=False)

# ------------------------------
# Train Model with Progress Bar
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = RechitGNN(k=16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(1, 5):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False)

    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

        avg_loss = total_loss / ((pbar.n + 1) * batch.num_graphs)
        pbar.set_postfix(loss=f"{avg_loss:.4f}")

    print(f"Epoch {epoch:03d} completed, Avg Loss: {total_loss / len(train_dataset):.4f}")

# Save trained model
torch.save(model.state_dict(), "rechit_regression_model.pt")

# ------------------------------
# Evaluate on Test Set
# ------------------------------
model.eval()
predictions = []
truths = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing", leave=False):
        batch = batch.to(device)
        out = model(batch)
        predictions.extend(out.cpu().numpy())
        truths.extend(batch.y.cpu().numpy())

predictions = np.array(predictions)
truths = np.array(truths)

# Save to files
np.save("predicted_energy_test.npy", predictions)
np.save("true_energy_test.npy", truths)
print("✅ Predictions saved as .npy files.")
