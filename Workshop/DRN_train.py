import pickle as pk
import numpy as np
# import pandas as pd
#import h5py
# import matplotlib.pyplot as plt
# import awkward as ak
import torch
#import psutil
from torch_geometric.data import Data
#import h5py
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset, random_split
import os
import os.path as osp
import math

import numpy as np
import torch
import gc
import torch.nn as nn
from torch.nn.functional import softplus
import torch_geometric.transforms as T

from torch.utils.checkpoint import checkpoint
from torch_cluster import knn_graph, graclus_cluster
from torch_scatter import scatter
from torch_sparse.storage import SparseStorage

from torch import Tensor
from torch_geometric.typing import OptTensor, Optional, Tuple


from torch_geometric.nn import EdgeConv, NNConv
from torch_geometric.nn.pool.pool import pool_batch
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import normalized_cut
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn import (max_pool, max_pool_x, global_max_pool,
                                avg_pool, avg_pool_x, global_mean_pool, 
                                global_add_pool)
from tqdm import tqdm

transform = T.Cartesian(cat=False)

def normalized_cut_2d(edge_index, pos):
    row, col = edge_index[0], edge_index[1]
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))

# jit compatible version of coalesce
def coalesce(index, value: OptTensor, m: int, n: int, op: str = "add"):
    storage = SparseStorage(row=index[0], col=index[1], value=value,
                            sparse_sizes=(m, n), is_sorted=False)
    storage = storage.coalesce(reduce=op)
    return torch.stack([storage.row(), storage.col()], dim=0), storage.value()

# jit compatible version of to_undirected
def to_undirected(edge_index, num_nodes: Optional[int] = None) -> Tensor:
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index[0], edge_index[1]
    temp = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    row, col = temp[0], temp[1]
    edge_index = torch.stack([row, col], dim=0)
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
    return edge_index

# jit compatible version of pool_edge, depends on coalesce
def pool_edge(cluster, edge_index, edge_attr: Optional[torch.Tensor] = None):
    num_nodes = cluster.size(0)
    edge_index = cluster[edge_index.view(-1)].view(2, -1)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    if edge_index.numel() > 0:
        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                         num_nodes)
    return edge_index, edge_attr

def _aggr_pool_x(cluster, x, aggr: str, size: Optional[int] = None):
    """Call into scatter with configurable reduction op"""
    return scatter(x, cluster, dim=0, dim_size=size, reduce=aggr)

def global_pool_aggr(x, batch: OptTensor, aggr: str, size: Optional[int] = None):
    """Global pool via passed aggregator: 'mean', 'add', 'max'"""
    if batch is None and size is None:
        raise Exception('Must provide at least one of "batch" or "size"')
    if batch is not None:
        size = int(batch.max().item() + 1)
    assert batch is not None
    return scatter(x, batch, dim=0, dim_size=size, reduce=aggr)

# this function is specialized compared to the more general non-jittable version
# in particular edge_attr can be removed since it is always None
def aggr_pool(cluster, x, batch: OptTensor, aggr: str) -> Tuple[Tensor, OptTensor]:
    """jit-friendly version of max/mean/add pool"""
    cluster, perm = consecutive_cluster(cluster)
    x = _aggr_pool_x(cluster, x, aggr)
    if batch is not None:
        batch = pool_batch(perm, batch)
    return x, batch

def aggr_pool_x(cluster, x, batch: OptTensor, aggr: str, size: Optional[int] = None) -> Tuple[Tensor, OptTensor]:
    """*_pool_x with configurable aggr method"""
    if batch is None and size is None:
        raise Exception('Must provide at least one of "batch" or "size"')
    if size is not None and batch is not None:
        batch_size = int(batch.max().item()) + 1
        return _aggr_pool_x(cluster, x, aggr, batch_size * size), None

    cluster, perm = consecutive_cluster(cluster)
    x = _aggr_pool_x(cluster, x, aggr)
    if batch is not None:
        batch = pool_batch(perm, batch)

    return x, batch
    
class DynamicReductionNetworkJit(nn.Module):
    '''
    This model iteratively contracts nearest neighbour graphs 
    until there is one output node.
    The latent space trained to group useful features at each level
    of aggregration.
    This allows single quantities to be regressed from complex point counts
    in a location and orientation invariant way.
    One encoding layer is used to abstract away the input features.

    @param input_dim: dimension of input features
    @param hidden_dim: dimension of hidden layers
    @param output_dim: dimensio of output
    
    @param k: size of k-nearest neighbor graphs
    @param aggr: message passing aggregation scheme. 
    @param norm: feature normaliztion. None is equivalent to all 1s (ie no scaling)
    @param loop: boolean for presence/absence of self loops in k-nearest neighbor graphs
    @param pool: type of pooling in aggregation layers. Choices are 'add', 'max', 'mean'
    
    @param agg_layers: number of aggregation layers. Must be >=0
    @param mp_layers: number of layers in message passing networks. Must be >=1
    @param in_layers: number of layers in inputnet. Must be >=1
    @param out_layers: number of layers in outputnet. Must be >=1
    '''
    latent_probe: Optional[int]
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=1, k=16, aggr='add', norm=None, 
            loop=True, pool='max',
            agg_layers=2, mp_layers=2, in_layers=1, out_layers=3,
            graph_features = 0,
            latent_probe=None):
        super(DynamicReductionNetworkJit, self).__init__()

        self.graph_features = graph_features

        if latent_probe is not None and (latent_probe>agg_layers+1 or latent_probe<-1*agg_layers-1):
            print("Error: asked for invalid latent_probe layer")
            return
        
        if latent_probe is not None and latent_probe < 0:
            latent_probe = agg_layers+1 - latent_probe

        if latent_probe is not None:
            print("Probing latent features after %dth layer"%latent_probe)

        self.latent_probe = latent_probe

        self.loop = loop

        print("Pooling with",pool)
        print("Using self-loops" if self.loop else "Not using self-loops")
        print("There are",agg_layers,'aggregation layers')

        if norm is None:
            norm = torch.ones(input_dim)

        #normalization vector
        self.datanorm = nn.Parameter(norm)
        
        self.k = k

        #construct inputnet
        in_layers_l = []
        in_layers_l += [nn.Linear(input_dim, hidden_dim),
                nn.ELU()]

        for i in range(in_layers-1):
            in_layers_l += [nn.Linear(hidden_dim, hidden_dim), 
                    nn.ELU()]

        self.inputnet = nn.Sequential(*in_layers_l)


        #construct aggregation layers
        self.agg_layers = nn.ModuleList()
        for i in range(agg_layers):
            #construct message passing network
            mp_layers_l = []

            for j in range(mp_layers-1):
                mp_layers_l += [nn.Linear(2*hidden_dim, 2*hidden_dim),
                        nn.ELU()]

            mp_layers_l += [nn.Linear(2*hidden_dim, hidden_dim),
                    nn.ELU()]
           
            convnn = nn.Sequential(*mp_layers_l)
            
            self.agg_layers.append(EdgeConv(nn=convnn, aggr=aggr).jittable())

        #construct outputnet
        out_layers_l = []

        for i in range(out_layers-1):
            out_layers_l += [nn.Linear(hidden_dim+self.graph_features, hidden_dim+self.graph_features), 
                    nn.ELU()]

        out_layers_l += [nn.Linear(hidden_dim+self.graph_features, output_dim)]

        self.output = nn.Sequential(*out_layers_l)

        if pool not in {'max', 'mean', 'add'}:
            raise Exception("ERROR: INVALID POOLING")
        
        self.aggr_type = pool

    def forward(self, x: Tensor, batch: OptTensor, graph_x: OptTensor) -> Tensor:
        '''
        Push the batch 'data' through the network
        '''
        x = self.datanorm * x
        x = self.inputnet(x)

        latent_probe = self.latent_probe
        
        if graph_x is not None:
            graph_x = graph_x.view((-1, self.graph_features))

        # if there are no aggregation layers just leave x, batch alone
        nAgg = len(self.agg_layers)
        for i, edgeconv in enumerate(self.agg_layers):
            if latent_probe is not None and i == latent_probe:
                return x
            knn = knn_graph(x, self.k, batch, loop=self.loop, flow=edgeconv.flow)
            edge_index = to_undirected(knn)
            x = edgeconv(x, edge_index)

            weight = normalized_cut_2d(edge_index, x)
            cluster = graclus_cluster(edge_index[0], edge_index[1], weight, x.size(0))

            if i == nAgg - 1:
                x, batch = aggr_pool_x(cluster, x, batch, self.aggr_type)
            else:
                x, batch = aggr_pool(cluster, x, batch, self.aggr_type)

        if latent_probe is not None and latent_probe == nAgg:
            return x

        # this xforms to batch-per-row so no need to return batch
        x = global_pool_aggr(x, batch, self.aggr_type)

        if latent_probe is not None and latent_probe == nAgg + 1:
            return x

        if graph_x is not None:
            x = torch.cat((x, graph_x), 1)

        x = self.output(x).squeeze(-1)

        return x

print("-----------------Loading torchified data .....")

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
    
with open('events.pkl', 'rb') as f:
    events = pk.load(f)

target_energy = np.load("target_energy.npy")

dataset = RechitEventDataset(events, target_energy)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# ------------------------------
# Train Model with Progress Bar
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
    
print("------------------Loading_finished-------------------")

device = 'cuda'

model = DynamicReductionNetworkJit(
    input_dim=4,        # [x, y, z, E]
    hidden_dim=64,
    output_dim=1,       # regression target
    k=16,                # or 16 or 24
    aggr='add',         # 'mean' or 'max' also possible
    pool='max',        # or 'max', 'add'
    agg_layers=6,
    mp_layers=3,
    in_layers=4,
    out_layers=2,
    graph_features=0,   # set >0 if graph_x is used
    latent_probe=None   # or an int to return intermediate latent
)
model = model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()  # or MAE, SmoothL1Loss, etc.

for epoch in range(1, 41):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False)

    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
#        out = model(batch)
        out = model(batch.x, batch.batch, batch.graph_x if hasattr(batch, 'graph_x') else None)
        loss = loss_fn(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

        avg_loss = total_loss / ((pbar.n + 1) * batch.num_graphs)
        pbar.set_postfix(loss=f"{avg_loss:.4f}")

    print(f"Epoch {epoch:03d} completed, Avg Loss: {total_loss / len(train_dataset):.4f}")

torch.save(model.state_dict(), "DRN_regression_model.pt")

model.eval()
predictions = []
truths = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing", leave=False):
        batch = batch.to(device)
#        out = model(batch)
        out = model(batch.x, batch.batch, batch.graph_x if hasattr(batch, 'graph_x') else None)
        predictions.extend(out.cpu().numpy())
        truths.extend(batch.y.cpu().numpy())

predictions = np.array(predictions)
truths = np.array(truths)

# Save to files
np.save("predicted_energy_test_DRN.npy", predictions)
np.save("true_energy_test_DRN.npy", truths)
print("✅ Predictions saved as .npy files.")


