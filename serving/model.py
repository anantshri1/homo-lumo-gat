import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

CONFIG = {
    'target_idx': 4,       # QM9 target 4 = HOMO-LUMO gap
    'batch_size': 32,
    'lr': 1e-3,
    'epochs': 100,
    'hidden_dim': 256,
    'num_heads': 4,
    'dropout': 0.1,
    'seed': 42,
}

class GATv2Model(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_dim = CONFIG['hidden_dim'], num_heads = CONFIG['num_heads'], dropout = CONFIG['dropout']):
        super().__init__()

        # 3 GATv2 convolution layers
        self.conv1 = GATv2Conv(in_channels, hidden_dim, heads=num_heads,
                                edge_dim=edge_dim, dropout=dropout, concat=False)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=num_heads,
                                edge_dim=edge_dim, dropout=dropout, concat=False)
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim, heads=num_heads,
                                edge_dim=edge_dim, dropout=dropout, concat=False)

        # Residual projections (to match dimensions)
        self.res1 = nn.Linear(in_channels, hidden_dim)
        self.res2 = nn.Linear(hidden_dim, hidden_dim)
        self.res3 = nn.Linear(hidden_dim, hidden_dim)

        # MLP head — input is hidden_dim*2 because we concat mean+max pooling
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_attr, batch):
        # Layer 1
        h = self.act(self.conv1(x, edge_index, edge_attr))
        h = h + self.res1(x)          # residual connection
        h = self.dropout(h)

        # Layer 2
        h2 = self.act(self.conv2(h, edge_index, edge_attr))
        h2 = h2 + self.res2(h)        # residual connection
        h2 = self.dropout(h2)

        # Layer 3
        h3 = self.act(self.conv3(h2, edge_index, edge_attr))
        h3 = h3 + self.res3(h2)       # residual connection

        # Global pooling — collapses all atom embeddings into one molecule embedding
        mean_pool = global_mean_pool(h3, batch)
        max_pool = global_max_pool(h3, batch)
        h_graph = torch.cat([mean_pool, max_pool], dim=1)

        return self.mlp(h_graph).squeeze(-1)
