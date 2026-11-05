import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_dense_batch


class DelayModel(nn.Module):
    def __init__(self, name, config, data):
        super().__init__()
        self.name, self.dropout = name, config["dropout"]
        hidden, layers = config["hidden"], config["layers"]
        # Map numerical features and category IDs into the same hidden space.
        self.project = nn.Linear(data.x.shape[1] + 2, hidden)
        self.categories = nn.ModuleList([nn.Embedding(size, hidden, padding_idx=0) for size in data.category_sizes])
        if name == "LSTM":
            self.encoder = nn.LSTM(hidden, hidden, layers, batch_first=True,
                                   dropout=self.dropout if layers > 1 else 0.)
        elif name == "Transformer":
            self.position = nn.Embedding(data.history, hidden)  # Stop-order embeddings.
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(hidden, 4, hidden * 2, self.dropout, batch_first=True),
                layers, enable_nested_tensor=False)
        else:
            # GCNConv combines each stop with itself and connected earlier stops.
            self.encoder = nn.ModuleList([GCNConv(hidden, hidden) for _ in range(layers)])
        self.head = (nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
                     if name == "GNN" else nn.Linear(hidden, 1))
        # Start with zero correction, so initial predictions equal the baseline.
        for parameter in (self.head[-1] if name == "GNN" else self.head).parameters():
            nn.init.zeros_(parameter)

    def forward(self, batch):
        values = self.project(batch.x) + sum(layer(batch.categories[:, i]) for i, layer in enumerate(self.categories))
        if self.name == "GNN":
            values = F.relu(values)
            for layer in self.encoder:
                values = F.dropout(F.relu(layer(values, batch.edge_index)), self.dropout, self.training)
            # Combine the target stop with an average of its full history.
            state = torch.cat([values[batch.ptr[1:] - 1], global_mean_pool(values, batch.batch)], dim=1)
        else:
            values, mask = to_dense_batch(values, batch.batch)  # Pad windows; True = real stop.
            if self.name == "LSTM":
                values = self.encoder(values)[0]
            else:
                values = self.encoder(values + self.position(torch.arange(values.size(1)))[None],
                                      src_key_padding_mask=~mask)
            state = values[torch.arange(len(values)), mask.sum(dim=1) - 1]  # Last real stop.
        return self.head(state).squeeze(-1)  # Scaled delay correction.
