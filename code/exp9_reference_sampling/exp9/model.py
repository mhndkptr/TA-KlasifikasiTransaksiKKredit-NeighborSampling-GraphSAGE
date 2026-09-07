"""Two mean GraphSAGE layers, BatchNorm, ReLU and a scalar classifier."""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(self, input_channels, hidden=256, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(input_channels, hidden, aggr="mean"), SAGEConv(hidden, hidden, aggr="mean")])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden), nn.BatchNorm1d(hidden)])
        self.classifier = nn.Linear(hidden, 1)
        self.dropout = dropout

    def activate(self, layer, values):
        return F.dropout(F.relu(self.norms[layer](values)), p=self.dropout, training=self.training)

    def forward(self, features, blocks):
        for i, block in enumerate(blocks):
            features = self.activate(i, self.convs[i]((features, features[block.self_index]), block.edge_index))
        return self.classifier(features).squeeze(-1)

    def from_mean(self, layer, self_features, mean_neighbors):
        # Equivalent to SAGEConv with mean aggregator and the default root path.
        conv = self.convs[layer]
        return self.activate(layer, conv.lin_l(mean_neighbors) + conv.lin_r(self_features))

    def classify_transactions(self, features, raw_entity_mean, hidden_entity_mean):
        first = self.from_mean(0, features, raw_entity_mean)
        second = self.from_mean(1, first, hidden_entity_mean)
        return self.classifier(second).squeeze(-1)
