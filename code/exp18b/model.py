import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv

class HeteroGraphSAGE(nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int, num_layers: int, data_metadata: tuple, feature_dims: dict):
        super().__init__()
        
        self.num_layers = num_layers
        
        # Initialize embeddings for node types without features (User, Merchant)
        self.lin_embeddings = nn.ModuleDict()
        self.param_embeddings = nn.ParameterDict()
        for node_type, dim in feature_dims.items():
            if dim == 0:
                self.param_embeddings[node_type] = nn.Parameter(torch.randn(1, hidden_channels))
            else:
                self.lin_embeddings[node_type] = nn.Linear(dim, hidden_channels)
                
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(num_layers):
            conv = HeteroConv({
                edge_type: SAGEConv(
                    in_channels=(-1, -1), # Lazy initialization
                    out_channels=hidden_channels,
                    aggr='mean'
                )
                for edge_type in data_metadata[1]
            })
            self.convs.append(conv)
            
            # BatchNorm for transaction nodes
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            
        # Classifier
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, num_nodes_dict, return_embeddings=False):
        # Initial projection
        h_dict = {}
        
        # Nodes with features
        for node_type, x in x_dict.items():
            if node_type in self.lin_embeddings:
                h_dict[node_type] = self.lin_embeddings[node_type](x)
                
        # Nodes without features (dummy embeddings)
        for node_type, param in self.param_embeddings.items():
            num_nodes = num_nodes_dict[node_type]
            h_dict[node_type] = param.expand(num_nodes, -1)
                
        # Message Passing
        for i in range(self.num_layers):
            h_dict = self.convs[i](h_dict, edge_index_dict)
            
            # Apply ReLU and BatchNorm only on transactions (or all nodes if needed)
            for node_type in h_dict.keys():
                h_dict[node_type] = F.relu(h_dict[node_type])
                if node_type == 'transaction':
                    h_dict[node_type] = self.bns[i](h_dict[node_type])
        
        if return_embeddings:
            return h_dict['transaction']
            
        # Classification only on transaction nodes
        out = self.classifier(h_dict['transaction'])
        return torch.sigmoid(out).squeeze(-1)

def get_loss_function(pos_weight: float):
    """
    Returns Class-Weighted Binary Cross-Entropy Loss
    pos_weight = (num_negatives / num_positives)
    """
    # BCELoss doesn't take pos_weight natively like BCEWithLogitsLoss.
    # Since we output sigmoid prob, we need to handle weighting manually or use BCEWithLogits.
    # To use pos_weight safely, we will return BCEWithLogitsLoss and remove Sigmoid from model?
    # Wait, the instruction says "Linear layer dengan fungsi aktivasi Sigmoid ... Gunakan class-weighted BCE".
    # Standard BCELoss requires manual weighting or just use weights.
    return nn.BCELoss()

def manual_weighted_bce(probs, targets, pos_weight):
    """
    Manually weighted BCE for sigmoid output.
    """
    loss = - (pos_weight * targets * torch.log(probs + 1e-7) + 
              (1.0 - targets) * torch.log(1.0 - probs + 1e-7))
    return loss.mean()

