import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv

class HeteroGraphSAGE(nn.Module):
    def __init__(
        self, 
        hidden_channels: int, 
        out_channels: int, 
        num_layers: int, 
        data_metadata: tuple, 
        feature_dims: dict, 
        aggr: str = 'mean',
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.aggr = aggr
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        # Inisialisasi proyeksi linear untuk tiap tipe node
        self.lin_embeddings = nn.ModuleDict()
        self.param_embeddings = nn.ParameterDict()
        for node_type, dim in feature_dims.items():
            if dim == 0:
                self.param_embeddings[node_type] = nn.Parameter(torch.randn(1, hidden_channels))
            else:
                self.lin_embeddings[node_type] = nn.Linear(dim, hidden_channels)
                
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: SAGEConv(
                    in_channels=(-1, -1), # Lazy initialization
                    out_channels=hidden_channels,
                    aggr=aggr
                )
                for edge_type in data_metadata[1]
            })
            self.convs.append(conv)
            
            # BatchNorm untuk node transaction
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            
        # Classifier linear bawaan untuk monitoring val AUPRC selama pelatihan GNN
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, num_nodes_dict, return_embeddings=False):
        # 1. Proyeksi fitur awal
        h_dict = {}
        
        # Node yang memiliki fitur (transaction, user, merchant)
        for node_type, x in x_dict.items():
            if node_type in self.lin_embeddings:
                h_dict[node_type] = self.lin_embeddings[node_type](x)
            elif node_type in self.param_embeddings:
                num_nodes = num_nodes_dict[node_type]
                h_dict[node_type] = self.param_embeddings[node_type].expand(num_nodes, -1)
                
        # 2. GraphSAGE Message Passing
        for i in range(self.num_layers):
            h_dict = self.convs[i](h_dict, edge_index_dict)
            
            if 'transaction' in h_dict and h_dict['transaction'].size(0) > 0:
                # Terapkan BatchNorm jika batch_size > 1
                if h_dict['transaction'].size(0) > 1:
                    h_dict['transaction'] = self.bns[i](h_dict['transaction'])
                h_dict['transaction'] = F.relu(h_dict['transaction'])
                h_dict['transaction'] = self.dropout(h_dict['transaction'])
                
        tx_repr = h_dict['transaction']
        
        if return_embeddings:
            return tx_repr
            
        logits = self.classifier(tx_repr).squeeze(-1)
        probs = torch.sigmoid(logits)
        return probs

def manual_weighted_bce(probs, targets, pos_weight):
    """
    Class-weighted Binary Cross-Entropy loss yang stabil secara numerik.
    """
    eps = 1e-7
    probs = torch.clamp(probs, eps, 1.0 - eps)
    loss = - (pos_weight * targets * torch.log(probs) + (1.0 - targets) * torch.log(1.0 - probs))
    return loss.mean()

def manual_unweighted_bce(probs, targets):
    """
    Unweighted BCE loss untuk monitoring loss objektif alami.
    """
    eps = 1e-7
    probs = torch.clamp(probs, eps, 1.0 - eps)
    loss = - (targets * torch.log(probs) + (1.0 - targets) * torch.log(1.0 - probs))
    return loss.mean()
