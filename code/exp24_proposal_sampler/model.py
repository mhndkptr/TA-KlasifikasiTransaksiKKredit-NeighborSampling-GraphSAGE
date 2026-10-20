import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv

class HeteroGraphSAGE(nn.Module):
    def __init__(
        self, 
        hidden_channels: int = 128, 
        out_channels: int = 1, 
        num_layers: int = 2, 
        data_metadata: tuple | None = None, 
        feature_dims: dict | None = None, 
        aggr: str = 'mean',
        dropout: float = 0.2,
        edge_types: list | tuple | None = None,
        in_channels_dict: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        
        # Penanganan fleksibel untuk data_metadata & edge_types
        if edge_types is not None:
            resolved_edge_types = list(edge_types)
        elif data_metadata is not None:
            resolved_edge_types = list(data_metadata[1])
        else:
            raise ValueError("Harus menyediakan data_metadata atau edge_types untuk HeteroGraphSAGE.")

        # Penanganan fleksibel untuk feature_dims & in_channels_dict
        resolved_feature_dims = feature_dims if feature_dims is not None else (in_channels_dict or {})
        
        self.num_layers = num_layers
        self.aggr = aggr
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        # Inisialisasi proyeksi linear untuk tiap tipe node
        self.lin_embeddings = nn.ModuleDict()
        self.param_embeddings = nn.ParameterDict()
        for node_type, dim in resolved_feature_dims.items():
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
                for edge_type in resolved_edge_types
            })
            self.convs.append(conv)
            
            # BatchNorm untuk node transaction
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            
        # Classifier linear bawaan untuk monitoring val AUPRC selama pelatihan GNN
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, num_nodes_dict=None, return_embeddings=False):
        if num_nodes_dict is None:
            num_nodes_dict = {ntype: x.size(0) for ntype, x in x_dict.items()}

        # 1. Proyeksi fitur awal
        h_dict = {}
        
        # Node yang memiliki fitur (transaction, user, merchant)
        for node_type, x in x_dict.items():
            if node_type in self.lin_embeddings:
                h_dict[node_type] = self.lin_embeddings[node_type](x)
            elif node_type in self.param_embeddings:
                num_nodes = num_nodes_dict.get(node_type, 1)
                h_dict[node_type] = self.param_embeddings[node_type].expand(num_nodes, -1)
                
        # 2. Message Passing Layer
        for i in range(self.num_layers):
            h_dict = self.convs[i](h_dict, edge_index_dict)
            if 'transaction' in h_dict and h_dict['transaction'].size(0) > 1:
                h_dict['transaction'] = self.bns[i](h_dict['transaction'])
            h_dict = {k: F.relu(v) for k, v in h_dict.items()}
            h_dict = {k: self.dropout(v) for k, v in h_dict.items()}
            
        tx_emb = h_dict['transaction']
        
        if return_embeddings:
            return tx_emb
            
        out = self.classifier(tx_emb)
        return out.squeeze(-1)

    def get_embeddings(self, x_dict, edge_index_dict, num_nodes_dict=None):
        return self.forward(x_dict, edge_index_dict, num_nodes_dict=num_nodes_dict, return_embeddings=True)

def manual_weighted_bce(logits, targets, pos_weight):
    """
    Weighted Binary Cross Entropy Loss stabil dengan koefisien pos_weight teredam.
    """
    max_val = torch.clamp(-logits, min=0)
    loss = (1 - targets) * logits + (1 + (pos_weight - 1) * targets) * (
        max_val + torch.log(torch.exp(-max_val) + torch.exp(-logits - max_val))
    )
    return loss.mean()

def manual_unweighted_bce(logits, targets):
    """
    Standar Binary Cross Entropy Loss tanpa pembobotan untuk evaluasi objektif.
    """
    max_val = torch.clamp(-logits, min=0)
    loss = (1 - targets) * logits + (
        max_val + torch.log(torch.exp(-max_val) + torch.exp(-logits - max_val))
    )
    return loss.mean()

