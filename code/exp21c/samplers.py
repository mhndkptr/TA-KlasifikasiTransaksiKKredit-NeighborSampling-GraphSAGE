import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

def calculate_degree_centrality(data: HeteroData, edge_type: tuple):
    """Menghitung degree centrality sederhana untuk edge type tertentu."""
    src, rel, dst = edge_type
    edge_index = data[edge_type].edge_index
    num_nodes = data[src].num_nodes
    degree = torch.zeros(num_nodes, dtype=torch.float)
    degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1]))
    return degree

def apply_uniform_sampling(data: HeteroData):
    """
    Baseline: Uniform Random Sampling.
    Memberikan probabilitas yang sama (1.0) untuk semua edge.
    """
    first_edge = data.edge_types[0]
    if hasattr(data[first_edge], 'edge_weight') and data[first_edge].edge_weight is not None:
        return data
    for edge_type in data.edge_types:
        num_edges = data[edge_type].edge_index.size(1)
        data[edge_type].edge_weight = torch.ones(num_edges, dtype=torch.float)
    return data

def apply_topology_aware_sampling(data: HeteroData):
    """
    Topology-Aware Sampling (Inverse Log-Degree Centrality - v5 Stabilized).
    Menggunakan inverse sublinear log1p(degree) untuk memprioritaskan edge struktural
    tanpa disparitas bobot yang ekstrem.
    """
    first_edge = data.edge_types[0]
    if hasattr(data[first_edge], 'edge_weight') and data[first_edge].edge_weight is not None:
        return data
    for edge_type in data.edge_types:
        deg_src = calculate_degree_centrality(data, edge_type)
        src_nodes = data[edge_type].edge_index[0]
        
        # Sublinear scaling: log1p(degree)
        log_deg = torch.log1p(deg_src[src_nodes])
        weights = 1.0 / (log_deg + 1e-5)
        min_val = weights.min()
        max_val = weights.max()
        if (max_val - min_val) > 1e-5:
            weights = (weights - min_val) / (max_val - min_val)
        else:
            weights = torch.zeros_like(weights)
        data[edge_type].edge_weight = weights + 0.1
    return data

def apply_importance_based_sampling(data: HeteroData):
    """
    Importance-Based Sampling (Log-Degree Centrality Proxy - v5 Stabilized).
    Menggunakan transformasi sublinear log1p untuk meredam varians sampling
    akibat super-hub merchant.
    """
    first_edge = data.edge_types[0]
    if hasattr(data[first_edge], 'edge_weight') and data[first_edge].edge_weight is not None:
        return data
    for edge_type in data.edge_types:
        deg_src = calculate_degree_centrality(data, edge_type)
        src_nodes = data[edge_type].edge_index[0]
        
        # Sublinear scaling: log1p(degree)
        log_deg = torch.log1p(deg_src[src_nodes])
        min_val = log_deg.min()
        max_val = log_deg.max()
        if (max_val - min_val) > 1e-5:
            weights = (log_deg - min_val) / (max_val - min_val)
        else:
            weights = torch.zeros_like(log_deg)
        data[edge_type].edge_weight = weights + 0.1
    return data

def get_neighbor_loader(
    data: HeteroData, 
    num_neighbors: list, 
    batch_size: int, 
    shuffle: bool = True, 
    sampling_method: str = "uniform",
    input_nodes: torch.Tensor = None
) -> NeighborLoader:
    """
    Membangun PyG NeighborLoader dengan metode sampling yang dipilih.
    """
    if sampling_method == "uniform":
        data = apply_uniform_sampling(data)
    elif sampling_method == "topology":
        data = apply_topology_aware_sampling(data)
    elif sampling_method == "importance":
        data = apply_importance_based_sampling(data)
    else:
        raise ValueError(f"Metode sampling tidak dikenal: {sampling_method}. Pilih 'uniform', 'topology', atau 'importance'.")
        
    if input_nodes is None:
        input_nodes_arg = ('transaction', None)
    else:
        input_nodes_arg = ('transaction', input_nodes)
        
    return NeighborLoader(
        data,
        num_neighbors={edge_type: num_neighbors for edge_type in data.edge_types},
        batch_size=batch_size,
        input_nodes=input_nodes_arg,
        shuffle=shuffle
    )
