import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

def calculate_degree_centrality(data: HeteroData, edge_type: tuple):
    """Menghitung degree centrality untuk source nodes pada edge type tertentu."""
    src, rel, dst = edge_type
    edge_index = data[edge_type].edge_index
    num_nodes = data[src].num_nodes
    degree = torch.zeros(num_nodes, dtype=torch.float)
    degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1]))
    return degree

def apply_uniform_sampling(data: HeteroData):
    """
    Baseline: Uniform Random Sampling.
    Memberikan probabilitas yang seragam (1.0) untuk seluruh edge.
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
    Topology-Aware Sampling (Inverse Sublinear Log-Degree Centrality - v5 Stabilized).
    Menggunakan transformasi 1 / log1p(degree) untuk memprioritaskan transaksi pada node berderajat rendah
    dan meredam dominasi merchant super-hub tanpa memicu anomali bobot.
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
    Menggunakan transformasi sublinear log1p(degree) untuk mengidentifikasi hub transaksi
    secara proporsional dan teratur.
    """
    first_edge = data.edge_types[0]
    if hasattr(data[first_edge], 'edge_weight') and data[first_edge].edge_weight is not None:
        return data
    for edge_type in data.edge_types:
        deg_src = calculate_degree_centrality(data, edge_type)
        src_nodes = data[edge_type].edge_index[0]
        
        # Sublinear scaling log1p
        weights = torch.log1p(deg_src[src_nodes])
        min_val = weights.min()
        max_val = weights.max()
        if (max_val - min_val) > 1e-5:
            weights = (weights - min_val) / (max_val - min_val)
        else:
            weights = torch.zeros_like(weights)
        data[edge_type].edge_weight = weights + 0.1
    return data

def get_neighbor_loader(
    data: HeteroData, 
    method: str = "uniform", 
    batch_size: int = 1024, 
    shuffle: bool = False, 
    num_neighbors: list = [25, 10]
):
    """
    Membangun NeighborLoader PyG dengan bobot edge sampling yang telah disesuaikan.
    Target node: ('transaction', input_nodes).
    """
    method = method.lower()
    if method == "uniform":
        data = apply_uniform_sampling(data)
    elif method == "topology":
        data = apply_topology_aware_sampling(data)
    elif method == "importance":
        data = apply_importance_based_sampling(data)
    else:
        raise ValueError(f"Metode sampling '{method}' tidak didukung. Pilih 'uniform', 'topology', atau 'importance'.")
        
    num_neighbors_dict = {edge_type: num_neighbors for edge_type in data.edge_types}
    
    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors_dict,
        batch_size=batch_size,
        input_nodes=('transaction', torch.arange(data['transaction'].num_nodes)),
        shuffle=shuffle,
        num_workers=0
    )
    return loader
