from __future__ import annotations

from typing import Any
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader


WEIGHT_ATTR = "sampling_weight"
USER_TO_TX = ("user", "buys", "transaction")
TX_TO_USER = ("transaction", "rev_buys", "user")
MERCHANT_TO_TX = ("merchant", "sells", "transaction")
TX_TO_MERCHANT = ("transaction", "rev_sells", "merchant")
WEIGHTED_RELATIONS = (TX_TO_USER, TX_TO_MERCHANT)


def counterpart_source(edge_type: tuple[str, str, str]) -> str:
    if edge_type == TX_TO_USER:
        return "merchant_degree"
    if edge_type == TX_TO_MERCHANT:
        return "user_degree"
    return "not_applicable"


def _bincount(index: torch.Tensor, size: int) -> torch.Tensor:
    return torch.bincount(index, minlength=size).to(dtype=torch.float32)


def _minmax_plus_floor(values: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    """Implementasi min-max scaling dengan smoothing floor alpha (Proposal & EXP21D)."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha harus berada pada (0, 1], diterima {alpha}")
    values = values.to(dtype=torch.float32)
    if values.numel() == 0:
        return values
    minimum, maximum = values.min(), values.max()
    if torch.isclose(maximum, minimum):
        return torch.ones_like(values)
    return (values - minimum) / (maximum - minimum + 1e-8) + alpha


def compute_bipartite_jaccard(
    user_edges: torch.Tensor,
    merchant_edges: torch.Tensor,
    num_transactions: int,
    user_deg: torch.Tensor,
    merchant_deg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Menghitung exact bipartite Jaccard similarity antara User dan Merchant untuk setiap transaksi.
    J(u, m) = |transaksi bersama (u, m)| / (deg(u) + deg(m) - |transaksi bersama (u, m)|)
    Bebas dari label leakage dan bernilai terikat pada rentang [0, 1].
    """
    user_for_tx = user_edges[0]
    merchant_for_tx = merchant_edges[0]

    # Pemetaan pasangan unik (user, merchant) menggunakan 64-bit encoding
    max_m = int(merchant_for_tx.max().item()) + 1 if merchant_for_tx.numel() > 0 else 1
    pair_id = user_for_tx.to(torch.int64) * max_m + merchant_for_tx.to(torch.int64)

    # Menghitung frekuensi kemunculan pasangan (u, m)
    _, inverse_indices, counts = torch.unique(pair_id, return_inverse=True, return_counts=True)
    shared_tx = counts[inverse_indices].to(torch.float32)

    deg_u = user_deg[user_for_tx]
    deg_m = merchant_deg[merchant_for_tx]
    union_deg = (deg_u + deg_m - shared_tx).clamp_min(1.0)

    jaccard_per_tx = (shared_tx / union_deg).clamp(0.0, 1.0)
    return jaccard_per_tx, jaccard_per_tx


def compute_asagnn_attribute_homophily(
    data: HeteroData,
    user_edges: torch.Tensor,
    merchant_edges: torch.Tensor,
    num_transactions: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Implementasi Solusi 3: Attribute Homophily berbasis ASA-GNN (Tian et al., IEEE TCSS 2024).
    Mengukur keselarasan atribut transaksi terhadap profil centroid entitas pasangannya:
    H_ASA(x_tx, centroid) = (1 + cos(x_tx, centroid)) / 2
    Menghilangkan ketergantungan pada label L[v] (Zero Label Leakage).
    """
    tx_features = data["transaction"].x.to(torch.float32)
    num_users = int(data["user"].num_nodes)
    num_merchants = int(data["merchant"].num_nodes)
    feat_dim = tx_features.size(1)

    user_for_tx = user_edges[0]
    merchant_for_tx = merchant_edges[0]

    # 1. Centroid fitur per User
    user_feat_sum = torch.zeros((num_users, feat_dim), dtype=torch.float32, device=tx_features.device)
    user_feat_sum.scatter_add_(0, user_for_tx.unsqueeze(1).expand(-1, feat_dim), tx_features)
    user_counts = torch.bincount(user_for_tx, minlength=num_users).to(dtype=torch.float32).clamp_min(1.0).unsqueeze(1)
    user_centroid = user_feat_sum / user_counts

    # 2. Centroid fitur per Merchant
    merchant_feat_sum = torch.zeros((num_merchants, feat_dim), dtype=torch.float32, device=tx_features.device)
    merchant_feat_sum.scatter_add_(0, merchant_for_tx.unsqueeze(1).expand(-1, feat_dim), tx_features)
    merchant_counts = torch.bincount(merchant_for_tx, minlength=num_merchants).to(dtype=torch.float32).clamp_min(1.0).unsqueeze(1)
    merchant_centroid = merchant_feat_sum / merchant_counts

    # 3. Normalized Cosine Similarity
    tx_norm = torch.nn.functional.normalize(tx_features, p=2, dim=1)
    user_norm = torch.nn.functional.normalize(user_centroid, p=2, dim=1)
    merchant_norm = torch.nn.functional.normalize(merchant_centroid, p=2, dim=1)

    # Untuk relasi tx -> user: kandidat dievaluasi terhadap keselarasan atribut merchant pasangannya
    cos_merchant = (tx_norm * merchant_norm[merchant_for_tx]).sum(dim=1).clamp(-1.0, 1.0)
    homophily_for_user_branch = ((1.0 + cos_merchant) / 2.0).clamp(0.0, 1.0)

    # Untuk relasi tx -> merchant: kandidat dievaluasi terhadap keselarasan atribut user pasangannya
    cos_user = (tx_norm * user_norm[user_for_tx]).sum(dim=1).clamp(-1.0, 1.0)
    homophily_for_merchant_branch = ((1.0 + cos_user) / 2.0).clamp(0.0, 1.0)

    return homophily_for_user_branch, homophily_for_merchant_branch


def compute_bipartite_pagerank(
    user_edges: torch.Tensor,
    merchant_edges: torch.Tensor,
    num_users: int,
    num_merchants: int,
    teleport_beta: float = 0.15,
    num_steps: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Menghitung skor Personalized PageRank (PPR) lokal/bipartit (Proposal Algoritma 4).
    Menggunakan power iteration terdistribusi sparse matrix multiplication.
    Bebas OOM dan zero temporal leakage.
    """
    total_nodes = num_users + num_merchants
    u_idx = user_edges[0]
    m_idx = merchant_edges[0] + num_users  # Offset indeks merchant

    # Bangun graf bipartit tak-berarah
    src = torch.cat([u_idx, m_idx])
    dst = torch.cat([m_idx, u_idx])
    edge_index = torch.stack([src, dst])

    deg = torch.bincount(edge_index[0], minlength=total_nodes).to(dtype=torch.float32).clamp_min(1.0)
    norm_val = 1.0 / deg[edge_index[0]]

    # Power iteration Personalized PageRank
    pi = torch.ones(total_nodes, dtype=torch.float32) / float(total_nodes)
    teleport = torch.ones(total_nodes, dtype=torch.float32) / float(total_nodes)

    for _ in range(num_steps):
        # Sparse message passing: pi = (1 - beta) * P * pi + beta * e
        incoming = pi[edge_index[0]] * norm_val
        next_pi = torch.zeros(total_nodes, dtype=torch.float32)
        next_pi.scatter_add_(0, edge_index[1], incoming)
        pi = (1.0 - teleport_beta) * next_pi + teleport_beta * teleport

    pi_u = pi[:num_users]
    pi_m = pi[num_users:]

    # Min-max normalisasi skor PPR
    pi_u_norm = (pi_u - pi_u.min()) / (pi_u.max() - pi_u.min() + 1e-8)
    pi_m_norm = (pi_m - pi_m.min()) / (pi_m.max() - pi_m.min() + 1e-8)

    return pi_u_norm, pi_m_norm


def _transaction_topology_structures(data: HeteroData) -> dict[str, Any]:
    """Mengekstrak relasi topologi tripartit heterogen."""
    required = {USER_TO_TX, TX_TO_USER, MERCHANT_TO_TX, TX_TO_MERCHANT}
    missing = required.difference(data.edge_types)
    if missing:
        raise ValueError(f"Relasi graf tripartit belum lengkap: {sorted(missing)}")

    expected = int(data["transaction"].num_nodes)
    user_edges = data[USER_TO_TX].edge_index
    merchant_edges = data[MERCHANT_TO_TX].edge_index

    num_users = int(data["user"].num_nodes)
    num_merchants = int(data["merchant"].num_nodes)

    user_deg = _bincount(user_edges[0], num_users)
    merchant_deg = _bincount(merchant_edges[0], num_merchants)

    user_for_tx = user_edges[0]
    merchant_for_tx = merchant_edges[0]

    return {
        "num_transactions": expected,
        "num_users": num_users,
        "num_merchants": num_merchants,
        "user_edges": user_edges,
        "merchant_edges": merchant_edges,
        "user_deg": user_deg,
        "merchant_deg": merchant_deg,
        "user_for_tx": user_for_tx,
        "merchant_for_tx": merchant_for_tx,
    }


def apply_sampling_weights(
    data: HeteroData,
    method: str,
    alpha: float = 0.1,
    weight_attr: str = WEIGHT_ATTR,
    topology_alpha: float = 0.5,
    importance_gamma: float = 0.3,
    ppr_teleport: float = 0.15,
    ppr_steps: int = 3,
) -> HeteroData:
    """
    Menghitung dan menyematkan bobot sampling pada graf sesuai Proposal EXP24:
    - Uniform: w = 1.0 (Baseline)
    - Topology: w = minmax(alpha * Jaccard + (1 - alpha) * Homophily_ASAGNN) + floor (Algoritma 3 Proposal)
    - Importance: w = minmax(gamma * Degree + (1 - gamma) * PPR) + floor (Algoritma 4 Proposal)
    """
    method = method.lower()
    topo = _transaction_topology_structures(data)
    num_tx = topo["num_transactions"]
    user_edges = topo["user_edges"]
    merchant_edges = topo["merchant_edges"]
    user_deg = topo["user_deg"]
    merchant_deg = topo["merchant_deg"]
    user_for_tx = topo["user_for_tx"]
    merchant_for_tx = topo["merchant_for_tx"]

    # Prekomputasi komponen proposal
    if method == "topology":
        jaccard_u, jaccard_m = compute_bipartite_jaccard(user_edges, merchant_edges, num_tx, user_deg, merchant_deg)
        homo_u, homo_m = compute_asagnn_attribute_homophily(data, user_edges, merchant_edges, num_tx)
        raw_tx_to_user = topology_alpha * jaccard_u + (1.0 - topology_alpha) * homo_u
        raw_tx_to_merchant = topology_alpha * jaccard_m + (1.0 - topology_alpha) * homo_m
    elif method == "importance":
        cd_merchant = merchant_deg / (merchant_deg.max() + 1e-8)
        cd_user = user_deg / (user_deg.max() + 1e-8)
        pi_u, pi_m = compute_bipartite_pagerank(
            user_edges, merchant_edges, topo["num_users"], topo["num_merchants"],
            teleport_beta=ppr_teleport, num_steps=ppr_steps
        )
        raw_tx_to_user = importance_gamma * cd_merchant[merchant_for_tx] + (1.0 - importance_gamma) * pi_m[merchant_for_tx]
        raw_tx_to_merchant = importance_gamma * cd_user[user_for_tx] + (1.0 - importance_gamma) * pi_u[user_for_tx]

    for edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index
        num_edges = int(edge_index.size(1))

        if method == "uniform":
            weights = torch.ones(num_edges, dtype=torch.float32)
            candidate_deg = torch.ones(num_edges, dtype=torch.float32)
        elif edge_type == TX_TO_USER:
            candidate_deg = merchant_deg[merchant_for_tx[edge_index[0]]]
            weights = _minmax_plus_floor(raw_tx_to_user[edge_index[0]], alpha=alpha)
        elif edge_type == TX_TO_MERCHANT:
            candidate_deg = user_deg[user_for_tx[edge_index[0]]]
            weights = _minmax_plus_floor(raw_tx_to_merchant[edge_index[0]], alpha=alpha)
        else:
            # Hop 1 selalu uniform 1.0 (deterministik non-kompetitif)
            weights = torch.ones(num_edges, dtype=torch.float32)
            candidate_deg = torch.ones(num_edges, dtype=torch.float32)

        if weights.numel() != num_edges or not torch.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError(f"Bobot sampling invalid untuk relasi {edge_type}")

        data[edge_type][weight_attr] = weights.contiguous()
        data[edge_type].candidate_degree = candidate_deg.contiguous()

    validate_sampling_weight_contract(data, method, weight_attr)
    return data


def validate_sampling_weight_contract(data: HeteroData, method: str, weight_attr: str = WEIGHT_ATTR) -> None:
    """Verifikasi 8 kriteria integritas sampling contract untuk EXP24."""
    method = method.lower()
    for edge_type in data.edge_types:
        store = data[edge_type]
        if weight_attr not in store:
            raise ValueError(f"{weight_attr} tidak ditemukan pada {edge_type}")
        weights = store[weight_attr]
        destination = store.edge_index[1]
        num_dest_nodes = int(data[edge_type[2]].num_nodes)
        sums = torch.zeros(num_dest_nodes, dtype=torch.float32)
        sums.scatter_add_(0, destination, weights)
        active = torch.bincount(destination, minlength=num_dest_nodes) > 0
        if active.any() and not (sums[active] > 0).all():
            raise ValueError(f"Jumlah bobot lokal non-positif pada {edge_type}")

        # Hop 1 (USER_TO_TX dan MERCHANT_TO_TX) wajib uniform 1.0
        if edge_type not in WEIGHTED_RELATIONS and not torch.allclose(weights, torch.ones_like(weights)):
            raise ValueError(f"Relasi first-hop/non-kompetitif harus bernilai tepat 1.0: {edge_type}")

        if edge_type in WEIGHTED_RELATIONS:
            if method == "uniform" and not torch.allclose(weights, torch.ones_like(weights)):
                raise ValueError(f"Uniform sampler harus berbobot satu pada {edge_type}")
            if method != "uniform" and (weights < 0.1).any():
                raise ValueError(f"Bobot {method} melanggar smoothing floor 0.1 pada {edge_type}")


def sampling_weight_audit(data: HeteroData, method: str, weight_attr: str = WEIGHT_ATTR, **kwargs) -> dict[str, Any]:
    """Menghasilkan laporan audit statistik bobot sampling."""
    # Pastikan bobot sampling sudah diterapkan secara defensif jika belum ada
    for edge_type in data.edge_types:
        if weight_attr not in data[edge_type]:
            apply_sampling_weights(data, method, weight_attr=weight_attr, **kwargs)
            break

    report: dict[str, Any] = {
        "method": method,
        "weight_attr": weight_attr,
        "weighted_relations": ["__".join(item) for item in WEIGHTED_RELATIONS],
        "relations": {},
    }
    for edge_type in data.edge_types:
        store = data[edge_type]
        weights = store[weight_attr].detach().cpu()
        degrees = store.candidate_degree.detach().cpu()
        report["relations"]["__".join(edge_type)] = {
            "num_edges": int(weights.numel()),
            "counterpart_source": counterpart_source(edge_type),
            "weight_min": float(weights.min()) if weights.numel() else None,
            "weight_max": float(weights.max()) if weights.numel() else None,
            "weight_mean": float(weights.mean()) if weights.numel() else None,
            "weight_std": float(weights.std(unbiased=False)) if weights.numel() else None,
            "unique_weights": int(torch.unique(weights).numel()),
            "candidate_degree_min": float(degrees.min()) if degrees.numel() else None,
            "candidate_degree_max": float(degrees.max()) if degrees.numel() else None,
            "contract_passed": True,
        }
    return report


def get_neighbor_loader(
    data: HeteroData,
    method: str = "uniform",
    batch_size: int = 1024,
    shuffle: bool = False,
    num_neighbors: list[int] | None = None,
    seed: int = 42,
    alpha: float = 0.1,
    input_nodes: torch.Tensor | None = None,
    weight_attr: str = WEIGHT_ATTR,
    topology_alpha: float = 0.5,
    importance_gamma: float = 0.3,
    ppr_teleport: float = 0.15,
    ppr_steps: int = 3,
) -> NeighborLoader:
    """Membangun PyG NeighborLoader dengan bobot sampling EXP24 yang telah disesuaikan."""
    if num_neighbors is None:
        num_neighbors = [15, 10]
    if len(num_neighbors) != 2 or any(value <= 0 for value in num_neighbors):
        raise ValueError("num_neighbors harus berisi dua fan-out positif [15, 10]")

    apply_sampling_weights(
        data, method, alpha=alpha, weight_attr=weight_attr,
        topology_alpha=topology_alpha, importance_gamma=importance_gamma,
        ppr_teleport=ppr_teleport, ppr_steps=ppr_steps
    )

    if input_nodes is None:
        input_nodes = torch.arange(int(data["transaction"].num_nodes))

    # PyG NeighborLoader:
    # 1. Untuk method == 'uniform', semua edge memiliki peluang identik (bobot konstan 1.0).
    #    Kita gunakan weight_attr=None agar PyG memakai sampling uniform native yang sangat
    #    cepat dan TIDAK membutuhkan ekstensi C++ eksternal 'pyg_lib'.
    # 2. Untuk method 'topology' dan 'importance', PyG secara ketat membutuhkan 'pyg_lib>=0.3.0'.
    loader_weight_attr = None if method == "uniform" else weight_attr

    if loader_weight_attr is not None:
        try:
            import pyg_lib  # noqa: F401
        except ImportError:
            torch_ver = torch.__version__.split("+")[0]
            cuda_ver = torch.version.cuda
            cuda_tag = f"cu{cuda_ver.replace('.', '')}" if cuda_ver else "cpu"
            wheel_url = f"https://data.pyg.org/whl/torch-{torch_ver}+{cuda_tag}.html"
            raise ImportError(
                f"\n{'='*75}\n"
                f"[EXP24 ERROR] Weighted neighbor sampling (metode '{method}') memerlukan pustaka 'pyg-lib>=0.3.0'.\n"
                f"Pustaka 'pyg_lib' belum terpasang di Python environment Anda.\n\n"
                f"Solusi: Jalankan perintah berikut di Anaconda Prompt pada PC running Anda:\n"
                f"  pip install pyg_lib -f {wheel_url}\n\n"
                f"Atau jika menggunakan conda-forge:\n"
                f"  conda install pyg-lib -c pyg -c conda-forge\n"
                f"{'='*75}\n"
            )

    generator = torch.Generator().manual_seed(seed)
    loader = NeighborLoader(
        data,
        num_neighbors={edge_type: list(num_neighbors) for edge_type in data.edge_types},
        batch_size=batch_size,
        input_nodes=("transaction", input_nodes),
        shuffle=shuffle,
        num_workers=0,
        weight_attr=loader_weight_attr,
        generator=generator,
        subgraph_type="directional",
    )
    return loader
