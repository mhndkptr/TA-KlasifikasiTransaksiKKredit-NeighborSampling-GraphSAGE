import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless rendering
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def load_sample_and_metrics(result_dir: Path, aggr: str):
    """
    Memuat sample embedding .npz dan file hasil .json untuk aggregator tertentu.
    """
    sample_file = result_dir / f"exp22_{aggr}_embeddings_sample.npz"
    result_file = result_dir / f"exp22_{aggr}_results.json"
    
    if not sample_file.exists():
        return None, None
        
    data = np.load(sample_file)
    embeddings = data['embeddings']
    labels = data['labels']
    
    metrics = None
    if result_file.exists():
        with open(result_file, 'r') as f:
            metrics = json.load(f)
            
    return (embeddings, labels), metrics

def calculate_cluster_separation(embeddings, labels):
    """
    Menghitung rasio separasi centroid antar-kelas vs variansi intra-kelas
    (Euclidean Distance between Fraud Centroid & Normal Centroid).
    """
    fraud_mask = (labels == 1)
    normal_mask = (labels == 0)
    
    if not np.any(fraud_mask) or not np.any(normal_mask):
        return 0.0
        
    c_fraud = embeddings[fraud_mask].mean(axis=0)
    c_normal = embeddings[normal_mask].mean(axis=0)
    
    dist = np.linalg.norm(c_fraud - c_normal)
    return float(dist)

def plot_tsne_comparison(result_dir: Path, output_file: Path):
    """
    Membuat visualisasi komparatif t-SNE 2D untuk agregator Mean vs Max (dan Sum jika ada).
    """
    aggregators = ['mean', 'max']
    # Cek apakah 'sum' juga ada
    if (result_dir / "exp22_sum_embeddings_sample.npz").exists():
        aggregators.append('sum')
        
    loaded_data = {}
    for aggr in aggregators:
        data_tuple, metrics = load_sample_and_metrics(result_dir, aggr)
        if data_tuple is not None:
            loaded_data[aggr] = (data_tuple, metrics)
            
    if not loaded_data:
        print(f"[WARNING] Tidak ditemukan berkas embedding sample (exp22_*_embeddings_sample.npz) di: {result_dir}")
        print("Pastikan pipeline training (main.py) telah dijalankan terlebih dahulu.")
        return False

    print(f"Menjalankan t-SNE embedding comparison untuk: {list(loaded_data.keys())}...")
    
    n_plots = len(loaded_data)
    fig, axes = plt.subplots(1, n_plots, figsize=(7.5 * n_plots, 6.5), dpi=300)
    if n_plots == 1:
        axes = [axes]
        
    fig.patch.set_facecolor('#FFFFFF')
    
    for idx, (aggr, (data_tuple, metrics)) in enumerate(loaded_data.items()):
        ax = axes[idx]
        embeddings, labels = data_tuple
        
        print(f"Menghitung t-SNE untuk {aggr.upper()} ({len(labels)} sampel)...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
        emb_2d = tsne.fit_transform(embeddings)
        
        sep_dist = calculate_cluster_separation(emb_2d, labels)
        
        normal_idx = np.flatnonzero(labels == 0)
        fraud_idx = np.flatnonzero(labels == 1)
        
        # Plot titik transaksi normal
        ax.scatter(
            emb_2d[normal_idx, 0], 
            emb_2d[normal_idx, 1], 
            c='#64748B', 
            alpha=0.25, 
            s=16, 
            label=f'Normal (N={len(normal_idx)})',
            edgecolors='none'
        )
        
        # Plot titik transaksi fraud (merah menyala dengan outline halus)
        ax.scatter(
            emb_2d[fraud_idx, 0], 
            emb_2d[fraud_idx, 1], 
            c='#E63946', 
            alpha=0.92, 
            s=48, 
            label=f'Fraud (N={len(fraud_idx)})',
            edgecolors='#1E293B',
            linewidths=0.6,
            zorder=5
        )
        
        # Format Subtitle dengan Metrik Evaluasi jika tersedia
        title_aggr = "Baseline: Mean Aggregator" if aggr == 'mean' else ("Proposed: Max Pooling (Elementwise Max)" if aggr == 'max' else "Ablation: Sum Aggregator")
        metric_str = ""
        if metrics:
            opt = metrics.get('metrics_optimal_threshold', {})
            ind = metrics.get('metrics_threshold_independent', {})
            f1 = opt.get('f1_score', 0.0)
            rec = opt.get('recall', 0.0) * 100
            aucpr = ind.get('auprc', 0.0)
            metric_str = f"\nAUCPR: {aucpr:.4f} | F1: {f1:.4f} | Recall: {rec:.1f}%"
            
        ax.set_title(f"{title_aggr}{metric_str}\nCentroid Separation: {sep_dist:.2f}", fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel("t-SNE Dimension 1", fontsize=10, fontweight='medium')
        ax.set_ylabel("t-SNE Dimension 2", fontsize=10, fontweight='medium')
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
        ax.set_facecolor('#F8FAFC')
        
        # Highlight anotasi temuan
        if aggr == 'mean':
            ax.text(0.03, 0.05, "Minority Feature Dilution:\nFraud tenggelam di klaster normal", 
                    transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec="#FCA5A5", alpha=0.9))
        elif aggr == 'max':
            ax.text(0.03, 0.05, "Anomaly Preservation:\nFraud terpisah jelas & terkonsentrasi", 
                    transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="#F0FDF4", ec="#86EFAC", alpha=0.9))

    plt.suptitle("Eksperimen 22: Separabilitas Representasi Embedding Transaksi (t-SNE)\n"
                 "Pengaruh Fungsi Agregasi GNN terhadap Ketahanan Minority Feature Dilution", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[SUCCESS] Visualisasi komparatif t-SNE tersimpan ke: {output_file}\n")
    return True

def main():
    parser = argparse.ArgumentParser(description="EXP22: Visualisasi komparasi t-SNE separabilitas embedding GNN")
    parser.add_argument('--result-dir', type=str, default='../../result/exp22', help='Direktori penyimpanan hasil exp22')
    parser.add_argument('--output-file', type=str, default='embeddings_tsne_comparison.png', help='Nama file gambar output')
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    output_file = result_dir / args.output_file
    plot_tsne_comparison(result_dir, output_file)

if __name__ == '__main__':
    main()
