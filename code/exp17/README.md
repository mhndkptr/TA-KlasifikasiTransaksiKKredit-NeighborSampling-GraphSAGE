# Exp17: Inductive Graph Representation Learning with Sampling Comparisons

This directory contains `exp17`, an experimental pipeline for large-scale credit card fraud classification using Graph Neural Networks (GNNs) and XGBoost. It implements a two-stage architecture: GraphSAGE operates on a heterogeneous tripartite graph to extract structural embeddings, and XGBoost classifies the combined graph embeddings and original features.

## Key Features
- **Heterogeneous Graph:** Bipartite/Tripartite design where Transaction nodes connect to User and Merchant nodes. Only Transaction nodes carry features and labels.
- **Graph-Level Undersampling:** Subsamples normal transactions while retaining all fraud transactions to mitigate extreme class imbalance during training, matching the reference methodology.
- **Custom Neighbor Sampling:** Allows comparing the performance and computational cost (inference time) of three distinct neighborhood sampling strategies during inductive inference.

## Project Structure
- `data_preprocessing.py`: Loads the raw dataset, processes timestamps, applies a strict chronological temporal split (Train, Validation, Test), encodes categorical features, and scales numerical values.
- `graph_builder.py`: Constructs the PyTorch Geometric `HeteroData` graph object. It maps CSV relationships into edges and implements the exact Graph-Level Undersampling strategy.
- `samplers.py`: Contains the logic for the three neighbor sampling algorithms. It computes dynamic edge weights based on structural properties which are then used by the PyG `NeighborLoader`.
- `model.py`: Implements the `HeteroGraphSAGE` neural network (2 layers, Mean Aggregator, 256 hidden units, ReLU, BatchNorm) and a custom manually-weighted Binary Cross-Entropy (BCE) loss.
- `main.py`: The entry point script that chains the pipeline components together, trains the model, and evaluates its inductive capabilities on the holdout test set using F1-Score, Recall, and AUPRC.

## Sampling Methods Explained

1. **Uniform Random Sampling (`uniform`)**: 
   The baseline approach where every neighbor has an equal probability of being sampled during the message-passing aggregation phase.

2. **Topology-Aware Sampling (`topology`)**:
   Aims to sample neighbors based on structural similarity (Jaccard Similarity / Local Homophily). Since this is a bipartite structure, the implementation approximates Jaccard overlap by assigning higher weights to nodes with critical structural roles (e.g., inverse-degree weighting simulating structural homophily).

3. **Importance-Based Sampling (`importance`)**:
   Prioritizes neighbors based on their overall importance in the local network topology. This uses Degree Centrality (and acts as a localized proxy for Personalized PageRank limits) to ensure that hub nodes (e.g., highly active merchants) are preserved in the computational graph.

## How to Run the Pipeline

Make sure you have installed the requirements (PyTorch, PyTorch Geometric, Pandas, Scikit-learn).
The script assumes the dataset is located at `../../dataset/credit_card_transactions-ibm_v2.csv`.

To execute the pipeline and compare the three sampling methods, run the following commands from this directory:

### 1. Run Baseline (Uniform Sampling)
```bash
python main.py --sampling uniform --epochs 5
```

### 2. Run Topology-Aware Sampling
```bash
python main.py --sampling topology --epochs 5
```

### 3. Run Importance-Based Sampling
```bash
python main.py --sampling importance --epochs 5
```

*(Note: If you are running on a machine with limited memory, you can append `--nrows 100000` to load only a subset of the data for testing the pipeline functionality).*
