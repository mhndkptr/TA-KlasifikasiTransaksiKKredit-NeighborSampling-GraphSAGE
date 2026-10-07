from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import sparse


@dataclass
class HinSAGEResult:
    train_embeddings: np.ndarray
    holdout_embeddings: np.ndarray
    history: list[dict]
    backend: str
    model: object


def train_stellargraph(train_graph, combined_graph, labels, gradient_local, validation_local, holdout_local,
                       *, embedding_size=64, num_samples=(2, 32), batch_size=50, epochs=10, seed=42):
    """Execute the historical library path. Requires the dedicated Python <3.9 environment."""
    try:
        import pandas as pd
        import stellargraph as sg
        import tensorflow as tf
        from stellargraph.layer import HinSAGE
        from stellargraph.mapper import HinSAGENodeGenerator
    except ImportError as exc:
        raise RuntimeError("Backend stellargraph memerlukan environment.reference.yml") from exc

    def convert(graph):
        tx_ids = [f"t:{int(x)}" for x in graph.transaction_rows]
        client_ids = [f"c:{i}" for i in range(graph.num_clients)]
        merchant_ids = [f"m:{i}" for i in range(graph.num_merchants)]
        nodes = {
            "transaction": pd.DataFrame(graph.transaction_features.toarray(), index=tx_ids),
            "client": pd.DataFrame(np.ones((graph.num_clients, 1), np.float32), index=client_ids),
            "merchant": pd.DataFrame(np.ones((graph.num_merchants, 1), np.float32), index=merchant_ids),
        }
        edges = pd.DataFrame({"source": np.repeat(tx_ids, 2),
                              "target": np.ravel(np.column_stack([
                                  [client_ids[x] for x in graph.client_codes],
                                  [merchant_ids[x] for x in graph.merchant_codes]]))})
        return sg.StellarGraph(nodes=nodes, edges=edges), tx_ids

    tf.random.set_seed(seed)
    stellar_train, train_ids = convert(train_graph)
    generator = HinSAGENodeGenerator(stellar_train, batch_size, list(num_samples),
                                     head_node_type="transaction", seed=seed)
    y = pd.Series(np.asarray(labels)[train_graph.transaction_rows], index=train_ids)
    grad_ids = [train_ids[i] for i in gradient_local]
    val_ids = [train_ids[i] for i in validation_local]
    model_def = HinSAGE(layer_sizes=[embedding_size] * len(num_samples), generator=generator, dropout=0)
    x_inp, x_out = model_def.in_out_tensors()
    prediction = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32")(x_out)
    classifier = tf.keras.Model(inputs=x_inp, outputs=prediction)
    classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="binary_crossentropy")
    hist = classifier.fit(generator.flow(grad_ids, y.loc[grad_ids], shuffle=True), epochs=epochs,
                          validation_data=generator.flow(val_ids, y.loc[val_ids]), shuffle=False, verbose=1)
    encoder = tf.keras.Model(inputs=x_inp, outputs=x_out)
    train_emb = encoder.predict(generator.flow(train_ids, shuffle=False), verbose=1)

    stellar_all, all_ids = convert(combined_graph)
    infer = HinSAGENodeGenerator(stellar_all, batch_size, list(num_samples),
                                 head_node_type="transaction", seed=seed)
    hold_ids = [all_ids[i] for i in holdout_local]
    held_emb = encoder.predict(infer.flow(hold_ids, shuffle=False), verbose=1)
    history = [{key: float(values[i]) for key, values in hist.history.items()}
               for i in range(len(hist.history.get("loss", [])))]
    return HinSAGEResult(train_emb, held_emb, history, "stellargraph_1.2.1", classifier)


def train_torch_compat(train_graph, combined_graph, labels, gradient_local, validation_local, holdout_local,
                       *, embedding_size=64, num_samples=(2, 32), batch_size=50, epochs=10, seed=42,
                       device="cpu", learning_rate=1e-3):
    """Relation-aware compatibility backend; artifacts never label it as StellarGraph parity."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    in_dim = train_graph.transaction_features.shape[1]

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.tx = nn.Linear(in_dim, embedding_size)
            self.client_self = nn.Linear(1, embedding_size)
            self.merchant_self = nn.Linear(1, embedding_size)
            self.client_neigh = nn.Linear(in_dim, embedding_size, bias=False)
            self.merchant_neigh = nn.Linear(in_dim, embedding_size, bias=False)
            self.out = nn.Linear(embedding_size * 3, embedding_size)
            self.pred = nn.Linear(embedding_size, 1)

        def embed(self, graph, roots, random_state):
            root_raw = torch.as_tensor(graph.transaction_features[roots].toarray(), dtype=torch.float32, device=dev)
            root_hidden = F.relu(self.tx(root_raw))
            client_vectors, merchant_vectors = [], []
            fanout = int(num_samples[-1])
            for root in roots:
                c = graph.client_codes[root]; m = graph.merchant_codes[root]
                cn, mn = graph.client_neighbors[c], graph.merchant_neighbors[m]
                cs = random_state.choice(cn, fanout, replace=len(cn) < fanout)
                ms = random_state.choice(mn, fanout, replace=len(mn) < fanout)
                cx = torch.as_tensor(graph.transaction_features[cs].mean(axis=0), dtype=torch.float32, device=dev)
                mx = torch.as_tensor(graph.transaction_features[ms].mean(axis=0), dtype=torch.float32, device=dev)
                one = torch.ones(1, device=dev)
                client_vectors.append(F.relu(self.client_self(one) + self.client_neigh(cx.flatten())))
                merchant_vectors.append(F.relu(self.merchant_self(one) + self.merchant_neigh(mx.flatten())))
            combined = torch.cat([root_hidden, torch.stack(client_vectors), torch.stack(merchant_vectors)], dim=1)
            return F.normalize(self.out(combined), p=2, dim=1)

    model = Model().to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    y = np.asarray(labels, dtype=np.float32)[train_graph.transaction_rows]
    history = []
    for epoch in range(epochs):
        order = rng.permutation(gradient_local)
        model.train(); losses = []
        for start in range(0, len(order), batch_size):
            roots = order[start:start + batch_size]
            optimizer.zero_grad()
            emb = model.embed(train_graph, roots, rng)
            target = torch.as_tensor(y[roots], device=dev)
            loss = F.binary_cross_entropy_with_logits(model.pred(emb).flatten(), target)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for start in range(0, len(validation_local), batch_size):
                roots = validation_local[start:start + batch_size]
                if not len(roots): continue
                emb = model.embed(train_graph, roots, np.random.default_rng(seed + epoch + 10_000))
                target = torch.as_tensor(y[roots], device=dev)
                val_losses.append(float(F.binary_cross_entropy_with_logits(model.pred(emb).flatten(), target).cpu()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)),
                        "val_loss": float(np.mean(val_losses)) if val_losses else None})

    def embed_all(graph, ids):
        output = []
        model.eval()
        fixed = np.random.default_rng(seed + 123456)
        with torch.no_grad():
            for start in range(0, len(ids), batch_size):
                output.append(model.embed(graph, ids[start:start + batch_size], fixed).cpu().numpy())
        return np.concatenate(output) if output else np.empty((0, embedding_size), np.float32)

    train_emb = embed_all(train_graph, np.arange(train_graph.num_transactions))
    held_emb = embed_all(combined_graph, np.asarray(holdout_local, dtype=np.int64))
    return HinSAGEResult(train_emb, held_emb, history, "pytorch_hinsage_compat", model)
