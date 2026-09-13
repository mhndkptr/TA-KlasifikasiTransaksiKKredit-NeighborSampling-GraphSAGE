"""EXP12 topology/importance scores on one frozen training graph.

The optional thesis comparison uses the same chronological train cutoff for
all three strategies. Neither validation/test transactions nor their labels
enter the statistics below.
"""
import numpy as np

from ..legacy import EXP12_DIR  # Ensures the audited EXP12 package is importable.
from exp12.sampling.weights import entity_ppr_three_steps, historical_topology


DEFAULTS = {
    'topology_alpha': .5,
    'topology_smoothing': 20.,
    'topology_balance': True,
    'importance_gamma': .3,
    'importance_degree_mode': 'projected_transaction',
    'ppr_beta': .15,
}


class FrozenWeightContext:
    """Train-cutoff degrees, fraud counts, and user-merchant pair counts."""

    def __init__(self, store, fit_end, config=None):
        if not 0 < fit_end <= store.n:
            raise ValueError('Cutoff bobot tidak valid')
        self.store = store
        self.fit_end = int(fit_end)
        self.config = {**DEFAULTS, **(config or {})}
        cfg = self.config
        if not 0 <= cfg['topology_alpha'] <= 1 or cfg['topology_smoothing'] <= 0:
            raise ValueError('Parameter topology tidak valid')
        if not 0 <= cfg['importance_gamma'] <= 1 or not 0 <= cfg['ppr_beta'] <= 1:
            raise ValueError('Parameter importance tidak valid')
        if cfg['importance_degree_mode'] not in {'literal_transaction','projected_transaction'}:
            raise ValueError('Mode degree tidak valid')
        users = np.asarray(store.users[:fit_end], dtype=np.int32)
        merchants = np.asarray(store.merchants[:fit_end], dtype=np.int32)
        labels = np.asarray(store.labels[:fit_end], dtype=np.int8)
        self.total_fraud = int(labels.sum())
        entity_count = store.num_users + store.num_merchants
        self.degree = (np.bincount(users, minlength=store.num_users).astype(np.int64))
        self.degree = np.r_[self.degree,
                            np.bincount(merchants, minlength=store.num_merchants).astype(np.int64)]
        self.fraud = np.r_[np.bincount(users, weights=labels, minlength=store.num_users),
                           np.bincount(merchants, weights=labels, minlength=store.num_merchants)]
        # np.unique sorts integer pair keys; the inverse gives the exact count
        # for each training transaction without materializing a graph object.
        pair_keys = users.astype(np.int64)*max(store.num_merchants,1)+merchants
        _, inverse, counts = np.unique(pair_keys, return_inverse=True, return_counts=True)
        self.pair_count = counts[inverse].astype(np.int32)
        self.history_nodes = fit_end + int((self.degree > 0).sum())
        if len(self.degree) != entity_count:
            raise AssertionError('Jumlah entity tidak konsisten')

    def scores(self, entity, transactions, strategy):
        tx = np.asarray(transactions, dtype=np.int64)
        if not len(tx):
            return np.empty(0, dtype=np.float64)
        if strategy not in {'topology','importance'} or tx.min() < 0 or tx.max() >= self.fit_end:
            raise ValueError('Strategi/calon tetangga melewati cutoff train')
        store = self.store
        other = (store.num_users + np.asarray(store.merchants[tx], dtype=np.int64)
                 if entity < store.num_users else
                 np.asarray(store.users[tx], dtype=np.int64))
        d_root = np.full(len(tx), self.degree[entity], dtype=np.float64)
        d_other = self.degree[other].astype(np.float64)
        pairs = self.pair_count[tx]
        cfg = self.config
        if strategy == 'topology':
            values = historical_topology(
                d_root, d_other, pairs, self.fraud[entity], self.fraud[other],
                np.asarray(store.labels[tx]), self.fit_end, self.total_fraud, cfg)
        else:
            ppr = entity_ppr_three_steps(d_root, d_other, pairs, cfg['ppr_beta'])
            if cfg['importance_degree_mode'] == 'projected_transaction':
                centrality = (d_root+d_other-pairs-1)/max(self.fit_end-1,1)
            else:
                centrality = 2/max(self.history_nodes-1,1)
            values = cfg['importance_gamma']*centrality + (1-cfg['importance_gamma'])*ppr
        values = np.maximum(np.asarray(values,dtype=np.float64),1e-12)
        if not np.isfinite(values).all():
            raise ValueError('Bobot sampling tidak finite')
        return values


def weighted_choice(rng, lo, hi, wanted, weights):
    """Exact weighted sampling without replacement over one eligible CSR row."""
    count = hi-lo
    if count <= wanted:
        return list(range(lo,hi))
    probabilities = weights/weights.sum(dtype=np.float64)
    return (lo+rng.choice(count,size=wanted,replace=False,p=probabilities)).tolist()
