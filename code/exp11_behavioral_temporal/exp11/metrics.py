"""Binary metrics, validation-only calibration and historical camouflage audit."""
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score


def binary_metrics(labels, probability, threshold=0.5):
    labels, probability = np.asarray(labels), np.asarray(probability)
    if len(labels) != len(probability) or not len(labels) or not np.isfinite(probability).all():
        raise ValueError("Label/probabilitas kosong, tidak sejajar, atau non-finite")
    predicted = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel().tolist()
    precision, recall = tp / max(tp+fp, 1), tp / max(tp+fn, 1)
    f1_fraud = 2*tp / max(2*tp+fp+fn, 1)
    f1_normal = 2*tn / max(2*tn+fp+fn, 1)
    specificity = tn / max(tn+fp, 1)
    prevalence = (tp+fn) / len(labels)
    ap = float(average_precision_score(labels, probability)) if np.sum(labels) else None
    return {"auprc": float(average_precision_score(labels, probability)) if np.sum(labels) else None,
        "roc_auc": float(roc_auc_score(labels, probability)) if len(np.unique(labels)) == 2 else None,
        "accuracy": (tn+tp) / len(labels), "precision": precision, "recall": recall,
        "f1": f1_fraud, 'f1_macro': (f1_fraud+f1_normal)/2,
        'gmean': float(np.sqrt(recall*specificity)), 'specificity': specificity,
        'false_positive_rate': 1-specificity, 'alert_rate': (tp+fp)/len(labels),
        'prevalence': prevalence, 'ap_lift': ap/prevalence if ap is not None else None,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def validation_score(labels, probability, mode='ap', bins=3):
    """Geometric mean of AP/prevalence across chronological validation blocks.

    Rank metrics cannot be evaluated on a one-class block. Exclude it explicitly
    and keep the counts in history; if every block is ineligible, stop.
    """
    ap = float(average_precision_score(labels, probability))
    records = []
    for positions in np.array_split(np.arange(len(labels)), min(bins, len(labels))):
        y, p = labels[positions], probability[positions]
        valid = len(np.unique(y)) == 2
        lift = float(average_precision_score(y, p)/y.mean()) if valid else None
        records.append({'count': len(y), 'fraud': int(y.sum()), 'ap_lift': lift})
    lifts = [r['ap_lift'] for r in records if r['ap_lift'] is not None]
    if mode == 'ap':
        return ap, records
    if not lifts:
        raise ValueError('Tidak ada validation block dengan dua kelas; gunakan lebih sedikit bins')
    return float(np.exp(np.mean(np.log(np.maximum(lifts, 1e-12))))), records


def score_diagnostics(labels, probability):
    result = {}
    for name, mask in [('all', np.ones(len(labels), dtype=bool)), ('normal', labels == 0), ('fraud', labels == 1)]:
        p = probability[mask]
        result[name] = {'count': len(p), 'mean': float(p.mean()) if len(p) else None,
            'quantiles': np.quantile(p, [0, .1, .5, .9, .99, 1]).tolist() if len(p) else [],
            'saturated_high_fraction': float((p >= .999).mean()) if len(p) else None}
    return result


def channel_metrics(graph, nodes, labels, probability, threshold):
    """Reporting only: locate train-fitted one-hot payment channel columns."""
    output = {}
    for column, name in enumerate(graph.metadata['feature_names']):
        if name.startswith('Use Chip='):
            mask = graph.features[nodes, column].numpy() > .5
            if mask.any():
                output[name.split('=', 1)[1]] = {'count': int(mask.sum()), 'fraud': int(labels[mask].sum()),
                    **binary_metrics(labels[mask], probability[mask], threshold)}
    return output


def calibrate_threshold(labels, probability, beta=1.0):
    if beta <= 0 or len(np.unique(labels)) != 2:
        raise ValueError("Kalibrasi memerlukan dua kelas dan beta positif")
    if not np.isfinite(probability).all():
        raise ValueError("Probabilitas kalibrasi non-finite")
    precision, recall, thresholds = precision_recall_curve(labels, probability)
    score = (1+beta**2)*precision[:-1]*recall[:-1] / np.maximum(beta**2*precision[:-1]+recall[:-1], 1e-15)
    # On an exact tie, prefer fewer alerts (the larger threshold).
    best = np.flatnonzero(score == score.max())[-1]
    return float(thresholds[best]), {"f_beta": float(score[best]), "precision": float(precision[best]), "recall": float(recall[best])}


def long_tail_mask(graph, nodes, cutoff):
    """Fraud queries with >cutoff normal TRAIN transactions in their 2-hop union.

    Direct user/merchant neighbors have no ground-truth fraud label. Unknown
    history is not treated as normal; duplicate user+merchant matches count once.
    """
    nodes = np.asarray(nodes)
    labels = graph.labels.numpy()
    entities, col, rowptr = graph.entities.numpy(), graph.col.numpy(), graph.rowptr.numpy()
    degree, fraud_count = graph.degree.numpy(), graph.fraud_count.numpy()
    mask = np.zeros(len(nodes), dtype=bool)
    unknown = 0
    for position in np.flatnonzero(labels[nodes] == 1):
        root = nodes[position]
        user, merchant = entities[root]
        user_history = col[rowptr[user]:rowptr[user+1]]
        shared = user_history[entities[user_history, 1] == merchant]
        total = int(degree[user]+degree[merchant]-len(shared))
        fraud = int(fraud_count[user]+fraud_count[merchant]-labels[shared].sum())
        if total == 0:
            unknown += 1
        else:
            mask[position] = (total-fraud)/total > cutoff
    return mask, unknown


def temporal_metrics(labels, probability, timestamps, threshold, bins=5):
    output = []
    for positions in np.array_split(np.arange(len(labels)), min(bins, len(labels))):
        output.append({"count": len(positions), "fraud": int(labels[positions].sum()),
            "start_ns": int(timestamps[positions[0]]), "end_ns": int(timestamps[positions[-1]]),
            **binary_metrics(labels[positions], probability[positions], threshold)})
    return output
