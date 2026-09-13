"""Validation-only thresholds and disaggregated assessment diagnostics."""
import numpy as np
from .legacy import binary_metrics, calibrate_threshold


def fpr_threshold(labels, scores, limit):
    y = np.asarray(labels,dtype=np.int8)
    p = np.asarray(scores,dtype=np.float64)
    if not 0 <= limit <= 1 or not (y==0).any() or not np.isfinite(p).all():
        raise ValueError('FPR/calibration tidak valid')
    normal = np.sort(p[y==0])
    allowed = int(np.floor(limit*len(normal)))
    if allowed == 0:
        return float(np.nextafter(normal[-1],np.inf))
    # Strictly above the score at the excluded normal's position; ties are safe.
    first_excluded = normal[-allowed-1] if allowed < len(normal) else -np.inf
    threshold = float(np.nextafter(first_excluded,np.inf))
    if int((normal >= threshold).sum()) > allowed:
        raise AssertionError('FPR threshold exceeds calibration budget')
    return threshold


def quota_metrics(labels,scores,fraction):
    y = np.asarray(labels,dtype=np.int8)
    p = np.asarray(scores,dtype=np.float64)
    if not 0 < fraction <= 1: raise ValueError('Kuota harus dalam (0,1]')
    k = min(len(y),max(1,int(np.floor(len(y)*fraction))))
    order = np.argsort(-p,kind='stable')[:k]
    tp = int(y[order].sum())
    return {'alerts':k,'tp':tp,'precision':tp/k,
            'recall':tp/int(y.sum()) if y.sum() else None,
            'quota_fraction':fraction}


def thresholds(labels,scores):
    if len(np.unique(labels)) != 2:
        raise ValueError('Calibration memerlukan normal dan fraud')
    f2,details = calibrate_threshold(labels,scores,beta=2)
    values = {'f2':float(f2),'fpr_0_001':fpr_threshold(labels,scores,.001),
              'fpr_0_0005':fpr_threshold(labels,scores,.0005)}
    return values,details


def report(store,ids,scores,chosen_threshold):
    ids = np.asarray(ids,dtype=np.int64)
    y = np.asarray(store.labels[ids],dtype=np.int8)
    def metrics(labels, values):
        result = binary_metrics(labels,values,chosen_threshold)
        if not np.any(labels == 1):
            for key in ('recall','f1','gmean','ap_lift'):
                result[key] = None
        if not np.any(labels == 0):
            for key in ('specificity','false_positive_rate','gmean'):
                result[key] = None
        return result

    out = {'overall':metrics(y,scores),
           'quota':{str(q):quota_metrics(y,scores,q) for q in (.001,.002)},
           'channels':{},'months':{}}
    channels = np.asarray(store.channels[ids])
    for code,name in enumerate(store.meta['channel_names']):
        mask = channels == code
        if mask.any():
            out['channels'][name] = metrics(y[mask],scores[mask])
    month = np.asarray(store.timestamps[ids]).astype('datetime64[ns]').astype('datetime64[M]')
    for key in np.unique(month):
        mask = month==key
        out['months'][str(key)] = metrics(y[mask],scores[mask])
    return out
