"""Read-only audit of EXP9 outputs and class-conditional temporal CSV drift.

Full CSV is streamed; raw transaction/customer records are never exported.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd


def audit_results(directory):
    runs = []
    for path in sorted(Path(directory).glob('*/metrics.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        m = data['metrics']
        tn, fp, fn, tp = (m[k] for k in ('tn', 'fp', 'fn', 'tp'))
        f1_normal = 2 * tn / max(2 * tn + fp + fn, 1)
        prevalence = (tp + fn) / (tn + fp + fn + tp)
        runs.append({'source': path.as_posix(), 'strategy': m['strategy'], 'seed': m['seed'],
            'roc_auc': m['roc_auc'], 'ap': m['auprc'], 'ap_lift': m['auprc'] / prevalence,
            'val_ap': m['best_val_auprc'], 'f1_fraud_0_5': m['f1'],
            'f1_macro_0_5': (m['f1'] + f1_normal) / 2,
            'gmean_0_5': (m['recall'] * tn / (tn + fp)) ** 0.5,
            'recall_0_5': m['recall'], 'precision_0_5': m['precision'],
            'tp': tp, 'fp': fp, 'val_threshold': data['calibrated_threshold'],
            'validation_threshold_test': data['test_metrics_at_validation_threshold'],
            'positive_class_weight': data['training']['positive_class_weight'],
            'best_epoch': m['best_epoch'], 'epochs': m['epochs']})
    return runs


def audit_csv(path, chunk_size=500_000):
    counts, categories = Counter(), Counter()
    columns = ['Year', 'Month', 'Day', 'Time', 'Use Chip', 'Errors?', 'MCC', 'Merchant City', 'Is Fraud?']
    n = 0
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunk_size):
        n += len(chunk)
        # Match EXP9 cut dates. Exact ties are marked separately: no arbitrary
        # reassignment of minute-equal boundary rows is hidden in this audit.
        dt = pd.to_datetime(dict(year=chunk.Year, month=chunk.Month, day=chunk.Day))
        dt += pd.to_timedelta(chunk.Time.astype(str) + ':00')
        split = np.select([dt < pd.Timestamp('2015-12-10 12:08'),
                           dt < pd.Timestamp('2018-01-27 08:07')], ['train', 'val'], default='test')
        label = chunk['Is Fraud?'].str.strip().str.lower()
        counts.update(chunk.groupby(['Year', label], dropna=False).size().to_dict())
        for name in ['Use Chip', 'Errors?', 'MCC', 'Merchant City']:
            value = chunk[name].fillna('<missing>').astype(str)
            if name == 'Merchant City':
                value = value.eq('ONLINE').map({True: 'ONLINE', False: 'not_online'})
            groups = pd.DataFrame({'split': split, 'label': label, 'value': value})
            for (s, y, v), count in groups.groupby(['split', 'label', 'value']).size().items():
                categories[(name, s, y, v)] += int(count)
        print(f'Audited {n:,} rows', flush=True)
    return {'rows': n, 'boundary_tie_policy': 'strict timestamp < cutoff, differs only at boundary ties',
        'annual_labels': [{'year': int(k[0]), 'label': k[1], 'count': int(v)} for k, v in sorted(counts.items())],
        'class_conditional_categories': [{'feature': k[0], 'split': k[1], 'label': k[2], 'value': k[3],
                                         'count': v} for k, v in sorted(categories.items())]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='result/exp9')
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'exp9_runs': audit_results(args.results)}
    if args.data:
        report['data_audit'] = audit_csv(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
