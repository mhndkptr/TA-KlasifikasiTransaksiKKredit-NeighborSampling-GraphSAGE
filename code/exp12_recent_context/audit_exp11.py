"""Reproducible full-data artifact comparison and optional streamed CSV audit."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp12.config import load_config
from exp12.protocol import selection_bounds, validation_partitions

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]


def completed_full_runs(folder):
    latest = {}
    for path in sorted(folder.glob('*/metrics.json'), key=lambda p: p.stat().st_mtime_ns):
        data = json.loads(path.read_text(encoding='utf-8'))
        status = json.loads(path.with_name('status.json').read_text(encoding='utf-8'))
        if status['status'] != 'complete' or data['config']['experiment']['max_rows'] is not None:
            continue
        m = data['metrics']
        latest[data['comparison_id'], m['strategy'], m['seed']] = (path, data)
    return list(latest.values())


def csv_validation_audit(path, reference):
    stats = reference['split_stats']
    lo, hi = pd.Timestamp(stats['val']['start']), pd.Timestamp(stats['val']['end'])
    columns = ['Year', 'Month', 'Day', 'Time', 'Use Chip', 'Is Fraud?']
    times, labels, channels = [], [], []
    before = 0
    names = ['Chip Transaction', 'Online Transaction', 'Swipe Transaction']
    for chunk in pd.read_csv(path, usecols=columns, chunksize=250000):
        dates = pd.to_datetime(dict(year=chunk.Year, month=chunk.Month, day=chunk.Day))
        parts = chunk.Time.str.split(':', expand=True).astype(int)
        seconds = parts[0]*3600 + parts[1]*60 + (parts[2] if parts.shape[1] == 3 else 0)
        dates += pd.to_timedelta(seconds, unit='s')
        before += int((dates < lo).sum())
        keep = (dates >= lo) & (dates <= hi)
        times.append(dates[keep].to_numpy(dtype='datetime64[ns]').view(np.int64))
        labels.append(chunk.loc[keep, 'Is Fraud?'].eq('Yes').to_numpy(np.int8))
        channels.append(chunk.loc[keep, 'Use Chip'].map({v: i for i, v in enumerate(names)}).fillna(-1).to_numpy(np.int8))
    t, y, c = np.concatenate(times), np.concatenate(labels), np.concatenate(channels)
    # Stable sorting retains source-row order for equal timestamps. Trim exact
    # rank boundaries, including the train/val and val/test boundary ties.
    order = np.argsort(t, kind='stable')
    start = stats['train']['total']-before
    order = order[start:start+stats['val']['total']]
    t, y, c = t[order], y[order], c[order]
    if len(t) != stats['val']['total'] or int(y.sum()) != stats['val']['fraud']:
        raise ValueError('CSV validation tidak cocok dengan artefak referensi')
    a, b = selection_bounds(len(t), reference['config']['training'], y)
    cfg = load_config(ROOT/'config.yaml')
    selected, calibrated, report = validation_partitions(t, y, cfg['training'], cfg['evaluation'])

    def describe(ids):
        return {'count': len(ids), 'fraud': int(y[ids].sum()),
                'start': str(pd.Timestamp(t[ids[0]])), 'end': str(pd.Timestamp(t[ids[-1]])),
                'channels': {name: {'count': int((c[ids] == i).sum()),
                                    'fraud': int(y[ids][c[ids] == i].sum())}
                             for i, name in enumerate(names)}}
    return {'source': str(path), 'exact_validation_match': True,
            'exp11_selection': describe(np.arange(a, b)),
            'exp12_selection': describe(selected), 'exp12_calibration': describe(calibrated),
            'partition_policy': report}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, help='Optional full CSV: stream only timestamps/labels/channel')
    parser.add_argument('--output', type=Path, default=ROOT/'validation'/'exp11_audit.json')
    args = parser.parse_args()
    report = {'scope': 'completed full-data runs only; latest attempt per comparison/strategy/seed', 'runs': []}
    sources = {}
    for version in ['exp10', 'exp11']:
        sources[version] = completed_full_runs(PROJECT/'result'/version)
        for path, data in sources[version]:
            m = data['metrics']
            report['runs'].append({'version': version, 'source': str(path.relative_to(PROJECT)),
                'comparison_id': data['comparison_id'], 'metrics': m,
                'selection_window': data['training'].get('selection_window'),
                'positive_channel_weights': data['training'].get('positive_channel_weights'),
                'test_channels': data['payment_channel_metrics']['test'],
                'validation_temporal_bins': data['validation_temporal_bins'],
                'threshold_validation': data['threshold_validation']})
            print(f"{version} {m['strategy']}: AP={m['auprc']:.6f} F1={m['f1']:.6f} recall={m['recall']:.6f}", flush=True)
    if not sources['exp11']:
        raise ValueError('Hasil full-data exp11 tidak ditemukan')
    if args.data:
        report['csv_validation_audit'] = csv_validation_audit(args.data, sources['exp11'][0][1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Audit saved: {args.output}', flush=True)


if __name__ == '__main__':
    main()
