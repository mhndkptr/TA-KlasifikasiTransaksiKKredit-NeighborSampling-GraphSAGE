"""Read-only EXP10 artifact audit; aggregate outputs, no transaction rows."""
import argparse
import json
from pathlib import Path


def audit(root):
    runs = []
    for path in sorted(Path(root).glob('*/metrics.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        if data['config']['experiment']['max_rows'] is not None:
            continue
        metrics, history = data['metrics'], data['history']
        state = json.loads(path.with_name('status.json').read_text(encoding='utf-8'))
        patience = data['config']['training']['early_stopping_patience']
        early = state['status'] == 'complete' and history[-1]['stale_epochs'] >= patience
        runs.append({'source': str(path), 'comparison_id': data['comparison_id'],
            'status': state['status'], 'inferred_stop_reason': 'early_stopping' if early else 'inspect_history',
            'last_stale_epochs': history[-1]['stale_epochs'], 'patience': patience,
            'metrics': metrics, 'split_stats': data['split_stats'],
            'checkpoint_blocks': history[metrics['best_epoch']-1]['validation_blocks'],
            'threshold_validation': data['threshold_validation'],
            'payment_channel_metrics': data['payment_channel_metrics'],
            'probability_diagnostics': data['probability_diagnostics'],
            'drift': data['drift'], 'training': data['training']})
    if not runs:
        raise ValueError('Tidak ditemukan hasil full-data EXP10')
    return {'runs': runs, 'scope': 'provided full-data artifacts only; no inference about unseen console errors'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parents[2]/'result'/'exp10')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'validation'/'exp10_audit.json')
    args = parser.parse_args()
    result = audit(args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    for run in result['runs']:
        m = run['metrics']
        print(f"{m['strategy']}: {run['status']}/{run['inferred_stop_reason']}; "
              f"best/end={m['best_epoch']}/{m['epochs']}; AP={m['auprc']:.6f}; F1={m['f1']:.6f}; recall={m['recall']:.6f}")
