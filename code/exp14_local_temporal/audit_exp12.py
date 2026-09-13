"""Read-only, concise local audit of completed n30 EXP12 artifacts."""
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]/'result'/'exp12'
    report = []
    for path in sorted(root.glob('*/metrics.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        cfg = data['config']
        if cfg['experiment']['max_rows'] is not None or cfg['training']['negatives_per_positive'] != 30:
            continue
        status = json.loads(path.with_name('status.json').read_text(encoding='utf-8'))
        if status['status'] != 'complete': continue
        m = data['metrics']
        chip = data['payment_channel_metrics']['test']['Chip Transaction']
        report.append({'run':path.parent.name,'source_manifest':data['source_manifest'],
                       'data_fingerprint':data['data_fingerprint'],
                       'ap':m['auprc'],'precision':m['precision'],'recall':m['recall'],
                       'f1':m['f1'],'tp':m['tp'],'fp':m['fp'],'fn':m['fn'],
                       'chip_ap':chip['auprc'],'chip_recall':chip['recall'],
                       'chip_fraud':chip['fraud'],'selection_ap':m['selection_score'],
                       'best_epoch':m['best_epoch']})
    compact = [{k:v for k,v in item.items() if k not in {'source_manifest'}} for item in report]
    print(json.dumps(compact,indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
