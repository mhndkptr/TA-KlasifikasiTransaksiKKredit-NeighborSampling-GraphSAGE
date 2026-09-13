"""Paired fold/seed summaries; static and rolling protocols remain separate."""
import json
from pathlib import Path
import numpy as np


def summarize(result_root):
    root = Path(result_root)
    rows = []
    for path in sorted(root.glob('*/metrics.json')):
        status = path.with_name('status.json')
        if not status.exists() or json.loads(status.read_text())['status'] != 'complete':
            continue
        m = json.loads(path.read_text())
        if 'assessment' not in m:
            continue
        assessment = m['assessment']
        chip = assessment['channels'].get('Chip Transaction',{})
        rows.append({'variant':m['variant'],'fold':m['fold'],'seed':m['seed'],
                     'protocol':'static' if m['variant'].startswith('A') else 'rolling',
                     'exploratory':m['assessment_exploratory'],
                     'ap':assessment['overall']['auprc'],
                     'chip_ap':chip.get('auprc'),'chip_recall':chip.get('recall'),
                     'precision':assessment['overall']['precision'],
                     'recall':assessment['overall']['recall'],
                     'fp':assessment['overall']['fp'],
                     'fpr':assessment['overall']['false_positive_rate']})
    by_key = {(r['variant'],r['fold'],r['seed']):r for r in rows}
    output = {'runs':rows,'groups':{},'paired_vs_B0':{}}
    for variant in sorted({r['variant'] for r in rows}):
        group = [r for r in rows if r['variant']==variant]
        output['groups'][variant] = {'protocol':group[0]['protocol'],'n':len(group)}
        for metric in ('ap','chip_ap','chip_recall','precision','recall','fp','fpr'):
            values = np.asarray([r[metric] for r in group if r[metric] is not None],dtype=float)
            output['groups'][variant][metric] = {
                'count':len(values),
                'mean':float(values.mean()) if len(values) else None,
                'std':float(values.std(ddof=1)) if len(values)>1 else None}
        if variant.startswith('B') and variant!='B0':
            paired=[]
            for row in group:
                baseline=by_key.get(('B0',row['fold'],row['seed']))
                if baseline is not None and row['ap'] is not None and baseline['ap'] is not None:
                    paired.append({'fold':row['fold'],'seed':row['seed'],
                                   'delta_ap':row['ap']-baseline['ap'],
                                   'delta_chip_recall':(row['chip_recall']-baseline['chip_recall']
                                                        if row['chip_recall'] is not None and baseline['chip_recall'] is not None else None),
                                   'delta_fp':row['fp']-baseline['fp']})
            output['paired_vs_B0'][variant]=paired
            seeds = {item['seed'] for item in paired}
            folds = {item['fold'] for item in paired}
            ap = [item['delta_ap'] for item in paired]
            chip = [item['delta_chip_recall'] for item in paired if item['delta_chip_recall'] is not None]
            output['groups'][variant]['promotion_audit'] = {
                'enough_replicates':len(seeds)>=3 and len(folds)>=3,
                'mean_delta_ap':float(np.mean(ap)) if ap else None,
                'mean_delta_chip_recall':float(np.mean(chip)) if chip else None,
                'ap_positive_majority':sum(x>0 for x in ap)>len(ap)/2 if ap else False,
                'chip_support_pairs':len(chip),
                'target_ap_0_02_met':float(np.mean(ap))>=.02 if ap else False,
                'target_chip_recall_0_05_met':float(np.mean(chip))>=.05 if chip else False}
    root.mkdir(parents=True,exist_ok=True)
    (root/'summary.json').write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8')
    return output
