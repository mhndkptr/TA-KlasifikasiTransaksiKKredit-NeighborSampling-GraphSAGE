"""Paired fold/seed summaries; static and rolling protocols remain separate."""
import json
import hashlib
from pathlib import Path
import numpy as np


def _research_family(config):
    """Hash controls that must stay fixed across R0 strategies and seeds."""
    comparable = {key:value for key,value in config.items()
                  if key not in {'strategy','seed'}}
    return hashlib.sha256(json.dumps(comparable,sort_keys=True).encode()).hexdigest()


def _exp12_reference():
    root = Path(__file__).resolve().parents[3]/'result'/'exp12'
    names = ('exp12_recent_context_c39b0e23c708fe40_uniform_seed42',
             'exp12_channel_equal_n30_e0fe1853a94ef333_uniform_seed42',
             'exp12_chip_weight_n30_dd39a2ff0c5542ec_uniform_seed42')
    found = {}
    for name in names:
        path = root/name/'metrics.json'
        if path.exists():
            values = json.loads(path.read_text(encoding='utf-8'))['metrics']
            found[name] = {'ap':values['auprc'],'recall':values['recall'],
                           'f1':values['f1'],'fp':values['fp']}
    return {'runs':found,'best_ap':max((v['ap'] for v in found.values()),default=None),
            'comparison_limit':'EXP12 memakai source/protokol/threshold berbeda; AP historis hanya konteks eksploratif'}


def _research_gate(rows, strategy, reference_ap, required_seeds=(42,43,44,45,46)):
    """Do not declare improvement from a prefix, one seed, or mismatched controls."""
    if strategy == 'uniform':
        raise ValueError('Gate memerlukan kandidat topology/importance')
    pairs=[]
    for seed in required_seeds:
        base = next((r for r in rows if r['variant']=='R0' and r['strategy']=='uniform'
                     and r['seed']==seed and r['full_data']),None)
        candidate = next((r for r in rows if r['variant']=='R0' and r['strategy']==strategy
                          and r['seed']==seed and r['full_data']),None)
        if base is None or candidate is None:
            continue
        if (base['family'] != candidate['family'] or base['fold'] != candidate['fold']):
            return {'status':'incomparable_controls','paired_seeds':len(pairs)}
        pairs.append((base,candidate))
    if len(pairs) != len(required_seeds):
        return {'status':'pending_full_data_replicates','paired_seeds':len(pairs),
                'required_seeds':list(required_seeds)}
    if len({base['family'] for base,_ in pairs}) != 1:
        return {'status':'incomparable_controls','paired_seeds':len(pairs)}
    delta_ap = [candidate['ap']-base['ap'] for base,candidate in pairs]
    delta_recall = [candidate['recall']-base['recall'] for base,candidate in pairs]
    delta_f1 = [candidate['f1']-base['f1'] for base,candidate in pairs]
    mean_candidate_ap = float(np.mean([candidate['ap'] for _,candidate in pairs]))
    criteria = {'mean_delta_ap_positive':float(np.mean(delta_ap))>0,
                'ap_wins_at_least_4_of_5':sum(value>0 for value in delta_ap)>=4,
                'mean_delta_recall_positive':float(np.mean(delta_recall))>0,
                'mean_delta_f1_positive':float(np.mean(delta_f1))>0,
                'historical_exp12_best_ap_exceeded':(
                    mean_candidate_ap>reference_ap if reference_ap is not None else None)}
    passed = all(value is True for value in criteria.values())
    return {'status':'meets_exploratory_gate' if passed else 'does_not_meet_gate',
            'paired_seeds':len(pairs),'criteria':criteria,
            'mean_delta_ap':float(np.mean(delta_ap)),
            'mean_delta_recall':float(np.mean(delta_recall)),
            'mean_delta_f1':float(np.mean(delta_f1)),
            'mean_candidate_ap':mean_candidate_ap,
            'historical_exp12_best_ap':reference_ap}


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
        config = json.loads(path.with_name('config.json').read_text(encoding='utf-8'))
        identity = config.get('data_identity',{})
        full_data = (identity.get('max_rows') is None and
                     identity.get('size',0) >= 2_000_000_000)
        assessment = m['assessment']
        chip = assessment['channels'].get('Chip Transaction',{})
        rows.append({'variant':m['variant'],'strategy':m.get('strategy','uniform'),
                     'fold':m['fold'],'seed':m['seed'],
                     'family':_research_family(config) if m['variant']=='R0' else None,
                     'full_data':full_data,
                     'protocol':'static' if m['variant'].startswith(('A','R')) else 'rolling',
                     'exploratory':m['assessment_exploratory'],
                     'ap':assessment['overall']['auprc'],
                     'chip_ap':chip.get('auprc'),'chip_recall':chip.get('recall'),
                     'precision':assessment['overall']['precision'],
                     'recall':assessment['overall']['recall'],
                     'f1':assessment['overall']['f1'],
                     'fp':assessment['overall']['fp'],
                     'fpr':assessment['overall']['false_positive_rate'],
                     'inference_seconds':m.get('assessment_inference',{}).get(
                         'seconds_including_sampling_and_features')})
    by_key = {(r['variant'],r['strategy'],r['fold'],r['seed']):r for r in rows}
    reference = _exp12_reference()
    output = {'runs':rows,'groups':{},'paired_vs_B0':{},'paired_vs_R0_uniform':{},
              'exp12_reference':reference,
              'research_gate':{strategy:_research_gate(rows,strategy,reference['best_ap'])
                               for strategy in ('topology','importance')}}
    for variant,strategy in sorted({(r['variant'],r['strategy']) for r in rows}):
        group = [r for r in rows if r['variant']==variant and r['strategy']==strategy]
        name = f'R0_{strategy}' if variant == 'R0' else variant
        output['groups'][name] = {'protocol':group[0]['protocol'],'n':len(group),
                                  'strategy':strategy}
        for metric in ('ap','chip_ap','chip_recall','precision','recall','f1','fp','fpr',
                       'inference_seconds'):
            values = np.asarray([r[metric] for r in group if r[metric] is not None],dtype=float)
            output['groups'][name][metric] = {
                'count':len(values),
                'mean':float(values.mean()) if len(values) else None,
                'std':float(values.std(ddof=1)) if len(values)>1 else None}
        if variant.startswith('B') and variant!='B0':
            paired=[]
            for row in group:
                baseline=by_key.get(('B0','uniform',row['fold'],row['seed']))
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
            output['groups'][name]['promotion_audit'] = {
                'enough_replicates':len(seeds)>=3 and len(folds)>=3,
                'mean_delta_ap':float(np.mean(ap)) if ap else None,
                'mean_delta_chip_recall':float(np.mean(chip)) if chip else None,
                'ap_positive_majority':sum(x>0 for x in ap)>len(ap)/2 if ap else False,
                'chip_support_pairs':len(chip),
                'target_ap_0_02_met':float(np.mean(ap))>=.02 if ap else False,
                'target_chip_recall_0_05_met':float(np.mean(chip))>=.05 if chip else False}
        if variant == 'R0' and strategy != 'uniform':
            pairs=[]
            for row in group:
                baseline=by_key.get(('R0','uniform',row['fold'],row['seed']))
                if baseline is None:
                    continue
                pairs.append({'fold':row['fold'],'seed':row['seed'],
                              'delta_ap':row['ap']-baseline['ap'],
                              'delta_recall':row['recall']-baseline['recall'],
                              'delta_f1':row['f1']-baseline['f1'],
                              'delta_inference_seconds':(
                                  row['inference_seconds']-baseline['inference_seconds']
                                  if row['inference_seconds'] is not None and
                                  baseline['inference_seconds'] is not None else None)})
            output['paired_vs_R0_uniform'][strategy]=pairs
    root.mkdir(parents=True,exist_ok=True)
    (root/'summary.json').write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8')
    return output
