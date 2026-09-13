"""Local fold training; assessment labels never choose epoch or threshold."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import psutil
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score

from .evaluation import report, thresholds
from .legacy import EXP12_DIR, GraphSAGE
from .sampling.roots import RootSampler
from .sampling.temporal import TemporalNeighborSampler, build_csr, forward_temporal
from .sampling.weights import FrozenWeightContext


VARIANTS = {
    'R0': {'mode':'static','roots':'balanced','graph':True},
    'A0': {'mode':'static','roots':'balanced','graph':True},
    'A1': {'mode':'static','roots':'balanced','graph':False},
    'B0': {'mode':'frozen','roots':'balanced','graph':True},
    'B_control': {'mode':'frozen','roots':'balanced','graph':True},
    'B1': {'mode':'moving','roots':'balanced','graph':True},
    'B2': {'mode':'recent','roots':'balanced','graph':True},
    'B3': {'mode':'frozen','roots':'channel_quarter','graph':True},
    'B4': {'mode':'recent','roots':'channel_quarter','graph':True},
}


def source_manifest():
    sources = {str(p.relative_to(Path(__file__).resolve().parents[1])):p for p in
               Path(__file__).resolve().parents[1].rglob('*.py')}
    for filename in ('behavior.py','behavior_v2.py','data.py','model.py','metrics.py'):
        sources['EXP12/'+filename] = EXP12_DIR/'exp12'/filename
    return {k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in sorted(sources.items())}


def _write(path,payload):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def _positive_weights(store,train_end,power=.5,cap=4.):
    y = np.asarray(store.labels[:train_end])
    groups,count = np.unique(np.asarray(store.channels[:train_end])[y==1],return_counts=True)
    costs = np.minimum((count.max()/count.astype(np.float64))**power,cap)
    costs /= np.dot(costs,count)/count.sum()
    return {int(group):float(weight) for group,weight in zip(groups,costs)}


def _predict(model,store,sampler,ids,fit_end,*,seed,batch_size,device):
    model.eval()
    ids = np.asarray(ids,dtype=np.int32)
    out = np.empty(len(ids),dtype=np.float64)
    with torch.no_grad():
        for start in range(0,len(ids),batch_size):
            roots = ids[start:start+batch_size]
            batch_seed = seed if sampler.mode == 'static' else seed+start
            neighbors,degree = sampler.sample(roots,fit_end,batch_seed,training=False)
            cached = sampler.static_means_for(roots) if sampler.mode == 'static' else None
            logits = forward_temporal(model,store,roots,neighbors,degree,device,cached)
            out[start:start+len(roots)] = logits.sigmoid().cpu().numpy()
    return out


def _save_predictions(path,store,ids,scores):
    ids = np.asarray(ids,dtype=np.int32)
    np.savez_compressed(path,tx_id=ids,source_row=np.asarray(store.source_rows[ids]),
                        timestamp_ns=np.asarray(store.timestamps[ids]),
                        channel=np.asarray(store.channels[ids]),
                        label=np.asarray(store.labels[ids]),score=np.asarray(scores,dtype=np.float32))


def run_fold(store,roles,variant,seed,output_root,*,epochs=100,min_epochs=20,patience=20,
             batch_size=512,eval_batch_size=2048,learning_rate=.001,fanout=25,
             label_delay_days=0,min_fraud=25,device='auto',assess=True,
             frozen_state=None,threshold_policy='fpr_0_001',strategy='uniform',
             weight_config=None,negatives_per_positive=30):
    """Roles contain arrays of transaction IDs and train_end; no test-role peeking."""
    if variant not in VARIANTS: raise ValueError(f'Varian tidak dikenal: {variant}')
    if strategy not in {'uniform','topology','importance'}:
        raise ValueError('Strategi sampling tidak dikenal')
    if variant != 'R0' and strategy != 'uniform':
        raise ValueError('Perbandingan strategi utama hanya pada R0 dengan graf beku yang sama')
    if not 0 < batch_size or epochs < 1 or patience < 1 or negatives_per_positive < 1:
        raise ValueError('Training budget tidak valid')
    mode = VARIANTS[variant]
    dev = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else
                       'cpu' if device == 'auto' else device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if dev.type == 'cuda': torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(torch.get_num_threads(),4))
    out = Path(output_root)/(f'{variant}_{strategy}_{roles["name"]}_seed{seed}'
                             if variant == 'R0' else f'{variant}_{roles["name"]}_seed{seed}')
    out.mkdir(parents=True,exist_ok=True)
    status_path = out/'status.json'
    cfg = {'variant':variant,'strategy':strategy,'weight_config':weight_config,
           'seed':seed,'epochs':epochs,'min_epochs':min_epochs,
           'patience':patience,'batch_size':batch_size,'eval_batch_size':eval_batch_size,
           'learning_rate':learning_rate,'fanout':fanout,
           'negatives_per_positive':negatives_per_positive,
           'label_delay_days':label_delay_days,
           'min_fraud':min_fraud,'device':str(dev),'threshold_policy':threshold_policy,
           'normalization':'batch' if variant == 'R0' else 'layer',
           'loss_policy':('inverse_frequency_corrected_for_negative_subsampling'
                          if variant == 'R0' else 'channel_weighted_bce'),
           'roles':{k:({'count':len(v),'sha256':hashlib.sha256(np.asarray(v,dtype=np.int32).tobytes()).hexdigest()}
                       if isinstance(v,np.ndarray) else v) for k,v in roles.items()},
           'source_manifest':source_manifest(),'data_identity':store.meta['identity']}
    if status_path.exists() and json.loads(status_path.read_text()).get('status') == 'complete':
        previous = json.loads((out/'config.json').read_text(encoding='utf-8'))
        if previous != cfg:
            raise RuntimeError(f'Run selesai memakai konfigurasi/source berbeda: {out}')
        return json.loads((out/'metrics.json').read_text(encoding='utf-8'))
    _write(out/'config.json',cfg)
    _write(status_path,{'status':'running','variant':variant,'fold':roles['name'],'seed':seed})
    try:
        started = time.perf_counter()
        process = psutil.Process()
        observed_rss = process.memory_info().rss
        if dev.type == 'cuda': torch.cuda.reset_peak_memory_stats(dev)
        train_end = int(roles['train_end'])
        if train_end <= 0 or train_end > store.n:
            raise ValueError('Cutoff training tidak valid')
        for role in ('selection','calibration','assessment'):
            ids = np.asarray(roles[role])
            if len(ids) and (np.any(ids < train_end) or np.any(ids >= store.n)):
                raise ValueError(f'{role} melewati cutoff/rentang')
        select = np.asarray(roles['selection'],dtype=np.int32)
        calibration = np.asarray(roles['calibration'],dtype=np.int32)
        assessment = np.asarray(roles['assessment'],dtype=np.int32)
        if len(np.intersect1d(select,calibration)) or len(np.intersect1d(select,assessment)) or len(np.intersect1d(calibration,assessment)):
            raise ValueError('Selection/calibration/assessment overlap')
        for role,ids in [('selection',select),('calibration',calibration)]:
            if int(np.asarray(store.labels[ids]).sum()) < min_fraud or not (np.asarray(store.labels[ids])==0).any():
                raise ValueError(f'{role} kurang dukungan fraud/normal')
        rowptr,col = build_csr(store)
        sampler = TemporalNeighborSampler(store,rowptr,col,fanout=fanout,mode=mode['mode'],
                                          strategy=strategy,weight_config=weight_config)
        eval_sampler = TemporalNeighborSampler(store,rowptr,col,fanout=fanout,mode=mode['mode'],
                                               strategy=strategy,weight_config=weight_config)
        weight_build_seconds = 0.
        if strategy != 'uniform':
            weight_started = time.perf_counter()
            weight_context = FrozenWeightContext(store,train_end,weight_config)
            weight_build_seconds = time.perf_counter()-weight_started
            for active_sampler in (sampler,eval_sampler):
                active_sampler.weight_context = weight_context
                active_sampler.weight_context_cutoff = train_end
        eval_seed = 10000+seed*100
        model = GraphSAGE(store.width+4,hidden=256,dropout=.2,
                          normalization='batch' if variant == 'R0' else 'layer',
                          use_graph_context=mode['graph']).to(dev)
        weights = _positive_weights(store,train_end) if variant != 'R0' else None
        root_sampler = RootSampler(store,train_end,ratio=negatives_per_positive,
                                   mode=mode['roots'],seed=seed)
        population_pos_weight = len(root_sampler.negative)/len(root_sampler.positive)
        # Inverse-frequency BCE on the full population, corrected for the
        # n30 negative subsample, has this same positive:negative ratio.
        sampled_pos_weight = root_sampler.negative_count/len(root_sampler.positive)
        history = []
        best,best_epoch,stale = -1.,0,0
        best_state = None
        if frozen_state is None:
            optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=.0001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=5,min_lr=1e-5)
            for epoch in range(1,epochs+1):
                model.train()
                epoch_started = time.perf_counter()
                roots = root_sampler.roots()
                losses = []
                for start in range(0,len(roots),batch_size):
                    if start == len(roots)-1:
                        break  # The previous batch absorbed this singleton.
                    stop = min(start+batch_size,len(roots))
                    if len(roots)-stop == 1:
                        stop = len(roots)
                    batch = roots[start:stop]
                    batch_seed = (seed*100000+epoch if sampler.mode == 'static' else
                                  seed*100000+epoch*1000+start)
                    neighbors,degree = sampler.sample(batch,train_end,batch_seed,training=True)
                    cached = sampler.static_means_for(batch) if sampler.mode == 'static' else None
                    y_np = np.asarray(store.labels[batch],dtype=np.float32)
                    ch = np.asarray(store.channels[batch])
                    cost = (np.where(y_np > 0,sampled_pos_weight,1.).astype(np.float32)
                            if variant == 'R0' else
                            np.asarray([weights.get(int(c),1.) if y else 1. for c,y in zip(ch,y_np)],dtype=np.float32))
                    y = torch.from_numpy(y_np).to(dev)
                    w = torch.from_numpy(cost).to(dev)
                    optimizer.zero_grad(set_to_none=True)
                    logits = forward_temporal(model,store,batch,neighbors,degree,dev,cached)
                    loss = F.binary_cross_entropy_with_logits(logits,y,weight=w)
                    if not torch.isfinite(loss): raise FloatingPointError('Loss non-finite')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                    optimizer.step()
                    losses.append(float(loss.detach().cpu())*len(batch))
                    if start % (batch_size*100) == 0:
                        observed_rss = max(observed_rss,process.memory_info().rss)
                scores = _predict(model,store,eval_sampler,select,train_end,seed=eval_seed,
                                  batch_size=eval_batch_size,device=dev)
                ap = float(average_precision_score(np.asarray(store.labels[select]),scores))
                if ap > best:
                    best,best_epoch,stale = ap,epoch,0
                    best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    torch.save({'state_dict':best_state,'input_channels':store.width+4,
                                'best_epoch':epoch,'selection_ap':ap,'config':cfg},out/'best.pt')
                else: stale += 1
                scheduler.step(ap)
                history.append({'epoch':epoch,'loss':sum(losses)/len(roots),
                                'selection_ap':ap,'best_epoch':best_epoch,'stale':stale,
                                'seconds':time.perf_counter()-epoch_started,
                                'observed_rss_gib':observed_rss/2**30})
                _write(out/'history.json',history)
                if epoch >= min_epochs and stale >= patience: break
            model.load_state_dict(best_state)
        else:
            model.load_state_dict(frozen_state['state_dict'])
            best_epoch = frozen_state['best_epoch']
            best = frozen_state['selection_ap']
        selection_scores = _predict(model,store,eval_sampler,select,train_end,
                                    seed=eval_seed,batch_size=eval_batch_size,device=dev)
        current_selection_ap = float(average_precision_score(np.asarray(store.labels[select]),selection_scores))
        calibration_scores = _predict(model,store,eval_sampler,calibration,train_end,
                                      seed=eval_seed,batch_size=eval_batch_size,device=dev)
        options,calibration_f2 = thresholds(np.asarray(store.labels[calibration]),calibration_scores)
        if threshold_policy not in options: raise ValueError('Threshold policy tidak dikenal')
        decision = options[threshold_policy]
        _save_predictions(out/'selection_scores.npz',store,select,selection_scores)
        _save_predictions(out/'calibration_scores.npz',store,calibration,calibration_scores)
        result = {'variant':variant,'strategy':strategy,'seed':seed,'fold':roles['name'],
                  'best_epoch':best_epoch,
                  'selection_ap':best,'selection_ap_current':current_selection_ap,
                  'thresholds':options,'decision_threshold':decision,
                  'threshold_policy':threshold_policy,'calibration_f2':calibration_f2,
                  'calibration':report(store,calibration,calibration_scores,decision),
                  'train':{'rows':train_end,'fraud':int(np.asarray(store.labels[:train_end]).sum()),
                           'roots_per_epoch':len(root_sampler.positive)+root_sampler.negative_count,
                           'positive_channel_weights':weights,
                           'population_inverse_frequency_pos_weight':population_pos_weight,
                           'sampled_corrected_pos_weight':(sampled_pos_weight if variant == 'R0' else None)},
                  'elapsed_seconds':time.perf_counter()-started,
                  'resources':{'observed_peak_rss_gib':observed_rss/2**30,
                               'peak_vram_gib':(torch.cuda.max_memory_allocated(dev)/2**30
                                                if dev.type=='cuda' else None)},
                  'sampling_preparation':{
                      'weight_context_seconds':weight_build_seconds,
                      'training_table_refresh_seconds':sampler.static_refresh_seconds,
                      'evaluation_table_refresh_seconds':eval_sampler.static_refresh_seconds},
                  'neighbor_diagnostics':eval_sampler.diagnostics(
                      assessment if len(assessment) else calibration,train_end,eval_seed),
                  'assessment_exploratory':bool(roles.get('assessment_exploratory',False))}
        if assess and len(assessment):
            inference_started = time.perf_counter()
            assessment_scores = _predict(model,store,eval_sampler,assessment,train_end,
                                         seed=eval_seed,batch_size=eval_batch_size,device=dev)
            inference_seconds = time.perf_counter()-inference_started
            result['assessment_inference'] = {
                'seconds_including_sampling_and_features':inference_seconds,
                'transactions_per_second':len(assessment)/inference_seconds,
                'milliseconds_per_transaction':1000*inference_seconds/len(assessment)}
            _save_predictions(out/'assessment_scores.npz',store,assessment,assessment_scores)
            result['assessment'] = report(store,assessment,assessment_scores,decision)
            result['assessment_other_thresholds'] = {
                name:report(store,assessment,assessment_scores,value)['overall']
                for name,value in options.items() if name != threshold_policy}
        result['elapsed_seconds'] = time.perf_counter()-started
        _write(out/'metrics.json',result)
        _write(status_path,{'status':'complete','variant':variant,'fold':roles['name'],
                            'seed':seed,'best_epoch':best_epoch,'epochs':len(history)})
        return result
    except Exception as exc:
        _write(status_path,{'status':'failed','error':str(exc),'type':type(exc).__name__})
        raise
