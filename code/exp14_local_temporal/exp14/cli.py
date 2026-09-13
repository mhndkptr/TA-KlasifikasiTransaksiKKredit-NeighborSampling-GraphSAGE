"""One local command for preprocessing, support audits, and bounded runs."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from .feature_store import prepare
from .protocol import DAY_NS, calendar_folds
from .trainer import VARIANTS, run_fold
from .summary import summarize
from .audit import temporal_support
from .runtime import LOGGER, logging_session


def _roles_static(store):
    val = np.arange(store.train_end,store.test_start,dtype=np.int32)
    ts = np.asarray(store.timestamps[val])
    recent = len(val)*3//4
    recent = int(np.searchsorted(ts,ts[recent],side='left'))
    block = (ts[recent:]//(7*DAY_NS)) % 2
    return {'name':'static_70_15_15','train_end':store.train_end,
            'selection':val[recent:][block==0],
            'calibration':val[recent:][block==1],
            'assessment':np.arange(store.test_start,store.n,dtype=np.int32),
            'assessment_exploratory':True}


def _roles_fold(fold):
    ranges = fold.ranges()
    return {'name':fold.name,'train_end':fold.train_end,
            'selection':np.arange(*ranges['selection'],dtype=np.int32),
            'calibration':np.arange(*ranges['calibration'],dtype=np.int32),
            'assessment':np.arange(*ranges['assessment'],dtype=np.int32),
            'assessment_exploratory':False}


def main(argv=None):
    parser = argparse.ArgumentParser(description='EXP14 GraphSAGE lokal; tanpa Kaggle')
    parser.add_argument('command',choices=['preprocess','audit','run','summarize'])
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'config.local.yaml')
    parser.add_argument('--variant',choices=sorted(VARIANTS))
    parser.add_argument('--strategy',choices=['uniform','topology','importance'],
                        help='Strategi neighbor sampling untuk perbandingan R0')
    parser.add_argument('--fold',help='Pilih satu origin yang diaudit')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--max-rows',type=int,help='Smoke prefix CSV; bukan evaluasi populasi')
    parser.add_argument('--results-dir',type=Path,help='Direktori hasil terpisah untuk smoke/ablation')
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--min-fraud',type=int,help='Smoke-only override guard')
    parser.add_argument('--device',choices=['auto','cpu','cuda'])
    parser.add_argument('--no-assessment',action='store_true')
    parser.add_argument('--no-progress',action='store_true',
                        help='Matikan progress bar; log tahap dan epoch tetap ditulis')
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    base = args.config.resolve().parent
    result_root = (args.results_dir.resolve() if args.results_dir else
                   (base/cfg['paths']['results']).resolve())
    with logging_session(result_root/'run.log'):
        LOGGER.info('EXP14 command=%s; log=%s', args.command, result_root/'run.log')
        try:
            return _execute(args, cfg)
        except KeyboardInterrupt:
            LOGGER.warning('Interrupted by user')
            raise
        except Exception:
            LOGGER.exception('EXP14 failed')
            raise


def _execute(args, cfg):
    base = args.config.resolve().parent
    data_path = (base/cfg['data']['transactions']).resolve()
    cache_root = (base/cfg['paths']['cache']).resolve()
    result_root = (args.results_dir.resolve() if args.results_dir else
                   (base/cfg['paths']['results']).resolve())
    enabled = cfg['runtime'].get('progress_bar',True) and not args.no_progress
    if args.command == 'summarize':
        output = summarize(result_root)
        LOGGER.info('Summary: %s; runs=%d',result_root/'summary.json',len(output['runs']))
        return 0
    store = prepare(data_path,cache_root,max_rows=args.max_rows,
                    memory_limit=cfg['runtime']['duckdb_memory_limit'],
                    threads=cfg['runtime']['cpu_threads'],
                    chunk_rows=cfg['runtime']['chunk_rows'],progress_bar=enabled)
    LOGGER.info('Feature store: %s, rows=%s, features=%d, train=%d, test=%d',
                store.path,f'{store.n:,}',store.width,store.train_end,store.test_start)
    if args.command == 'preprocess': return 0
    chip_name = 'Chip Transaction'
    chip_code = (store.meta['channel_names'].index(chip_name)
                 if chip_name in store.meta['channel_names'] else None)
    pcfg = cfg['protocol']
    assessment_end = pcfg.get('assessment_end_date')
    assessment_end_ns = (int(datetime.fromisoformat(assessment_end)
                             .replace(tzinfo=timezone.utc).timestamp()*1e9)
                         if assessment_end and args.max_rows is None else None)
    min_fraud = args.min_fraud if args.min_fraud is not None else pcfg['minimum_fraud']
    folds,audit = calendar_folds(store.timestamps,store.train_end,store.test_start,
        origins=pcfg['origins'],window_days=pcfg['window_days'],
        delay_days=pcfg['label_delay_days'],min_fraud=min_fraud,
        min_chip_fraud=pcfg['minimum_chip_fraud'],labels=store.labels,
        channels=store.channels,chip_code=chip_code,
        assessment_end_ns=assessment_end_ns)
    result_root.mkdir(parents=True,exist_ok=True)
    manifest = {'data_identity':store.meta['identity'],'train_end':store.train_end,
                'test_start':store.test_start,'protocol':pcfg,'origins':audit,
                'static':{'selection':len(_roles_static(store)['selection']),
                          'calibration':len(_roles_static(store)['calibration'])}}
    (result_root/'fold_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    if args.command == 'audit':
        support = temporal_support(store,folds=folds)
        (result_root/'temporal_support.json').write_text(json.dumps(support,indent=2),encoding='utf-8')
        print(f'Temporal support: {result_root/"temporal_support.json"}; '
              f'fraud={support["fraud_transactions"]}, episodes={support["fraud_episodes"]}')
        print(json.dumps(manifest,indent=2))
        return 0
    variants = [args.variant] if args.variant else cfg['experiment']['variants']
    seeds = [args.seed] if args.seed is not None else cfg['experiment']['seeds']
    train_cfg = cfg['training']
    for variant in variants:
        if variant in {'A0','A1','R0'}:
            selected = [_roles_static(store)]
            earliest = None
        else:
            eligible_names = {entry['bounds']['name'] for entry in audit
                              if entry['eligible'] and 'bounds' in entry}
            selected = [_roles_fold(fold) for fold in folds if fold.name in eligible_names]
            earliest = selected[0] if selected else None
            if args.fold:
                selected = [role for role in selected if role['name'] == args.fold]
        if not selected:
            raise RuntimeError(f'Tidak ada fold layak untuk {variant}; lihat fold_manifest.json')
        strategies = ([args.strategy] if args.strategy else
                      cfg['experiment'].get('strategies',['uniform'])) if variant == 'R0' else ['uniform']
        if args.strategy and variant != 'R0':
            raise ValueError('--strategy hanya untuk R0')
        for strategy in strategies:
            for seed in seeds:
                for roles in selected:
                    state = None
                    if variant == 'B_control':
                        checkpoint = result_root/f'B0_{earliest["name"]}_seed{seed}'/'best.pt'
                        if not checkpoint.exists():
                            raise RuntimeError('Jalankan B0 origin pertama sebelum B_control')
                        state = torch.load(checkpoint,map_location='cpu',weights_only=True)
                        roles = {**roles,'train_end':earliest['train_end']}
                    LOGGER.info('Run %s/%s %s seed %d',variant,strategy,roles['name'],seed)
                    result = run_fold(store,roles,variant,seed,result_root,
                        epochs=args.epochs or train_cfg['epochs'],
                        min_epochs=train_cfg['min_epochs'],patience=train_cfg['patience'],
                        batch_size=train_cfg['batch_size'],
                        eval_batch_size=train_cfg['eval_batch_size'],
                        learning_rate=train_cfg['learning_rate'],fanout=train_cfg['fanout'],
                        label_delay_days=pcfg['label_delay_days'],min_fraud=min_fraud,
                        device=args.device or cfg['runtime']['device'],
                        assess=not args.no_assessment,
                        frozen_state=state,
                        threshold_policy=cfg['evaluation']['primary_threshold'],
                        strategy=strategy,weight_config=cfg.get('sampling'),
                        negatives_per_positive=train_cfg.get('negatives_per_positive',30),
                        progress_bar=enabled,
                        progress_update_batches=cfg['runtime'].get('progress_update_batches',100))
                    LOGGER.info('Complete: AP selection=%.6f; output=%s',result['selection_ap'],result_root)
                    summarize(result_root)
                    LOGGER.info('Summary updated: %s', result_root/'summary.json')
    return 0
