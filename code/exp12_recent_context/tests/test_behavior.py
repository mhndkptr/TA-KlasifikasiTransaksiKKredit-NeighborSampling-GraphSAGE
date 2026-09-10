import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch

from exp12.behavior import FEATURE_NAMES, fill_behavior_features
from exp12.config import load_config, validate_config
from exp12.data import load_transactions
from exp12.graph import build_graph
from exp12.protocol import positive_channel_weights, selection_bounds, stopping_decision
from exp12.runs import RunDirectory
from exp12.trainer import run_one
from tests.test_core import ROOT, transactions


def frame(count=180):
    data = transactions(count)
    data['Card'] = np.arange(count) % 3
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'data.csv'
        data.to_csv(path, index=False)
        return load_transactions(path)


def transform(data):
    out = np.empty((len(data), len(FEATURE_NAMES)), dtype=np.float32)
    fill_behavior_features(data, out)
    return out


class BehavioralTests(unittest.TestCase):
    def test_matches_independent_slow_history_and_excludes_ties(self):
        df = frame(40)
        df.loc[7, '_datetime'] = df.loc[0, '_datetime']
        df = df.sort_values('_datetime', kind='stable').reset_index(drop=True)
        actual = transform(df)
        for i, row in df.iterrows():
            previous = df[(df.User == row.User) & (df._datetime < row._datetime)]
            self.assertAlmostEqual(actual[i, 0], np.log1p(len(previous))/10, places=6)
            self.assertEqual(actual[i, 1], len(previous) == 0)
            for j, seconds in enumerate([3600, 86400, 604800], 3):
                count = (previous._datetime >= row._datetime-pd.Timedelta(seconds=seconds)).sum()
                self.assertAlmostEqual(actual[i, j], np.log1p(count)/5, places=6)
            if len(previous):
                values = np.sign(previous._amount)*np.log1p(abs(previous._amount))
                delta = np.sign(row._amount)*np.log1p(abs(row._amount))-values.mean()
                self.assertAlmostEqual(actual[i, 6], np.clip(delta, -10, 10)/5, places=6)
            card_previous = previous[previous.Card == row.Card]
            self.assertAlmostEqual(actual[i, 8], np.log1p(len(card_previous))/10, places=6)
            seen_merchant = (previous['Merchant Name'] == row['Merchant Name']).any()
            self.assertEqual(actual[i, FEATURE_NAMES.index('user_merchant_new')], not seen_merchant)

    def test_future_values_and_all_labels_cannot_change_earlier_features(self):
        df = frame()
        expected = transform(df)
        changed = df.copy()
        changed.loc[100:, '_amount'] = 999999
        changed.loc[100:, 'User'] = 999
        changed.loc[100:, 'Merchant Name'] = 888
        changed['_label'] = 1-changed['_label']
        np.testing.assert_array_equal(transform(changed)[:100], expected[:100])
        df['_label'] = 1-df['_label']
        np.testing.assert_array_equal(transform(df), expected)
        np.testing.assert_array_equal(transform(df[:100]), expected[:100])

    def test_missing_amount_card_and_cold_start_remain_finite(self):
        df = frame().drop(columns=['Card'])
        df.loc[:10, '_amount'] = np.nan
        values = transform(df)
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_array_equal(values[:, :8], values[:, 8:16])
        self.assertEqual(values[0, 6], 0)

    def test_graph_preprocessing_train_features_do_not_read_heldout_values(self):
        df = frame()
        cfg = {'encoder': 'behavioral', 'storage_dtype': 'float32'}
        first = build_graph(df, [.7, .15, .15], cfg)
        df.loc[first.train_end:, '_amount'] = -999999
        df.loc[first.train_end:, '_label'] = 1-df.loc[first.train_end:, '_label']
        second = build_graph(df, [.7, .15, .15], cfg)
        torch.testing.assert_close(first.features[:first.train_end], second.features[:second.train_end], rtol=0, atol=0)
        self.assertTrue(first.metadata['behavior']['card_available'])

    def test_channel_weights_are_train_only_bounded_and_positive_mean_one(self):
        y = torch.tensor([1]*12+[0]*5)
        c = torch.tensor([0]*10+[1]*2+[0]*5)
        weights, report = positive_channel_weights(y, c, .5, 4)
        self.assertAlmostEqual(float(weights[y == 1].mean()), 1, places=6)
        self.assertTrue((weights[y == 0] == 1).all())
        self.assertGreater(weights[10], weights[0])
        self.assertLessEqual(float(weights.max()), 4)
        self.assertEqual(report['1']['train_fraud'], 2)

    def test_selection_and_calibration_are_disjoint_and_stop_has_warmup(self):
        cfg = validate_config(load_config(ROOT/'config.exp11_control.yaml'))
        start, end = selection_bounds(1000, cfg['training'])
        self.assertEqual((start, end), (500, 750))
        sparse = np.zeros(1000, dtype=np.int8)
        sparse[100:150] = 1
        expanded, boundary = selection_bounds(1000, cfg['training'], sparse)
        self.assertEqual((expanded, boundary), (125, 750))
        self.assertFalse(stopping_decision(19, 20, cfg['training']))
        self.assertTrue(stopping_decision(20, 20, cfg['training']))
        cfg['evaluation']['calibration_tail_fraction'] = .5
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_config(cfg)

    def test_calibration_labels_do_not_select_model_and_early_stop_is_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = load_config(ROOT/'config.exp11_control.yaml')
            cfg['experiment'].update(device='cpu', seeds=[42], cache_graph=False)
            cfg['runtime']['progress_bar'] = False
            cfg['model']['hidden_channels'] = 8
            cfg['training'].update(epochs=3, min_epochs=2, early_stopping_patience=1,
                early_stopping_min_delta=2., batch_size=32)
            cfg['evaluation'].update(batch_size=64, latency_nodes=5, latency_repeats=1)
            cfg['paths'].update(model_dir=str(Path(directory)/'model'), result_dir=str(Path(directory)/'result'))
            graph = build_graph(frame(360), [.7, .15, .15], cfg['features'])
            graph.metadata['data_fingerprint'] = 'fixture'
            first = run_one(cfg, graph, 'uniform', 42, torch.device('cpu'))
            model1 = torch.load(first['checkpoint'], weights_only=True)
            _, end = selection_bounds(graph.val_end-graph.train_end, cfg['training'])
            graph.labels[graph.train_end+end:graph.val_end] = 1-graph.labels[graph.train_end+end:graph.val_end]
            cfg['experiment']['on_existing'] = 'new'
            second = run_one(cfg, graph, 'uniform', 42, torch.device('cpu'))
            model2 = torch.load(second['checkpoint'], weights_only=True)
            for key in model1['model_state']:
                torch.testing.assert_close(model1['model_state'][key], model2['model_state'][key], rtol=0, atol=0)
            self.assertEqual(first['metrics']['stop_reason'], 'early_stopping')
            status = json.loads((Path(directory)/'result'/Path(first['checkpoint']).parent.name/'status.json').read_text())
            self.assertEqual(status['status'], 'complete')
            self.assertEqual(status['stop_reason'], 'early_stopping')

    def test_keyboard_interrupt_is_a_different_status(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = load_config(ROOT/'config.exp11_control.yaml')
            cfg['paths'].update(model_dir=str(Path(directory)/'model'), result_dir=str(Path(directory)/'result'))
            with self.assertRaises(KeyboardInterrupt):
                with RunDirectory(cfg, 'interrupt_fixture', 'uniform', 42) as attempt:
                    path = attempt.output
                    raise KeyboardInterrupt('fixture')
            status = json.loads((path/'status.json').read_text())
            self.assertEqual(status['status'], 'interrupted')
            self.assertFalse((path/'.run.lock').exists())
