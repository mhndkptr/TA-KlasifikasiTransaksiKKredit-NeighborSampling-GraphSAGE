import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from exp12.behavior_v2 import FEATURE_NAMES, fill_context_features
from exp12.config import load_config, validate_config
from exp12.graph import build_graph
from exp12.artifacts import replace_with_retry
from exp12.protocol import validation_partitions
from exp12.trainer import run_one
from tests.test_behavior import frame
from tests.test_core import ROOT


def transform(df):
    output = np.empty((len(df), len(FEATURE_NAMES)), dtype=np.float32)
    fill_context_features(df, output)
    return output


class ContextTests(unittest.TestCase):
    def test_recent_and_conditional_amount_match_independent_history(self):
        df = frame(100)
        df['_datetime'] = pd.date_range('2020-01-01', periods=len(df), freq='12h')
        df.loc[7, '_datetime'] = df.loc[0, '_datetime']
        df.loc[12, '_amount'] = np.nan
        df = df.sort_values('_datetime', kind='stable').reset_index(drop=True)
        actual = transform(df)
        for i, row in df.iterrows():
            before = df[df._datetime < row._datetime]
            recent = before[(before.User == row.User) & (before._datetime >= row._datetime-pd.Timedelta(days=7))]
            conditioned = before[(before.User == row.User) & (before.Card == row.Card)
                                 & (before['Use Chip'] == row['Use Chip'])]
            for prefix, history in [('user_amount_7d', recent), ('card_channel_amount', conditioned)]:
                values = history._amount.dropna().to_numpy()
                values = np.sign(values)*np.log1p(np.abs(values))
                delta = np.sign(row._amount)*np.log1p(abs(row._amount))-values.mean() if len(values) and pd.notna(row._amount) else 0.
                z = delta/np.sqrt(max(values.var(), .25)) if len(values) else 0.
                self.assertAlmostEqual(actual[i, FEATURE_NAMES.index(prefix+'_delta')], np.clip(delta, -10, 10)/5, places=5)
                self.assertAlmostEqual(actual[i, FEATURE_NAMES.index(prefix+'_z')], np.clip(z, -10, 10)/5, places=5)

    def test_future_labels_and_one_missing_card_do_not_change_past(self):
        df = frame(180)
        expected = transform(df)
        changed = df.copy()
        changed['Card'] = changed['Card'].astype(float)
        changed.loc[120:, 'Card'] = np.nan
        changed.loc[120:, '_amount'] = np.inf
        changed.loc[120:, 'Merchant Name'] = 888
        changed['_label'] = 1-changed['_label']
        np.testing.assert_array_equal(transform(changed)[:120], expected[:120])
        np.testing.assert_array_equal(transform(df[:120]), expected[:120])
        df['_label'] = 1-df['_label']
        np.testing.assert_array_equal(transform(df), expected)
        self.assertTrue(np.isfinite(transform(changed)).all())

    def test_equal_time_peer_order_does_not_change_any_context(self):
        df = frame(100)
        df['_datetime'] = df['_datetime'].dt.floor('12h')
        expected = transform(df)
        shuffled = df.sample(frac=1, random_state=9).sort_values('_datetime', kind='stable')
        actual = transform(shuffled)
        np.testing.assert_allclose(actual[np.argsort(shuffled.index)], expected, atol=1e-6)

    def test_partial_missing_card_preserves_known_card_histories(self):
        df = frame(80)
        df['User'] = 1
        df['Card'] = np.arange(len(df), dtype=float) % 2
        df.loc[3, 'Card'] = np.nan
        actual = transform(df)
        for i, row in df.iterrows():
            same_card = df.Card.isna() if pd.isna(row.Card) else df.Card.eq(row.Card)
            history = df[(df._datetime < row._datetime) & same_card]
            self.assertAlmostEqual(actual[i, FEATURE_NAMES.index('card_prior_count_log')], np.log1p(len(history))/10, places=6)
            self.assertEqual(actual[i, -1], pd.isna(row.Card))

    def test_recent_partition_uses_latest_fraud_without_label_assignment(self):
        cfg = load_config(ROOT/'config.yaml')
        times = pd.date_range('2017-01-01', periods=8000, freq='h').asi8
        labels = (np.arange(len(times)) % 11 == 0).astype(np.int8)
        a, b, report = validation_partitions(times, labels, cfg['training'], cfg['evaluation'])
        self.assertEqual(report['overlap_count'], 0)
        self.assertGreaterEqual(a.min(), 6000)
        self.assertGreaterEqual(b.min(), 6000)
        self.assertGreater(report['selection']['fraud'], 25)
        aa, bb, _ = validation_partitions(times, 1-labels, cfg['training'], cfg['evaluation'])
        np.testing.assert_array_equal(a, aa)
        np.testing.assert_array_equal(b, bb)
        labels[6000:] = 0
        with self.assertRaisesRegex(ValueError, 'tidak memperluas'):
            validation_partitions(times, labels, cfg['training'], cfg['evaluation'])

    def test_config_presets_are_valid(self):
        for path in ROOT.glob('config*.yaml'):
            with self.subTest(path=path.name):
                validate_config(load_config(path))

    def test_cli_exposes_subset_partition_guards(self):
        source = (ROOT/'exp12'/'cli.py').read_text(encoding='utf-8')
        self.assertIn('--selection-min-fraud', source)
        self.assertIn('--calibration-min-fraud', source)

    def test_atomic_replace_retries_transient_permission_error(self):
        class FlakyPath:
            def __init__(self):
                self.calls = 0
            def replace(self, target):
                self.calls += 1
                if self.calls < 3:
                    raise PermissionError('scanner fixture')
        source = FlakyPath()
        replace_with_retry(source, 'target', attempts=3)
        self.assertEqual(source.calls, 3)

    def test_calibration_and_test_labels_cannot_select_checkpoint(self):
        cfg = load_config(ROOT/'config.yaml')
        cfg['experiment'].update(device='cpu', seeds=[42], cache_graph=False)
        cfg['runtime']['progress_bar'] = False
        cfg['model']['hidden_channels'] = 8
        cfg['training'].update(epochs=3, min_epochs=2, early_stopping_patience=1,
                              early_stopping_min_delta=2., batch_size=64, selection_min_fraud=1)
        cfg['evaluation'].update(batch_size=128, latency_nodes=5, latency_repeats=1, calibration_min_fraud=1)
        df = frame(720)
        df['_datetime'] = pd.date_range('2020-01-01', periods=len(df), freq='D')
        with tempfile.TemporaryDirectory() as directory:
            cfg['paths'].update(model_dir=str(Path(directory)/'model'), result_dir=str(Path(directory)/'result'))
            graph = build_graph(df, [.7, .15, .15], cfg['features'])
            graph.metadata['data_fingerprint'] = 'context-fixture'
            a, b, report = validation_partitions(graph.timestamps[graph.train_end:graph.val_end].numpy(),
                graph.labels[graph.train_end:graph.val_end].numpy(), cfg['training'], cfg['evaluation'])
            first = run_one(cfg, graph, 'uniform', 42, torch.device('cpu'))
            model1 = torch.load(first['checkpoint'], weights_only=True)
            graph.labels[graph.train_end+b] = 1-graph.labels[graph.train_end+b]
            graph.labels[graph.val_end:] = 1-graph.labels[graph.val_end:]
            cfg['experiment']['on_existing'] = 'new'
            second = run_one(cfg, graph, 'uniform', 42, torch.device('cpu'))
            model2 = torch.load(second['checkpoint'], weights_only=True)
            for key in model1['model_state']:
                torch.testing.assert_close(model1['model_state'][key], model2['model_state'][key], rtol=0, atol=0)
            self.assertEqual(first['metrics']['selection_score'], second['metrics']['selection_score'])
            self.assertEqual(first['metrics']['best_epoch'], second['metrics']['best_epoch'])
            self.assertEqual(first['training']['validation_partitions']['overlap_count'], 0)
