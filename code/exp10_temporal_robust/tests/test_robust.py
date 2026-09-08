import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import torch

from exp10.artifacts import load_or_prepare, write_summaries
from exp10.config import load_config, validate_config
from exp10.data import FeatureEncoder, load_transactions
from exp10.features import FeatureStore, neighbor_feature_means
from exp10.factorized import forward_factorized
from exp10.evaluation import build_contexts, predict
from exp10.metrics import binary_metrics, validation_score
from exp10.minibatch import RootSampler
from exp10.model import GraphSAGE
from exp10.sampling import NeighborTableSampler
from exp10.trainer import run_one
from tests.test_core import ROOT, fixture_graph, transactions


class RobustTests(unittest.TestCase):
    def test_missing_error_distinct_from_actual_error_and_oov(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'data.csv'
            transactions().to_csv(path, index=False)
            df = load_transactions(path)
        encoder = FeatureEncoder().fit(df[:100])
        values, _ = encoder.transform(df[:3])
        missing = encoder.feature_names.index('Errors?=<missing>')
        error = encoder.feature_names.index('Errors?=Error')
        self.assertEqual(values[0, error], 1)
        self.assertEqual(values[1, missing], 1)
        self.assertEqual(values[1, error], 0)
        df.loc[100:, 'MCC'] = 999999
        actual, diagnostics = FeatureEncoder(json.loads(json.dumps(encoder.state))).transform(df[100:])
        self.assertEqual(diagnostics['category_oov_rate']['MCC'], 1)
        self.assertTrue((actual[:, encoder.feature_names.index('MCC=<oov>')] == 1).all())

    def test_balanced_roots_unique_all_positives_reproducible_and_rotate(self):
        labels = torch.tensor([1]*8 + [0]*200)
        a = RootSampler(labels, 'balanced', 4, 42)
        b = RootSampler(labels, 'balanced', 4, 42)
        first = torch.cat(list(a.batches(13, 'cpu')))
        torch.testing.assert_close(first, torch.cat(list(b.batches(13, 'cpu'))))
        self.assertEqual(len(first), 40)
        self.assertEqual(len(first.unique()), 40)
        self.assertEqual(int(labels[first].sum()), 8)
        self.assertTrue(set(range(8)) <= set(first.tolist()))
        self.assertNotEqual(set(first.tolist()), set(torch.cat(list(a.batches(13, 'cpu'))).tolist()))

    def test_balancing_guard_and_all_presets(self):
        for path in ROOT.glob('config*.yaml'):
            validate_config(load_config(path))
        cfg = load_config(ROOT/'config.yaml')
        cfg['training']['positive_class_weight_power'] = 1
        with self.assertRaisesRegex(ValueError, 'double balancing'):
            validate_config(cfg)

    def test_metrics_match_independent_sklearn_and_all_normal_baseline(self):
        y = np.array([0, 0, 0, 1, 1])
        p = np.array([.1, .2, .8, .4, .9])
        m = binary_metrics(y, p, .5)
        self.assertAlmostEqual(m['f1_macro'], f1_score(y, p >= .5, average='macro'))
        self.assertAlmostEqual(m['gmean'], np.sqrt(.5*2/3))
        self.assertAlmostEqual(m['roc_auc'], roc_auc_score(y, p))
        self.assertAlmostEqual(m['ap_lift'], average_precision_score(y, p)/y.mean())
        normal = binary_metrics(y, np.zeros(len(y)))
        self.assertEqual(normal['gmean'], 0)
        self.assertGreater(normal['f1_macro'], normal['f1'])

    def test_temporal_selection_penalizes_one_easy_window(self):
        y = np.tile([0, 0, 1, 1], 3)
        strong = np.tile([.1, .2, .8, .9], 3)
        fragile = np.concatenate([strong[:8], 1-strong[8:]])
        good, _ = validation_score(y, strong, 'temporal_geometric_ap_lift', 3)
        bad, records = validation_score(y, fragile, 'temporal_geometric_ap_lift', 3)
        self.assertGreater(good, bad)
        self.assertEqual(len(records), 3)

    def test_layernorm_factorized_cached_and_self_only_context_invariance(self):
        graph = fixture_graph()
        store = FeatureStore(graph)
        table = NeighborTableSampler(graph, None, 4, 30, 'cpu').sample(42)
        means = neighbor_feature_means(graph, store, table, 8, 'cpu')
        roots = graph.nodes('test')[:8]
        for use_graph in [True, False]:
            model = GraphSAGE(graph.input_channels, 16, 0, 'layer', use_graph).eval()
            with torch.no_grad():
                expected = forward_factorized(model, store, roots, means).sigmoid().numpy()
            contexts, _ = build_contexts(model, graph, store, [table], 8, torch.device('cpu'), [means])
            actual, _ = predict(model, graph, store, roots, contexts, 3, torch.device('cpu'))
            np.testing.assert_allclose(actual, expected, atol=1e-7)
            if not use_graph:
                with torch.no_grad():
                    changed = forward_factorized(model, store, roots, means+999).sigmoid().numpy()
                np.testing.assert_array_equal(changed, expected)

    def test_primary_threshold_summary_checkpoint_and_heldout_label_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'data.csv'
            transactions(360).to_csv(path, index=False)
            cfg = load_config(ROOT/'config.yaml')
            cfg['data']['transactions'] = str(path)
            cfg['experiment'].update(device='cpu', seeds=[42])
            cfg['runtime']['progress_bar'] = False
            cfg['model']['hidden_channels'] = 16
            cfg['training'].update(epochs=2, batch_size=32)
            cfg['evaluation'].update(batch_size=64, latency_nodes=10, latency_repeats=1)
            cfg['paths'].update(model_dir=str(Path(directory)/'model'), result_dir=str(Path(directory)/'result'))
            graph, cache = load_or_prepare(cfg)
            first = run_one(cfg, graph, 'uniform', 42, torch.device('cpu'))
            checkpoint = torch.load(first['checkpoint'], weights_only=True)
            self.assertEqual(first['metrics']['decision_threshold'], first['calibrated_threshold'])
            self.assertEqual(checkpoint['decision_threshold'], first['calibrated_threshold'])
            self.assertEqual(first['metrics']['f1'], first['test_metrics_at_validation_threshold']['f1'])
            self.assertEqual(sum(v['count'] for v in first['payment_channel_metrics']['test'].values()),
                             graph.num_transactions-graph.val_end)
            write_summaries(cfg['paths']['result_dir'])
            summary = pd.read_csv(Path(directory)/'result'/'summary.csv')
            self.assertIn('roc_auc_mean', summary)
            self.assertIn('f1_macro_mean', summary)
            self.assertAlmostEqual(summary.f1_mean.iloc[0], first['metrics']['f1'])
            # Perturb only TEST targets. Selected checkpoint and threshold must
            # stay identical; replay uses the same val labels and RNG seed.
            graph.labels[graph.val_end:] = 1-graph.labels[graph.val_end:]
            cfg['experiment']['on_existing'] = 'new'
            second = run_one(cfg, graph, 'uniform', 42, torch.device('cpu'))
            replay = torch.load(second['checkpoint'], weights_only=True)
            self.assertEqual(first['calibrated_threshold'], second['calibrated_threshold'])
            self.assertEqual(first['metrics']['best_epoch'], second['metrics']['best_epoch'])
            for name in checkpoint['model_state']:
                torch.testing.assert_close(checkpoint['model_state'][name], replay['model_state'][name], rtol=0, atol=0)
            cfg['features']['storage_dtype'] = 'float32'
            _, changed = load_or_prepare(cfg)
            self.assertNotEqual(cache, changed)


if __name__ == '__main__':
    unittest.main()
