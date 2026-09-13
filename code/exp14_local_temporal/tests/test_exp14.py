import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from exp14.feature_store import prepare
from exp14.legacy import CONTEXT_NAMES, fill_context_features
from exp12.data import load_transactions
from exp14.protocol import DAY_NS, calendar_folds, tie_safe_boundary
from exp14.sampling.temporal import build_csr, TemporalNeighborSampler
from exp14.sampling.roots import RootSampler
from exp14.evaluation import fpr_threshold, quota_metrics
from exp14.trainer import run_fold
from exp14.audit import temporal_support
from exp14.sampling.temporal import forward_temporal
from exp12.graph import TransactionGraph, FeatureStore as BlockStore
from exp12.minibatch import sample_blocks
from exp12.model import GraphSAGE


def synthetic_csv(path, n=180):
    columns = ['User','Card','Year','Month','Day','Time','Amount','Use Chip',
               'Merchant Name','Merchant City','Merchant State','Zip','MCC','Errors?','Is Fraud?']
    origin = np.datetime64('2020-01-01')
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for i in range(n):
            day = str(origin+np.timedelta64(i//2,'D'))
            yy,mm,dd = day.split('-')
            writer.writerow([i%5,i%2,yy,int(mm),int(dd),'08:00' if i%2==0 else '16:00',
                             f'${(i%27)+1}.50',
                             'Chip Transaction' if i%3 else 'Online Transaction',
                             str(i%13),'ONLINE' if i%3==0 else 'City','CA','91750',
                             '5300' if i%4 else '5411','','Yes' if i%11==0 else 'No'])


class Exp14Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.temporary.name)
        cls.csv = cls.dir/'transactions.csv'
        synthetic_csv(cls.csv)
        cls.store = prepare(cls.csv,cls.dir/'cache',memory_limit='512MB',threads=2,chunk_rows=2048)

    @classmethod
    def tearDownClass(cls):
        cls.store.close()
        del cls.store
        cls.temporary.cleanup()

    def test_context_matches_exp12_formula(self):
        df = load_transactions(self.csv)
        output = np.empty((len(df),len(CONTEXT_NAMES)),dtype=np.float32)
        fill_context_features(df,output)
        actual = self.store.get(np.arange(len(df)))[:,-len(CONTEXT_NAMES):]
        np.testing.assert_allclose(actual,output,atol=.002,rtol=.002)
        np.testing.assert_array_equal(self.store.source_rows,df._source_row.to_numpy())
        self.assertEqual(self.store.meta['feature_names'].count('merchant_channel_support_log'),1)

    def test_tie_safe_split_and_calendar(self):
        ts = np.asarray(self.store.timestamps)
        self.assertTrue(np.all(ts[:self.store.train_end] < ts[self.store.train_end]))
        self.assertTrue(np.all(ts[:self.store.test_start] < ts[self.store.test_start]))
        self.assertEqual(tie_safe_boundary(ts,0),1)
        folds,audit = calendar_folds(ts,self.store.train_end,self.store.test_start,
                                      origins=1,window_days=1,labels=self.store.labels,
                                      min_fraud=1)
        self.assertTrue(isinstance(folds,list) and isinstance(audit,list))

    def test_label_delay_creates_real_gap(self):
        days = np.arange(500,dtype=np.int64)*DAY_NS
        folds,audit = calendar_folds(days,100,400,origins=1,window_days=60,
                                      delay_days=7)
        self.assertEqual(len(folds),1)
        self.assertEqual(folds[0].selection_start-folds[0].train_end,7)
        self.assertEqual(folds[0].ranges()['train'][1],folds[0].train_end)
        self.assertEqual(folds[0].ranges()['selection'][0],folds[0].selection_start)

    def test_zero_delay_keeps_valid_calendar_folds(self):
        days = np.arange(800,dtype=np.int64)*DAY_NS
        folds,audit = calendar_folds(days,300,790,origins=3,window_days=90)
        self.assertEqual(len(folds),3)
        self.assertTrue(all(item['eligible'] for item in audit))
        self.assertTrue(all(fold.train_end == fold.selection_start for fold in folds))

    def test_no_fraud_subgroup_reports_undefined_recall(self):
        from exp14.evaluation import quota_metrics, report
        ids = np.flatnonzero(np.asarray(self.store.labels) == 0)[:20]
        scores = np.linspace(.1,.9,len(ids))
        metrics = report(self.store,ids,scores,.5)
        self.assertIsNone(metrics['overall']['auprc'])
        self.assertIsNone(metrics['overall']['recall'])
        self.assertIsNone(metrics['quota']['0.001']['recall'])

    def test_preadited_assessment_end_stays_before_test(self):
        days = np.arange(800,dtype=np.int64)*DAY_NS
        folds,_ = calendar_folds(days,300,790,origins=1,window_days=90,
                                 assessment_end_ns=700*DAY_NS)
        self.assertEqual(folds[0].assessment_end,700)
        self.assertEqual(folds[0].selection_start,430)

    def test_heldout_label_changes_do_not_change_context(self):
        alternate = self.dir/'alternate.csv'
        with self.csv.open(newline='',encoding='utf-8') as stream:
            rows = list(csv.reader(stream))
        for row in rows[151:]:
            row[-1] = 'No' if row[-1] == 'Yes' else 'Yes'
        with alternate.open('w',newline='',encoding='utf-8') as stream:
            csv.writer(stream).writerows(rows)
        changed = prepare(alternate,self.dir/'alternate_cache',memory_limit='512MB',
                          threads=2,chunk_rows=2048)
        try:
            np.testing.assert_array_equal(self.store.get(np.arange(self.store.n)),
                                          changed.get(np.arange(changed.n)))
            self.assertFalse(np.array_equal(self.store.labels,changed.labels))
        finally:
            changed.close()

    def test_temporal_neighbors_exclude_same_day_and_future(self):
        rowptr,col = build_csr(self.store)
        root = 75
        sampler = TemporalNeighborSampler(self.store,rowptr,col,fanout=4,mode='moving')
        neighbors,degree = sampler.sample([root],self.store.train_end,seed=12)
        valid = neighbors[neighbors>=0]
        day = int(self.store.timestamps[root])//DAY_NS*DAY_NS
        self.assertTrue(np.all(np.asarray(self.store.timestamps[valid]) < day))
        self.assertTrue(np.all(valid < root))
        # Held-out labels cannot influence the sampled transaction IDs.
        again,_ = sampler.sample([root],self.store.train_end,seed=12)
        np.testing.assert_array_equal(neighbors,again)
        self.assertTrue((degree>=0).all())

    def test_root_mixture_has_unique_negatives(self):
        sampler = RootSampler(self.store,self.store.train_end,ratio=3,
                              mode='channel_quarter',seed=5)
        roots = sampler.roots()
        self.assertEqual(len(np.unique(roots)),len(roots))
        self.assertEqual(int(np.sum(self.store.labels[roots])),len(sampler.positive))

    def test_factorized_query_matches_sage_blocks(self):
        store = self.store
        rowptr,col = build_csr(store)
        cutoff = 110
        roots = np.array([145,147],dtype=np.int32)
        sampler = TemporalNeighborSampler(store,rowptr,col,fanout=3,mode='static')
        neighbors,degree = sampler.sample(roots,cutoff,seed=17)
        partial_degree = np.zeros(store.num_users+store.num_merchants,dtype=np.int64)
        for entity in range(len(partial_degree)):
            lo,hi = sampler.eligible(entity,cutoff)
            partial_degree[entity] = hi-lo
        entities = np.column_stack((store.users,store.merchants+store.num_users)).astype(np.int64)
        graph = TransactionGraph(torch.from_numpy(store.get(np.arange(store.n))),
            torch.from_numpy(np.asarray(store.labels).copy()),torch.from_numpy(entities),
            torch.from_numpy(np.asarray(rowptr).copy()),torch.from_numpy(np.asarray(col).copy()),
            torch.from_numpy(partial_degree),torch.zeros(len(partial_degree),dtype=torch.long),
            torch.zeros(store.n,dtype=torch.long),torch.from_numpy(np.asarray(store.source_rows).copy()),
            torch.from_numpy(np.asarray(store.timestamps).copy()),cutoff,store.test_start,
            store.num_users,{'feature_names':store.meta['feature_names']})
        table = torch.full((graph.num_entities,3),-1,dtype=torch.long)
        for i,root in enumerate(roots):
            for side,entity in enumerate(entities[root]):
                table[entity] = torch.from_numpy(neighbors[i,side].astype(np.int64))
        model = GraphSAGE(store.width+4,hidden=16,dropout=0.,normalization='layer').eval()
        with torch.no_grad():
            direct = forward_temporal(model,store,roots,neighbors,degree,torch.device('cpu'))
            blocks = sample_blocks(graph,torch.from_numpy(roots.astype(np.int64)),table,[3,3])
            expanded = model(BlockStore(graph).get(blocks[0].source,torch.device('cpu')),blocks)
        np.testing.assert_allclose(direct.numpy(),expanded.numpy(),atol=1e-5,rtol=1e-5)
        model.zero_grad(set_to_none=True)
        forward_temporal(model,store,roots,neighbors,degree,torch.device('cpu')).sum().backward()
        direct_grad = {k:v.grad.detach().clone() for k,v in model.named_parameters() if v.grad is not None}
        model.zero_grad(set_to_none=True)
        model(BlockStore(graph).get(blocks[0].source,torch.device('cpu')),blocks).sum().backward()
        for name,value in model.named_parameters():
            if name in direct_grad:
                np.testing.assert_allclose(direct_grad[name].numpy(),value.grad.numpy(),atol=1e-5,rtol=1e-5)

    def test_static_entity_cache_matches_direct_features(self):
        store=self.store
        rowptr,col=build_csr(store)
        sampler=TemporalNeighborSampler(store,rowptr,col,fanout=3,mode='static')
        roots=np.array([145,147],dtype=np.int32)
        neighbors,degree=sampler.sample(roots,110,seed=91)
        model=GraphSAGE(store.width+4,hidden=16,dropout=0.,normalization='layer').eval()
        with torch.no_grad():
            direct=forward_temporal(model,store,roots,neighbors,degree,torch.device('cpu'))
            cached=forward_temporal(model,store,roots,neighbors,degree,torch.device('cpu'),
                                    sampler.static_means_for(roots))
        np.testing.assert_allclose(direct.numpy(),cached.numpy(),atol=1e-6,rtol=1e-6)

    def test_fpr_threshold_and_quota(self):
        labels = np.array([0,0,0,0,1,1],dtype=np.int8)
        scores = np.array([.2,.4,.4,.9,.8,.3])
        threshold = fpr_threshold(labels,scores,.25)
        self.assertLessEqual(int(np.sum(scores[labels==0]>=threshold)),1)
        self.assertEqual(quota_metrics(labels,scores,1/3)['alerts'],2)

    def test_monthly_fraud_and_episode_audit(self):
        audit = temporal_support(self.store)
        self.assertEqual(audit['fraud_transactions'],int(np.sum(self.store.labels[:self.store.test_start])))
        self.assertEqual(sum(record['fraud'] for record in audit['monthly']),audit['fraud_transactions'])
        self.assertLessEqual(audit['fraud_episodes'],audit['fraud_transactions'])

    def test_local_training_checkpoint_and_role_isolation(self):
        store = self.store
        roles = {'name':'synthetic','train_end':110,
                 'selection':np.arange(110,130,dtype=np.int32),
                 'calibration':np.arange(130,150,dtype=np.int32),
                 'assessment':np.arange(150,170,dtype=np.int32)}
        output = self.dir/'result'
        result = run_fold(store,roles,'B1',42,output,epochs=2,min_epochs=1,
                          patience=1,batch_size=16,eval_batch_size=32,
                          fanout=3,min_fraud=1,device='cpu')
        self.assertIn('assessment',result)
        self.assertEqual(json.loads((output/'B1_synthetic_seed42'/'status.json').read_text())['status'],'complete')
        self.assertTrue((output/'B1_synthetic_seed42'/'best.pt').exists())
        self.assertEqual(len(np.load(output/'B1_synthetic_seed42'/'assessment_scores.npz')['score']),20)
        changed = dict(roles)
        changed['calibration'] = roles['selection']
        with self.assertRaises(RuntimeError):
            run_fold(store,changed,'B1',42,output,epochs=2,min_epochs=1,
                     patience=1,batch_size=16,eval_batch_size=32,
                     fanout=3,min_fraud=1,device='cpu')


if __name__ == '__main__': unittest.main()
