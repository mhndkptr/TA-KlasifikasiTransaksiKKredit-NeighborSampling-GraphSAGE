"""Query-time, label-free causal neighbors for two-layer mean GraphSAGE."""
from pathlib import Path
import time
import numpy as np
import torch

from ..protocol import DAY_NS
from .weights import FrozenWeightContext, weighted_choice


def build_csr(store):
    """Rows are entities; columns are chronological transaction IDs."""
    path = store.path/'entity_csr.npy'
    ptr_path = store.path/'entity_rowptr.npy'
    if path.exists() and ptr_path.exists():
        return np.load(ptr_path,mmap_mode='r'),np.load(path,mmap_mode='r')
    n = store.n
    if n >= np.iinfo(np.int32).max:
        raise ValueError('Jumlah transaksi melampaui indeks int32')
    u = np.asarray(store.users)
    m = np.asarray(store.merchants)
    uc = np.bincount(u,minlength=store.num_users)
    mc = np.bincount(m,minlength=store.num_merchants)
    ptr = np.r_[0,np.cumsum(np.r_[uc,mc],dtype=np.int64)]
    cols = np.lib.format.open_memmap(path,mode='w+',dtype=np.int32,shape=(2*n,))
    cols[:n] = np.argsort(u,kind='stable').astype(np.int32)
    cols[n:] = np.argsort(m,kind='stable').astype(np.int32)
    cols.flush()
    np.save(ptr_path,ptr)
    return np.load(ptr_path,mmap_mode='r'),np.load(path,mmap_mode='r')


def _choice(rng, lo, hi, wanted, exclude=()):
    """Sample small k from an integer range without materializing a large hub."""
    count = hi-lo
    excluded = set(exclude)
    take = min(wanted,max(0,count-len([x for x in excluded if lo <= x < hi])))
    if take <= 0: return []
    if count <= wanted*2:
        available = np.array([x for x in range(lo,hi) if x not in excluded],dtype=np.int64)
        return rng.choice(available,size=take,replace=False).tolist()
    chosen = []
    used = excluded.copy()
    while len(chosen) < take:
        x = int(rng.integers(lo,hi))
        if x not in used:
            used.add(x)
            chosen.append(x)
    return chosen


class TemporalNeighborSampler:
    def __init__(self, store, rowptr, col, *, fanout=25, mode='frozen',
                 recency_days=180, recent_count=12, strategy='uniform',
                 weight_config=None):
        if fanout < 1 or mode not in {'static','frozen','moving','recent'} or (mode == 'recent' and not 0 <= recent_count <= fanout):
            raise ValueError('Konfigurasi sampler temporal tidak valid')
        if strategy not in {'uniform','topology','importance'}:
            raise ValueError('Strategi tetangga tidak valid')
        if strategy != 'uniform' and mode != 'static':
            raise ValueError('Topology/importance membutuhkan satu graf beku; bobot rolling belum tersedia')
        self.store,self.rowptr,self.col = store,rowptr,col
        self.fanout,self.mode = fanout,mode
        self.strategy,self.weight_config = strategy,weight_config
        self.recency_days,self.recent_count = recency_days,recent_count
        self.static_key = None
        self.static_table = None
        self.static_degree = None
        self.static_mean = None
        self.weight_context = None
        self.weight_context_cutoff = None
        self.static_refresh_seconds = 0.

    def prepare_static(self, fit_end, seed, chunk_entities=512):
        """One sampled history and raw mean per entity, reused by every root."""
        if self.mode != 'static':
            return
        key = (fit_end,seed)
        if self.static_key == key:
            return
        refresh_started = time.perf_counter()
        rng = np.random.default_rng(seed)
        if self.strategy != 'uniform' and self.weight_context_cutoff != fit_end:
            self.weight_context = FrozenWeightContext(self.store,fit_end,self.weight_config)
            self.weight_context_cutoff = fit_end
        count_entities = self.store.num_users+self.store.num_merchants
        table = np.full((count_entities,self.fanout),-1,dtype=np.int32)
        degree = np.zeros(count_entities,dtype=np.int32)
        for entity in range(count_entities):
            lo,hi = self.eligible(entity,fit_end)
            degree[entity] = hi-lo
            if self.strategy == 'uniform':
                positions = _choice(rng,lo,hi,self.fanout)
            elif hi-lo <= self.fanout:
                positions = list(range(lo,hi))
            else:
                candidate = np.asarray(self.col[lo:hi])
                weights = self.weight_context.scores(entity,candidate,self.strategy)
                positions = weighted_choice(rng,lo,hi,self.fanout,weights)
            if positions:
                table[entity,:len(positions)] = self.col[positions]
        means = np.zeros((count_entities,self.store.width+4),dtype=np.float32)
        for start in range(0,count_entities,chunk_entities):
            selected = table[start:start+chunk_entities]
            mask = selected>=0
            flat = selected[mask]
            raw = np.zeros((*selected.shape,self.store.width+4),dtype=np.float32)
            if len(flat):
                values = self.store.get(flat)
                raw[mask,:self.store.width] = values
                raw[mask,-3] = 1
            means[start:start+len(selected)] = raw.sum(axis=1)/np.maximum(mask.sum(axis=1)[:,None],1)
        self.static_key,self.static_table,self.static_degree,self.static_mean = key,table,degree,means
        self.static_refresh_seconds += time.perf_counter()-refresh_started

    def static_means_for(self, roots):
        ids = np.asarray(roots,dtype=np.int64)
        endpoints = np.column_stack((self.store.users[ids],
                                     self.store.num_users+self.store.merchants[ids]))
        return self.static_mean[endpoints]

    def diagnostics(self, roots, fit_end, seed=42, maximum=5000):
        """Unlabeled age/support audit on evenly spaced queries."""
        ids = np.asarray(roots,dtype=np.int32)
        if len(ids)>maximum:
            ids = ids[np.linspace(0,len(ids)-1,maximum,dtype=np.int64)]
        neighbors,degree = self.sample(ids,fit_end,seed,training=False)
        valid = neighbors>=0
        age = (np.broadcast_to(np.asarray(self.store.timestamps[ids])[:,None,None],neighbors.shape)[valid]
               -np.asarray(self.store.timestamps[neighbors[valid]]))/DAY_NS
        return {'queries':len(ids),'neighbor_samples':int(valid.sum()),
                'cold_endpoint_fraction':float((degree==0).mean()) if len(ids) else None,
                'mean_degree':float(degree.mean()) if len(ids) else None,
                'neighbor_age_days_median':float(np.median(age)) if len(age) else None,
                'neighbor_age_days_p90':float(np.quantile(age,.9)) if len(age) else None,
                'neighbor_within_180_days':float((age<=180).mean()) if len(age) else None}

    def cutoff_for(self, root, fit_end, *, training=False):
        if self.mode == 'static':
            return fit_end
        ts = int(self.store.timestamps[root])
        start_day_ns = ts//DAY_NS*DAY_NS
        day_cutoff = int(np.searchsorted(self.store.timestamps,start_day_ns,side='left'))
        return min(day_cutoff,fit_end) if training or self.mode == 'frozen' else day_cutoff

    def eligible(self, entity, cutoff):
        lo,hi = int(self.rowptr[entity]),int(self.rowptr[entity+1])
        # CSR rows are sorted by global chronological transaction ID.
        end = lo+int(np.searchsorted(self.col[lo:hi],cutoff,side='left'))
        return lo,end

    def sample(self, roots, fit_end, seed, *, training=False):
        roots = np.asarray(roots,dtype=np.int64)
        if self.mode == 'static':
            self.prepare_static(fit_end,seed)
            endpoints = np.column_stack((self.store.users[roots],
                                         self.store.num_users+self.store.merchants[roots]))
            return self.static_table[endpoints],self.static_degree[endpoints]
        output = np.full((len(roots),2,self.fanout),-1,dtype=np.int32)
        degree = np.zeros((len(roots),2),dtype=np.int32)
        rng = np.random.default_rng(seed)
        for i,root in enumerate(roots):
            cutoff = self.cutoff_for(int(root),fit_end,training=training)
            for side,entity in enumerate((int(self.store.users[root]),
                                          self.store.num_users+int(self.store.merchants[root]))):
                lo,hi = self.eligible(entity,cutoff)
                degree[i,side] = hi-lo
                if hi == lo: continue
                if self.mode == 'recent':
                    before_ns = int(self.store.timestamps[root])-self.recency_days*DAY_NS
                    before_id = int(np.searchsorted(self.store.timestamps,before_ns,side='left'))
                    pivot = lo+int(np.searchsorted(self.col[lo:hi],before_id,side='left'))
                    recent = _choice(rng,pivot,hi,self.recent_count)
                    older = _choice(rng,lo,pivot,self.fanout-len(recent))
                    extra = _choice(rng,pivot,hi,self.fanout-len(recent)-len(older),recent)
                    positions = recent+older+extra
                else:
                    positions = _choice(rng,lo,hi,self.fanout)
                output[i,side,:len(positions)] = self.col[positions]
        return output,degree


def forward_temporal(model, store, roots, neighbors, degree, device, neighbor_mean_override=None):
    """Exactly the two SAGE mean layers for root and its two entity neighbors.

    LayerNorm makes the computation independent of other roots in the batch.
    Sampled entities are duplicated per root; dropout is independently drawn
    for each occurrence in training, as in a query-time graph.
    """
    roots = np.asarray(roots,dtype=np.int64)
    width = store.width+4
    root_features = np.zeros((len(roots),width),dtype=np.float32)
    root_features[:,:store.width] = store.get(roots)
    root_features[:,-3] = 1
    if not model.use_graph_context:
        x = torch.from_numpy(root_features).to(device)
        first = model.from_mean(0,x,torch.zeros_like(x))
        second = model.from_mean(1,first,torch.zeros_like(first))
        return model.classifier(second).squeeze(-1)
    entity = np.zeros((len(roots),2,width),dtype=np.float32)
    entity[:,:,-4] = np.log1p(degree)
    entity[:,0,-2] = 1
    entity[:,1,-1] = 1
    if neighbor_mean_override is None:
        count = (neighbors >= 0).sum(axis=2)
        flat = neighbors.reshape(-1)
        valid = flat >= 0
        sampled = np.zeros((len(flat),width),dtype=np.float32)
        if valid.any():
            sampled[valid,:store.width] = store.get(flat[valid])
            sampled[valid,-3] = 1
        sampled = sampled.reshape(len(roots),2,neighbors.shape[2],width)
        mean = sampled.sum(axis=2)/np.maximum(count[:,:,None],1)
    else:
        mean = neighbor_mean_override
    x = torch.from_numpy(root_features).to(device)
    e = torch.from_numpy(entity).to(device)
    neighbor_mean = torch.from_numpy(mean.astype(np.float32)).to(device)
    first_conv = model.convs[0]
    root_raw = first_conv.lin_l(e.mean(dim=1)) + first_conv.lin_r(x)
    entity_raw = (first_conv.lin_l(neighbor_mean.flatten(0,1)) +
                  first_conv.lin_r(e.flatten(0,1)))
    # A single BatchNorm call sees both transaction and entity destinations.
    # LayerNorm remains pointwise, so the same path serves the temporal arms.
    first = model.activate(0,torch.cat((root_raw,entity_raw),dim=0))
    root_h1,entity_h1 = first[:len(roots)],first[len(roots):]
    root_h2 = model.from_mean(1,root_h1,entity_h1.view(len(roots),2,-1).mean(dim=1))
    return model.classifier(root_h2).squeeze(-1)
