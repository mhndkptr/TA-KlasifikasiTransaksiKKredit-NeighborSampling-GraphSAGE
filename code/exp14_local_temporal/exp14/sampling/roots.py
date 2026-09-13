"""N:P=30 root sampling with an optional channel-quarter normal mixture."""
import numpy as np


class RootSampler:
    def __init__(self, store, train_end, ratio=30, mode='balanced', seed=42):
        if mode not in {'balanced','channel_quarter'} or ratio < 1:
            raise ValueError('Root sampling harus balanced/channel_quarter dan ratio positif')
        self.store,self.train_end,self.ratio,self.mode = store,train_end,ratio,mode
        y = np.asarray(store.labels[:train_end])
        self.positive = np.flatnonzero(y == 1).astype(np.int32)
        self.negative = np.flatnonzero(y == 0).astype(np.int32)
        if not len(self.positive) or not len(self.negative):
            raise ValueError('Training memerlukan kedua kelas')
        self.negative_count = min(len(self.negative),len(self.positive)*ratio)
        self.rng = np.random.default_rng(seed)
        self.groups = None
        if mode == 'channel_quarter':
            months = np.asarray(store.timestamps[:train_end]).astype('datetime64[ns]').astype('datetime64[M]').astype(np.int32)
            quarter = months//3
            keys = np.asarray(store.channels[:train_end]).astype(np.int32)*100000+quarter
            self.pos_keys,self.pos_counts = np.unique(keys[self.positive],return_counts=True)
            neg_keys = keys[self.negative]
            order = np.argsort(neg_keys,kind='stable')
            unique,start,count = np.unique(neg_keys[order],return_index=True,return_counts=True)
            self.groups = {int(k): self.negative[order[s:s+c]] for k,s,c in zip(unique,start,count)}

    def roots(self):
        if self.mode == 'balanced':
            negative = self.rng.choice(self.negative,self.negative_count,replace=False)
        else:
            wanted = self.negative_count//2
            allocation = self.rng.multinomial(wanted,self.pos_counts/self.pos_counts.sum())
            matched = []
            for key,size in zip(self.pos_keys,allocation):
                pool = self.groups.get(int(key),np.empty(0,dtype=np.int32))
                if len(pool):
                    matched.extend(self.rng.choice(pool,min(int(size),len(pool)),replace=False))
            matched = np.asarray(matched,dtype=np.int32)
            # Uniform backfill excludes every already selected normal.
            excluded = np.zeros(self.train_end,dtype=bool)
            excluded[matched] = True
            pool = self.negative[~excluded[self.negative]]
            remaining = self.negative_count-len(matched)
            negative = np.concatenate((matched,self.rng.choice(pool,remaining,replace=False)))
        all_roots = np.concatenate((self.positive,negative))
        self.rng.shuffle(all_roots)
        return all_roots
