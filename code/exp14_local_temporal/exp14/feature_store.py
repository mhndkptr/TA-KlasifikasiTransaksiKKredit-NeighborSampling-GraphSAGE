"""Local disk-backed 220-feature store. Transactions are indexed by chronological ID.

Context is computed per user, which preserves every EXP12 strictly-past group
except merchant-channel. That cross-user group is computed in a second pass.
Neither pass uses the fraud label to create features.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd

from .legacy import EXP12_DIR, CONTEXT_NAMES, FeatureEncoder, fill_context_features, amount_context
from .protocol import split_70_15_15


def _sql(value):
    return "'" + str(value).replace("'", "''") + "'"


def _query(csv_path, max_rows):
    scan = (f"read_csv({_sql(Path(csv_path).resolve().as_posix())}, header=true, "
            "auto_detect=true, all_varchar=true, sample_size=20480)")
    if max_rows is not None:
        scan = f'(select * from {scan} limit {int(max_rows)})'
    source = f'select *, row_number() over ()-1 as source_row from {scan}'
    def col(name):
        return f"coalesce(nullif(trim(cast(\"{name}\" as varchar)),''),'<missing>')"
    stamp = ("try_strptime(concat(trim(\"Year\"),'-',lpad(trim(\"Month\"),2,'0'),'-',"
             "lpad(trim(\"Day\"),2,'0'),' ',trim(\"Time\")),"
             "['%Y-%m-%d %H:%M','%Y-%m-%d %H:%M:%S'])")
    return f"""with source as ({source}) select
      cast(source_row as bigint) source_row,
      cast("User" as varchar) user_key, {col('Card')} card_key,
      cast("Merchant Name" as varchar) merchant_key,
      {col('Merchant City')} city_key, {col('Merchant State')} state_key,
      {col('Zip')} zip_key, {col('Use Chip')} use_chip_key,
      {col('MCC')} mcc_key, {col('Errors?')} errors_key,
      {stamp} ts,
      try_cast(replace(replace(trim("Amount"),'$',''),',','') as double) amount,
      case lower(trim("Is Fraud?")) when 'yes' then 1 when 'no' then 0 else null end as label
      from source"""


def _connection(cache_dir, memory_limit, threads):
    conn = duckdb.connect(str(cache_dir / 'local.duckdb'))
    temp = cache_dir / 'spill'
    temp.mkdir(parents=True, exist_ok=True)
    conn.execute(f"set temp_directory={_sql(temp.resolve().as_posix())}")
    conn.execute(f"set memory_limit={_sql(memory_limit)}")
    conn.execute(f'set threads={int(threads)}')
    return conn


def _sorted_parquet(conn, csv_path, target, max_rows, row_group_size):
    parsed = _query(csv_path, max_rows)
    audit = conn.execute(f"select count(*), count(*) filter (where ts is null), "
                         "count(*) filter (where label is null), "
                         f"count(*) filter (where user_key is null or merchant_key is null) from ({parsed}) p").fetchone()
    if audit[0] < 3 or any(audit[1:]):
        raise ValueError(f'Data CSV tidak valid: rows/time/label/entity={audit}')
    ordered = (f'with parsed as ({parsed}) select '
               'row_number() over (order by ts,source_row)-1 as tx_id,* '
               'from parsed order by ts,source_row')
    conn.execute(f'copy ({ordered}) to {_sql(target.resolve().as_posix())} '
                 f'(format parquet,compression zstd,row_group_size {int(row_group_size)})')
    return int(audit[0])


def _read_chunks(cursor, key_columns, batch_rows=100_000):
    """Stream complete sorted groups; retain only one unfinished group."""
    carry = None
    vectors = max(1, batch_rows // 2048)
    while True:
        chunk = cursor.fetch_df_chunk(vectors)
        if chunk.empty:
            break
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        keys = chunk[key_columns]
        if len(key_columns) == 1:
            changes = keys.iloc[:, 0].ne(keys.iloc[:, 0].shift(-1)).to_numpy()
        else:
            changes = keys.ne(keys.shift(-1)).any(axis=1).to_numpy()
        boundary = np.flatnonzero(changes)
        if len(boundary) < 2:
            carry = chunk
            continue
        last_start = int(boundary[-2])+1
        complete = chunk.iloc[:last_start]
        carry = chunk.iloc[last_start:].copy()
        for _, group in complete.groupby(key_columns, sort=False, dropna=False):
            yield group
    if carry is not None and len(carry):
        for _, group in carry.groupby(key_columns, sort=False, dropna=False):
            yield group


def _encoder_state(conn, relation, train_end):
    median = conn.execute(f'select median(amount) from {relation} where tx_id < {train_end}').fetchone()[0]
    median = 0. if median is None else float(median)
    logged = f'sign(coalesce(amount,{median}))*ln(1+abs(coalesce(amount,{median})))'
    q1, center, q3 = conn.execute(
        f'select quantile_cont({logged},.25),quantile_cont({logged},.5),quantile_cont({logged},.75) '
        f'from {relation} where tx_id < {train_end}').fetchone()
    categories = {}
    for original, column in [('Use Chip', 'use_chip_key'), ('MCC', 'mcc_key'), ('Errors?', 'errors_key')]:
        values = [str(row[0]) for row in conn.execute(
            f'select distinct {column} from {relation} where tx_id < {train_end} order by {column}').fetchall()]
        categories[original] = sorted(set(values) | {'<missing>'})
    locations = conn.execute(
        f"select city_key||'|'||state_key||'|'||zip_key, count(*)::double/{train_end} "
        f'from {relation} where tx_id < {train_end} group by 1').fetchall()
    return {'version': 1, 'amount': {'median': median, 'center': float(center),
                                    'scale': float(q3-q1) or 1.},
            'categorical': categories,
            'location_frequency': {str(key): float(freq) for key, freq in locations}}


def _frame(group):
    frame = group.rename(columns={'user_key':'User', 'card_key':'Card',
        'merchant_key':'Merchant Name', 'city_key':'Merchant City',
        'state_key':'Merchant State', 'zip_key':'Zip',
        'use_chip_key':'Use Chip', 'mcc_key':'MCC', 'errors_key':'Errors?'}).copy()
    frame['Card'] = frame['Card'].replace('<missing>', pd.NA)
    frame['_datetime'] = pd.to_datetime(frame['ts'])
    frame['_amount'] = frame['amount'].astype('float64')
    frame['_dow'] = frame['_datetime'].dt.dayofweek
    frame['_weekend'] = frame['_dow'].isin((5,6)).astype(float)
    frame['Month'] = frame['_datetime'].dt.month
    return frame


def _map_id(key, mapping):
    if key not in mapping:
        mapping[key] = len(mapping)
    return mapping[key]


def _memmap(path, shape, dtype):
    return np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)


class LocalFeatureStore:
    def __init__(self, cache_dir):
        self.path = Path(cache_dir)
        self.meta = json.loads((self.path/'complete.json').read_text(encoding='utf-8'))
        self.n = self.meta['rows']
        self.width = self.meta['features']
        self.num_users = self.meta['num_users']
        self.num_merchants = self.meta['num_merchants']
        def read(name): return np.load(self.path/f'{name}.npy', mmap_mode='r')
        self.feature_rows = read('feature_rows')
        self.merchant_rows = read('merchant_rows')
        self.features = read('features_user')
        self.merchant_features = read('features_merchant_channel')
        self.labels = read('labels')
        self.timestamps = read('timestamps')
        self.users = read('users')
        self.merchants = read('merchants')
        self.channels = read('channels')
        self.source_rows = read('source_rows')
        self.train_end = self.meta['train_end']
        self.test_start = self.meta['test_start']
        self.merchant_column = self.meta['merchant_column']

    def get(self, tx_ids):
        ids = np.asarray(tx_ids, dtype=np.int64)
        data = np.asarray(self.features[self.feature_rows[ids]], dtype=np.float32).copy()
        data[:, self.merchant_column:self.merchant_column+5] = np.asarray(
            self.merchant_features[self.merchant_rows[ids]], dtype=np.float32)
        return data

    def close(self):
        """Release read-only memmap handles before deleting a Windows test cache."""
        for name in ('feature_rows','merchant_rows','features','merchant_features',
                     'labels','timestamps','users','merchants','channels','source_rows'):
            array = getattr(self,name)
            if hasattr(array,'_mmap') and array._mmap is not None:
                array._mmap.close()


def prepare(csv_path, cache_root, *, max_rows=None, memory_limit='4GB', threads=4,
            chunk_rows=100_000):
    """Build once; an incomplete directory is never treated as a usable cache."""
    csv_path = Path(csv_path).resolve()
    stat = csv_path.stat()
    definitions = [Path(__file__),EXP12_DIR/'exp12'/'data.py',
                   EXP12_DIR/'exp12'/'behavior.py',EXP12_DIR/'exp12'/'behavior_v2.py']
    code_digest = hashlib.sha256(b''.join(p.read_bytes() for p in definitions)).hexdigest()
    identity = {'source': str(csv_path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                'max_rows': max_rows, 'schema': 2, 'feature_code_sha256': code_digest}
    digest = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    cache_dir = Path(cache_root).resolve()/f'contextual_{digest}'
    complete = cache_dir/'complete.json'
    if complete.exists():
        result = LocalFeatureStore(cache_dir)
        if result.meta['identity'] == identity:
            return result
        raise ValueError('Identitas cache tidak cocok')
    if cache_dir.exists():
        raise RuntimeError(f'Cache belum selesai: {cache_dir}. Periksa atau gunakan path cache baru.')
    cache_dir.mkdir(parents=True)
    start = time.perf_counter()
    conn = _connection(cache_dir, memory_limit, threads)
    parquet = cache_dir/'chronological.parquet'
    n = _sorted_parquet(conn,csv_path,parquet,max_rows,chunk_rows)
    relation = f'read_parquet({_sql(parquet.resolve().as_posix())})'
    bounds = []
    for ratio in (.70,.85):
        nominal_ts = conn.execute(f'select ts from {relation} where tx_id={int(n*ratio)}').fetchone()[0]
        bounds.append(int(conn.execute(f'select count(*) from {relation} where ts <= ?', [nominal_ts]).fetchone()[0]))
    train_end, test_start = bounds
    if not 0 < train_end < test_start < n:
        raise ValueError('Split temporal tie-safe kosong')
    state = _encoder_state(conn,relation,train_end)
    encoder = FeatureEncoder(state)
    width = len(encoder.feature_names)+len(CONTEXT_NAMES)
    user_features = _memmap(cache_dir/'features_user.npy',(n,width),np.float16)
    user_rows = _memmap(cache_dir/'feature_rows.npy',(n,),np.int32)
    merchant_rows = _memmap(cache_dir/'merchant_rows.npy',(n,),np.int32)
    labels = _memmap(cache_dir/'labels.npy',(n,),np.int8)
    timestamps = _memmap(cache_dir/'timestamps.npy',(n,),np.int64)
    users = _memmap(cache_dir/'users.npy',(n,),np.int32)
    merchants = _memmap(cache_dir/'merchants.npy',(n,),np.int32)
    channels = _memmap(cache_dir/'channels.npy',(n,),np.int16)
    source_rows = _memmap(cache_dir/'source_rows.npy',(n,),np.int32)
    channel_map = {v:i for i,v in enumerate(state['categorical']['Use Chip'])}
    user_map, merchant_map = {},{}
    select = (f'select tx_id,source_row,user_key,card_key,merchant_key,city_key,state_key,'
              f'zip_key,use_chip_key,mcc_key,errors_key,ts,amount,label from {relation} '
              f'order by user_key,ts,source_row')
    conn.execute(select)
    position = 0
    for group in _read_chunks(conn,['user_key'],chunk_rows):
        length = len(group)
        frame = _frame(group)
        context = np.empty((length,len(CONTEXT_NAMES)),dtype=np.float32)
        fill_context_features(frame,context)
        base,_ = encoder.transform(frame)
        user_features[position:position+length] = np.concatenate((base,context),axis=1).astype(np.float16)
        ids = group.tx_id.to_numpy(dtype=np.int64)
        user_rows[ids] = np.arange(position,position+length,dtype=np.int32)
        labels[ids] = group.label.to_numpy(np.int8)
        timestamps[ids] = frame['_datetime'].to_numpy(dtype='datetime64[ns]').view(np.int64)
        source_rows[ids] = group.source_row.to_numpy(np.int32)
        users[ids] = _map_id(str(group.user_key.iloc[0]),user_map)
        merchants[ids] = np.fromiter((_map_id(str(v),merchant_map) for v in group.merchant_key),np.int32,length)
        channels[ids] = group.use_chip_key.map(channel_map).fillna(-1).to_numpy(np.int16)
        position += length
    if position != n:
        raise AssertionError(f'User pass kurang baris: {position}/{n}')
    merchant_column = (encoder.feature_names+CONTEXT_NAMES).index('merchant_channel_support_log')
    merchant_values = _memmap(cache_dir/'features_merchant_channel.npy',(n,5),np.float16)
    select = (f'select tx_id,source_row,merchant_key,use_chip_key,ts,amount from {relation} '
              f'order by merchant_key,use_chip_key,ts,source_row')
    conn.execute(select)
    position = 0
    for group in _read_chunks(conn,['merchant_key','use_chip_key'],chunk_rows):
        length = len(group)
        seconds = pd.to_datetime(group.ts).to_numpy(dtype='datetime64[s]').astype(np.int64)
        end = np.searchsorted(seconds,seconds,side='left')
        raw = group.amount.to_numpy(np.float64)
        valid = np.isfinite(raw)
        amount = np.where(valid,np.sign(raw)*np.log1p(np.abs(raw)),0.)
        support,delta,z = amount_context(amount,valid,np.arange(length),np.zeros(length,dtype=np.int64),end)
        age = np.where(end>0,seconds-seconds[np.maximum(end-1,0)],0)
        block = np.column_stack((np.log1p(support)/10,end==0,np.log1p(age)/16,delta,z))
        merchant_values[position:position+length] = block.astype(np.float16)
        ids = group.tx_id.to_numpy(dtype=np.int64)
        merchant_rows[ids] = np.arange(position,position+length,dtype=np.int32)
        position += length
    if position != n:
        raise AssertionError(f'Merchant pass kurang baris: {position}/{n}')
    for array in (user_features,user_rows,merchant_rows,labels,timestamps,users,merchants,channels,source_rows,merchant_values):
        array.flush()
    if (np.diff(timestamps) < 0).any():
        raise AssertionError('Timestamp cache tidak kronologis')
    if split_70_15_15(timestamps) != (train_end,test_start):
        raise AssertionError('Batas DuckDB/NumPy berbeda')
    monthly = conn.execute(
        f"select strftime(ts,'%Y-%m') as month,use_chip_key,count(*) as rows,"
        f"sum(label) as fraud,count(distinct user_key) filter (where label=1) as fraud_users,"
        f"count(distinct user_key||'|'||card_key) filter (where label=1) as fraud_cards "
        f'from {relation} group by 1,2 order by 1,2').fetchall()
    monthly_report = [{'month':str(month),'channel':str(channel),'rows':int(rows),
                       'fraud':int(fraud),'fraud_users':int(fraud_users),
                       'fraud_cards':int(fraud_cards)}
                      for month,channel,rows,fraud,fraud_users,fraud_cards in monthly]
    (cache_dir/'monthly_audit.json').write_text(
        json.dumps(monthly_report,ensure_ascii=False,indent=2),encoding='utf-8')
    meta = {'identity':identity,'rows':n,'features':width,'num_users':len(user_map),
            'num_merchants':len(merchant_map),'train_end':train_end,'test_start':test_start,
            'merchant_column':merchant_column,'feature_names':encoder.feature_names+CONTEXT_NAMES,
            'encoder':state,'channel_names':state['categorical']['Use Chip'],
            'seconds':time.perf_counter()-start,'backend':'local_duckdb_contextual_memmap',
            'monthly_audit':'monthly_audit.json'}
    complete.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    conn.close()
    return LocalFeatureStore(cache_dir)
