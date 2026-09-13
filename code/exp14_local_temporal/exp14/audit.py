"""Read-only support audit for chip fraud and independent user-card episodes."""
import duckdb
import pandas as pd


def temporal_support(store, folds=(), episode_gap_days=7):
    relation = str((store.path/'chronological.parquet').resolve().as_posix()).replace("'","''")
    conn = duckdb.connect()
    rows = conn.execute(
        f"select strftime(ts,'%Y-%m') as month,use_chip_key,count(*) as rows,"
        "sum(label) as fraud,count(distinct user_key) filter (where label=1) as fraud_users,"
        "count(distinct user_key||'|'||card_key) filter (where label=1) as fraud_cards "
        f"from read_parquet('{relation}') where tx_id < {store.test_start} "
        'group by 1,2 order by 1,2').fetchall()
    monthly = [{'month':str(month),'channel':str(channel),'rows':int(count),
                'fraud':int(fraud),'fraud_users':int(users),'fraud_cards':int(cards)}
               for month,channel,count,fraud,users,cards in rows]
    fraud = conn.execute(
        f"select tx_id,ts,user_key,card_key,use_chip_key from read_parquet('{relation}') "
        f"where label=1 and tx_id < {store.test_start} "
        "order by user_key,card_key,ts,source_row").fetch_df()
    conn.close()
    if len(fraud):
        ts = pd.to_datetime(fraud['ts'])
        prior = ts.groupby([fraud.user_key,fraud.card_key]).shift()
        start = prior.isna() | ((ts-prior).dt.total_seconds() > episode_gap_days*86400)
        fraud['month'] = ts.dt.to_period('M').astype(str)
        fraud['episode_start'] = start.astype(int)
        episode = fraud.groupby(['month','use_chip_key'],sort=True)['episode_start'].sum().to_dict()
    else:
        episode = {}
        fraud['episode_start'] = pd.Series(dtype=int)
    for record in monthly:
        record['fraud_episodes'] = int(episode.get((record['month'],record['channel']),0))
    fold_support = {}
    for fold in folds:
        fold_support[fold.name] = {}
        for role,(lo,hi) in fold.ranges().items():
            if role == 'train':
                continue
            part = fraud[(fraud.tx_id >= lo) & (fraud.tx_id < hi)]
            fold_support[fold.name][role] = {
                'fraud':len(part),
                'chip_fraud':int((part.use_chip_key == 'Chip Transaction').sum()),
                'fraud_users':int(part.user_key.nunique()),
                'fraud_cards':int(part[['user_key','card_key']].drop_duplicates().shape[0]),
                'fraud_episodes':int(part.episode_start.sum())}
    return {'scope':'development_before_old_test','test_start_tx_id':store.test_start,
            'monthly':monthly,'fraud_transactions':len(fraud),
            'fraud_episodes':int(sum(episode.values())),
            'episode_gap_days':episode_gap_days,'fold_support':fold_support}
