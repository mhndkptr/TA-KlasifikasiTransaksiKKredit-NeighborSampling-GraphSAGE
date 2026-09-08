"""Label-free behavioral features from observations strictly before query time.

Held-out observations update these counters, but never the graph or model.
All simultaneous transactions see the same history, independent of CSV order.
No per-transaction Python loop or dense user x merchant table is allocated.
"""
import logging
import numpy as np
import pandas as pd

HISTORY_NAMES = ['prior_count_log', 'cold_start', 'seconds_since_previous_log',
                 'count_1h_log', 'count_24h_log', 'count_7d_log',
                 'amount_log_delta', 'amount_log_z']
PAIRS = ['merchant', 'mcc', 'channel', 'location']
FEATURE_NAMES = [f'{entity}_{name}' for entity in ['user', 'card'] for name in HISTORY_NAMES]
FEATURE_NAMES += [f'user_{pair}_{name}' for pair in PAIRS for name in ['new', 'surprisal']]


def _codes(values):
    return pd.factorize(values, sort=False)[0].astype(np.int64)


def _pair(left, right):
    # Codes are compact, nonnegative. No raw ID magnitude enters the model.
    width = int(right.max()) + 1
    return _codes(left * width + right)


def _history_index(codes, seconds):
    # Pack group and timestamp into int64 for vectorized searchsorted. Reserve
    # enough space for a seven-day backward query; clip at group start anyway.
    relative = seconds - seconds.min()
    span = int(relative.max()) + 604801
    if (int(codes.max()) + 1) * span > np.iinfo(np.int64).max:
        raise ValueError('Rentang waktu/entitas terlalu besar untuk indeks int64')
    keys = codes * span + relative
    order = np.argsort(keys, kind='stable')
    sorted_keys = keys[order]
    group_start = np.maximum.accumulate(np.where(
        np.r_[True, codes[order][1:] != codes[order][:-1]], np.arange(len(order)), 0))
    # left excludes the current transaction AND every equal-timestamp peer.
    end = np.searchsorted(sorted_keys, sorted_keys, side='left')
    return order, sorted_keys, group_start, end


def _sum_before(values, start, end):
    prefix = np.r_[0., np.cumsum(values, dtype=np.float64)]
    return prefix[end] - prefix[start]


def fill_behavior_features(df, output):
    """Fill a preallocated n x 24 view; return a small, serializable audit."""
    if output.shape != (len(df), len(FEATURE_NAMES)):
        raise ValueError('Ukuran buffer fitur perilaku tidak cocok')
    seconds = df['_datetime'].to_numpy(dtype='datetime64[s]').astype(np.int64)
    if (np.diff(seconds) < 0).any():
        raise ValueError('Fitur perilaku memerlukan urutan kronologis')
    user = _codes(df['User'])
    card_available = 'Card' in df and df['Card'].notna().all()
    if card_available:
        card = _pair(user, _codes(df['Card']))
    else:
        card = user
        logging.getLogger('exp11').warning('Card kosong/tidak tersedia: histori card memakai histori user')
    raw = pd.to_numeric(df['_amount'], errors='coerce').to_numpy(dtype=np.float64)
    valid = np.isfinite(raw)
    amounts = np.where(valid, np.sign(raw) * np.log1p(np.abs(raw)), 0.)
    offset = 0
    user_counts = np.empty(len(df), dtype=np.int64)
    cold_rates = {}
    for name, codes in [('user', user), ('card', card)]:
        order, keys, start, end = _history_index(codes, seconds)
        count = end-start
        cold = count == 0
        if name == 'user':
            user_counts[order] = count
        age = np.where(cold, 0, seconds[order]-seconds[order[np.maximum(end-1, 0)]])
        output[order, offset] = np.log1p(count)/10
        output[order, offset+1] = cold
        output[order, offset+2] = np.log1p(age)/16
        for j, window in enumerate([3600, 86400, 604800], 3):
            left = np.maximum(start, np.searchsorted(keys, keys-window, side='left'))
            output[order, offset+j] = np.log1p(end-left)/5
        valid_count = _sum_before(valid[order], start, end)
        mean = _sum_before(amounts[order], start, end)/np.maximum(valid_count, 1)
        var = _sum_before(amounts[order]**2, start, end)/np.maximum(valid_count, 1)-mean**2
        delta = np.where(valid[order] & (valid_count > 0), amounts[order]-mean, 0)
        output[order, offset+6] = np.clip(delta, -10, 10)/5
        # A floor avoids absurd scores for users with one/constant past amount.
        output[order, offset+7] = np.clip(delta/np.sqrt(np.maximum(var, .25)), -10, 10)/5
        cold_rates[name] = float(cold.mean())
        offset += len(HISTORY_NAMES)
    for pair in PAIRS:
        if pair == 'location':
            values = (df['Merchant City'].fillna('<missing>').astype(str) + '|' +
                      df['Merchant State'].fillna('<missing>').astype(str))
        else:
            column = {'merchant': 'Merchant Name', 'mcc': 'MCC', 'channel': 'Use Chip'}[pair]
            values = df[column].fillna('<missing>').astype(str)
        pair_codes = _pair(user, _codes(values))
        order, keys, start, end = _history_index(pair_codes, seconds)
        count = end-start
        output[order, offset] = count == 0
        output[order, offset+1] = np.log((user_counts[order]+1)/(count+1))/10
        offset += 2
    return {'policy': 'strictly_earlier_unlabeled_observations',
            'equal_timestamp_policy': 'exclude_all_peers', 'card_available': bool(card_available),
            'cold_start_rate': cold_rates, 'features': len(FEATURE_NAMES),
            'scaling': 'fixed log divisors; no fitted future statistics',
            'heldout_observations_update_counters': True, 'labels_used': False}
