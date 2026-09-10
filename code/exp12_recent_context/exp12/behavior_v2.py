"""Recent and conditional, label-free behavioral context for EXP12.

Each history is [start, query_time), excluding every simultaneous peer.
Only group-sort/search and prefix sums are used, with no dense group table.
"""
import numpy as np
import pandas as pd

from .behavior import (FEATURE_NAMES as BASE_NAMES, _codes, _pair,
                       _history_index, _sum_before, card_codes, fill_behavior_features)

RECENT = ['amount_7d_delta', 'amount_7d_z', 'amount_7d_support_log',
          'hour_deviation', 'hour_concentration', 'activity_24h_share_7d']
CONDITIONS = ['card_channel', 'card_mcc', 'user_merchant', 'merchant_channel']
CONDITIONAL = ['support_log', 'cold_start', 'recency_log', 'amount_delta', 'amount_z']
PAIRS = ['merchant', 'mcc', 'channel', 'location']
FEATURE_NAMES = (BASE_NAMES + [f'{entity}_{name}' for entity in ['user', 'card'] for name in RECENT]
                 + [f'{group}_{name}' for group in CONDITIONS for name in CONDITIONAL]
                 + [f'card_{group}_{name}' for group in PAIRS for name in ['new', 'surprisal']]
                 + ['card_missing'])


def amount_context(values, valid, order, start, end):
    count = _sum_before(valid[order], start, end)
    mean = _sum_before(values[order], start, end) / np.maximum(count, 1)
    variance = _sum_before(values[order]**2, start, end) / np.maximum(count, 1) - mean**2
    delta = np.where(valid[order] & (count > 0), values[order] - mean, 0.)
    return count, np.clip(delta, -10, 10)/5, np.clip(delta/np.sqrt(np.maximum(variance, .25)), -10, 10)/5


def fill_context_features(df, output):
    if output.shape != (len(df), len(FEATURE_NAMES)):
        raise ValueError('Ukuran buffer fitur context tidak cocok')
    report = fill_behavior_features(df, output[:, :len(BASE_NAMES)], card_policy='per_row')
    seconds = df['_datetime'].to_numpy(dtype='datetime64[s]').astype(np.int64)
    user = _codes(df['User'])
    card, _ = card_codes(df, user, 'per_row')
    raw = pd.to_numeric(df['_amount'], errors='coerce').to_numpy(dtype=np.float64)
    valid = np.isfinite(raw)
    amount = np.where(valid, np.sign(raw)*np.log1p(np.abs(raw)), 0.)
    phase = (seconds % 86400) * (2*np.pi/86400)
    sine, cosine = np.sin(phase), np.cos(phase)
    offset = len(BASE_NAMES)
    card_count = np.empty(len(df), dtype=np.int64)
    for entity, codes in [('user', user), ('card', card)]:
        order, keys, start, end = _history_index(codes, seconds)
        count = end-start
        if entity == 'card':
            card_count[order] = count
        left7 = np.maximum(start, np.searchsorted(keys, keys-604800, side='left'))
        left1 = np.maximum(start, np.searchsorted(keys, keys-86400, side='left'))
        support, delta, z = amount_context(amount, valid, order, left7, end)
        output[order, offset] = delta
        output[order, offset+1] = z
        output[order, offset+2] = np.log1p(support)/5
        # Circular mean keeps 23:59 and 00:01 close. Low concentration means
        # there is no strong usual hour; a cold entity has zero evidence.
        s = _sum_before(sine[order], start, end)/np.maximum(count, 1)
        c = _sum_before(cosine[order], start, end)/np.maximum(count, 1)
        output[order, offset+3] = np.where(count > 0, (1-s*sine[order]-c*cosine[order])/2, 0.)
        output[order, offset+4] = np.sqrt(np.maximum(s*s+c*c, 0.))
        output[order, offset+5] = (end-left1)/np.maximum(end-left7, 1)
        offset += len(RECENT)
    columns = {'merchant': 'Merchant Name', 'mcc': 'MCC', 'channel': 'Use Chip'}
    category = {key: _codes(df[column].fillna('<missing>').astype(str).str.strip())
                for key, column in columns.items()}
    category['location'] = _codes(df['Merchant City'].fillna('<missing>').astype(str) + '|'
                                  + df['Merchant State'].fillna('<missing>').astype(str))
    groups = [('card_channel', card, category['channel']), ('card_mcc', card, category['mcc']),
              ('user_merchant', user, category['merchant']),
              ('merchant_channel', category['merchant'], category['channel'])]
    for _, left, right in groups:
        order, keys, start, end = _history_index(_pair(left, right), seconds)
        count = end-start
        support, delta, z = amount_context(amount, valid, order, start, end)
        age = np.where(count > 0, seconds[order]-seconds[order[np.maximum(end-1, 0)]], 0)
        output[order, offset] = np.log1p(support)/10
        output[order, offset+1] = count == 0
        output[order, offset+2] = np.log1p(age)/16
        output[order, offset+3] = delta
        output[order, offset+4] = z
        offset += len(CONDITIONAL)
    for group in PAIRS:
        order, _, start, end = _history_index(_pair(card, category[group]), seconds)
        count = end-start
        output[order, offset] = count == 0
        output[order, offset+1] = np.log((card_count[order]+1)/(count+1))/10
        offset += 2
    output[:, offset] = df['Card'].isna().to_numpy() if 'Card' in df else 1.
    # Check in chunks: avoid an n x features boolean allocation on full data.
    for start in range(0, len(df), 250000):
        if not np.isfinite(output[start:start+250000]).all():
            raise ValueError('Fitur context non-finite')
    return {**report, 'features': len(FEATURE_NAMES), 'version': 2,
            'card_missing_policy': 'unknown card grouped per user; known cards retained',
            'card_missing_rate': float(output[:, -1].mean()),
            'recent_amount_window_seconds': 604800,
            'conditional_amount_groups': CONDITIONS}
