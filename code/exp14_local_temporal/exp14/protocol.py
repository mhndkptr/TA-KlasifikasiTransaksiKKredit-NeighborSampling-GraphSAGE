"""All cutoffs are chronological row boundaries; ties stay on the earlier side."""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import numpy as np

DAY_NS = 86_400_000_000_000


def tie_safe_boundary(timestamps, nominal):
    timestamps = np.asarray(timestamps)
    if not 0 <= nominal < len(timestamps) or (np.diff(timestamps) < 0).any():
        raise ValueError('Timestamp tidak urut atau batas di luar data')
    return int(np.searchsorted(timestamps, timestamps[nominal], side='right'))


def split_70_15_15(timestamps):
    n = len(timestamps)
    train = tie_safe_boundary(timestamps, int(.70*n))
    val = tie_safe_boundary(timestamps, int(.85*n))
    if not 0 < train < val < n:
        raise ValueError('Split temporal tidak valid')
    return train, val


@dataclass(frozen=True)
class Fold:
    name: str
    train_end: int
    selection_start: int
    selection_end: int
    calibration_end: int
    assessment_end: int
    delay_days: int

    def ranges(self):
        return {'train': (0, self.train_end),
                'selection': (self.selection_start, self.selection_end),
                'calibration': (self.selection_end, self.calibration_end),
                'assessment': (self.calibration_end, self.assessment_end)}

    def manifest(self):
        return asdict(self)


def _boundary(timestamps, day_ns):
    return int(np.searchsorted(timestamps, day_ns, side='left'))


def calendar_folds(timestamps, val_start, test_start, origins=3, window_days=90,
                   delay_days=0, min_fraud=25, min_chip_fraud=25,
                   labels=None, channels=None, chip_code=None,
                   assessment_end_ns=None):
    """Fixed calendar; label counts audit support but never move a boundary."""
    if origins < 1 or window_days < 1 or delay_days < 0:
        raise ValueError('Origin/window/delay tidak valid')
    ts = np.asarray(timestamps)
    if not 0 < val_start < test_start < len(ts):
        raise ValueError('Periode pengembangan tidak valid')
    end_day = ((int(ts[test_start]) if assessment_end_ns is None else int(assessment_end_ns))
               // DAY_NS) * DAY_NS
    if not int(ts[val_start]) < end_day <= int(ts[test_start]):
        raise ValueError('Akhir assessment harus berada dalam periode pengembangan')
    folds, audits = [], []
    for j in range(origins, 0, -1):
        assessment_end = end_day - (j-1)*window_days*DAY_NS
        assessment_start = assessment_end - window_days*DAY_NS
        calibration_start = assessment_start - window_days*DAY_NS
        selection_start = calibration_start - window_days*DAY_NS
        train_cutoff = selection_start - delay_days*DAY_NS
        bounds = [_boundary(ts, x) for x in (train_cutoff, selection_start,
                                             calibration_start, assessment_start, assessment_end)]
        # With zero label delay, train cutoff and selection start are the same
        # boundary by design. Only the four role boundaries must be distinct.
        if (bounds[0] <= val_start or bounds[-1] > test_start or
                bounds[0] > bounds[1] or len(set(bounds[1:])) != 4 or
                (delay_days > 0 and bounds[0] == bounds[1])):
            audits.append({'origin': j, 'eligible': False, 'reason': 'outside development calendar'})
            continue
        fold = Fold(f'origin_{origins-j+1}', *bounds, delay_days)
        support = {'origin': j, 'eligible': True, 'bounds': fold.manifest()}
        if labels is not None:
            y = np.asarray(labels)
            for name, (lo, hi) in fold.ranges().items():
                if name == 'train': continue
                count = int(y[lo:hi].sum())
                chip = (int(((y[lo:hi] == 1) & (np.asarray(channels)[lo:hi] == chip_code)).sum())
                        if channels is not None and chip_code is not None else None)
                support[name] = {'rows': hi-lo, 'fraud': count, 'chip_fraud': chip,
                                 'min_fraud_met': count >= min_fraud,
                                 'chip_support_met': chip is None or chip >= min_chip_fraud}
            support['eligible'] = all(support[n]['min_fraud_met'] for n in
                                      ('selection','calibration','assessment'))
            if not support['eligible']:
                support['reason'] = 'one or more evaluation roles lack fraud; calendar unchanged'
        folds.append(fold)
        audits.append(support)
    return folds, audits


def time_iso(ns):
    return datetime.fromtimestamp(int(ns)/1e9, tz=timezone.utc).isoformat()
