"""Validation-only model/threshold selection and train-only subtype weighting."""
import numpy as np
import torch


def selection_bounds(count, training, labels=None):
    end = int(count * training['selection_fraction'])
    start = int(end * (1-training['selection_tail_fraction'])) if training['selection_metric'] == 'recent_ap' else 0
    if labels is not None and training['selection_metric'] == 'recent_ap':
        positives = np.flatnonzero(np.asarray(labels[:end]) == 1)
        if len(positives):
            # Expand only backwards inside the selection partition. Never borrow
            # calibration or test labels to obtain enough positive examples.
            need = min(training['selection_min_fraud'], len(positives))
            start = min(start, int(positives[-need]))
    if end <= start:
        raise ValueError('Selection window kosong; perbesar dataset')
    return start, end


def positive_channel_weights(labels, channels, power, cap):
    """Normalize positive subtype costs to mean 1; negatives retain weight 1.

    Does not reapply the population inverse fraud frequency after root sampling.
    Rare positive channels get bounded emphasis; no duplicate fraud roots.
    """
    labels = torch.as_tensor(labels).cpu()
    channels = torch.as_tensor(channels).cpu()
    positive = labels == 1
    weights = torch.ones(len(labels))
    groups, counts = channels[positive].unique(return_counts=True)
    if not len(groups):
        raise ValueError('Training tidak memiliki fraud')
    costs = (counts.max().float()/counts.float()).pow(power).clamp(max=cap)
    costs /= (costs*counts).sum()/counts.sum()
    report = {}
    for group, count, cost in zip(groups, counts, costs):
        weights[positive & (channels == group)] = cost
        report[str(int(group))] = {'train_fraud': int(count), 'positive_loss_weight': float(cost)}
    return weights, report


def stopping_decision(epoch, stale, training):
    return epoch >= training['min_epochs'] and stale >= training['early_stopping_patience']


def validate_selection_labels(labels):
    if len(np.unique(labels)) != 2:
        raise ValueError('Selection window memerlukan dua kelas; perbesar subset/window tanpa melihat test')


def validation_partitions(timestamps, labels, training, evaluation):
    """Select model and threshold on separate samples of the recent regime.

    Weekly time blocks alternate roles, anchored to the Unix epoch. Assignment
    depends on time only; fraud counts validate a predeclared partition and
    NEVER expand it into an older fraud regime. Test data is not an argument.
    """
    times = np.asarray(timestamps, dtype=np.int64)
    labels = np.asarray(labels)
    if len(times) != len(labels) or not len(times) or (np.diff(times) < 0).any():
        raise ValueError('Validation timestamps/labels harus sejajar dan kronologis')
    policy = training.get('validation_partition', 'chronological')
    if policy == 'chronological':
        start, end = selection_bounds(len(labels), training, labels)
        selection = np.arange(start, end)
        calibration = np.arange(int(len(labels)*(1-evaluation['calibration_tail_fraction'])), len(labels))
        report = {'policy': policy, 'selection': {'start_offset': start, 'end_offset': end},
                  'calibration': {'start_offset': int(calibration[0]), 'end_offset': len(labels)}}
    else:
        start = int(len(times)*(1-training['recent_validation_fraction']))
        # Include all equal-time peers at the start of the recent population.
        start = int(np.searchsorted(times, times[start], side='left'))
        block_ns = int(training['validation_block_days'])*86400*10**9
        ids = np.arange(start, len(times))
        role = (times[start:]//block_ns) % 2
        selection, calibration = ids[role == 0], ids[role == 1]
        report = {'policy': policy, 'recent_start_offset': start,
                  'recent_fraction': training['recent_validation_fraction'],
                  'block_days': training['validation_block_days'],
                  'assignment': 'Unix-epoch time block parity: even=selection, odd=calibration',
                  'selection': {}, 'calibration': {}}
        for name, idx in [('selection', selection), ('calibration', calibration)]:
            minimum = training['selection_min_fraud'] if name == 'selection' else evaluation['calibration_min_fraud']
            if len(idx) == 0 or int(labels[idx].sum()) < minimum:
                raise ValueError(f'{name} recent partition kurang dari {minimum} fraud; '
                                 'gunakan subset lebih besar atau protokol validation yang ditetapkan sebelumnya; '
                                 'tidak memperluas otomatis ke periode lama/test')
        if np.intersect1d(selection, calibration).size:
            raise AssertionError('Selection dan calibration overlap')
    validate_selection_labels(labels[selection])
    if evaluation['calibrate_threshold'] and len(np.unique(labels[calibration])) != 2:
        raise ValueError('Calibration window memerlukan dua kelas')
    for name, idx in [('selection', selection), ('calibration', calibration)]:
        report[name].update(count=len(idx), fraud=int(labels[idx].sum()),
                            start_ns=int(times[idx[0]]), end_ns=int(times[idx[-1]]))
    report['overlap_count'] = int(np.intersect1d(selection, calibration).size)
    return selection, calibration, report
