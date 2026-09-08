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
