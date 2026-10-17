"""Phase weights for a controlled release-head fit, without choosing actions."""
import numpy as np


def fit_weights(features, labels, previous_legal_wait_s, *, mode='stop_move',
        recovery_motion_share=.5, recovery_wait_s=2.):
    """Return fixed CE weights; stop keeps half of the total loss budget.

    The legacy mode is exactly 0.5 * mean(stop) + 0.5 * mean(move).
    The optional experiment divides the movement budget between resumed and
    ordinary motion. Exact duplicate head inputs share one example's weight.
    Every label must already be admitted from a complete safe trajectory.
    """
    features = np.asarray(features)
    labels = np.asarray(labels)
    waiting = np.asarray(previous_legal_wait_s, float)
    if (features.ndim != 2 or not np.isfinite(features).all()
            or labels.shape != (len(features),) or waiting.shape != labels.shape
            or not np.isfinite(waiting).all() or (waiting < 0).any()
            or not np.issubdtype(labels.dtype, np.integer) or ((labels < 0) | (labels > 11)).any()):
        raise ValueError('Finite head inputs, action labels and causal wait times required')
    if mode not in ('stop_move', 'recovery_phase'):
        raise ValueError('Unknown fit-weight mode')
    if not 0 < recovery_motion_share < 1 or not np.isfinite(recovery_wait_s) or recovery_wait_s <= 0:
        raise ValueError('Positive recovery wait and movement share strictly between zero and one required')
    stop = labels == 0
    if not stop.any() or stop.all():
        raise ValueError('Both safe stop and motion labels required')
    groups = [('stop', stop, .5), ('move', ~stop, .5)]
    if mode == 'recovery_phase':
        recovery = ~stop & (waiting >= recovery_wait_s - 1e-6)
        groups = [('stop', stop, .5),
            ('ordinary_motion', ~stop & ~recovery, .5 * (1. - recovery_motion_share)),
            ('resumed_motion', recovery, .5 * recovery_motion_share)]
        if any(not selected.any() for _, selected, _ in groups):
            raise ValueError('Recovery-phase fit needs safe resumed and ordinary motion sources; collect data first')
    keys = [np.ascontiguousarray(value).tobytes() for value in features]
    known = {}
    for key, label in zip(keys, labels):
        if key in known and known[key] != int(label):
            raise ValueError('Conflicting action labels for identical trainable input')
        known[key] = int(label)
    weights = np.zeros(len(labels), np.float64)
    records = []
    for name, selected, budget in groups:
        indices = np.flatnonzero(selected)
        counts = {}
        for i in indices:
            counts[keys[i]] = counts.get(keys[i], 0) + 1
        relative = np.ones(len(indices)) if mode == 'stop_move' else np.asarray(
            [1. / counts[keys[i]] for i in indices])
        weights[indices] = budget * relative / relative.sum()
        records.append(dict(phase=name, labels=len(indices), unique_head_inputs=len(counts),
            allocated_loss_weight=float(weights[indices].sum())))
    if not np.isclose(weights.sum(), 1.) or not (weights > 0).all():
        raise ValueError('All admitted examples need positive normalized weight')
    return weights, dict(mode=mode, duplicate_weighting='inverse exact-input multiplicity' if mode == 'recovery_phase' else 'legacy unchanged',
        recovery_wait_s=float(recovery_wait_s), recovery_motion_share=float(recovery_motion_share),
        phases=records, stop_loss_weight=float(weights[stop].sum()),
        no_action_override=True, no_source_safety_admission=True)
