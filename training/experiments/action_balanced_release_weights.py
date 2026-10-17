"""Balance movement classes inside fixed stop/ordinary/recovery loss budgets."""
import numpy as np

from experiments.release_fit_weights import fit_weights as original_weights


MODE = 'recovery_phase_action_balanced'


def fit_weights(features, labels, previous_legal_wait_s, *, mode='stop_move',
        recovery_motion_share=.5, recovery_wait_s=2.):
    if mode != MODE:
        return original_weights(features, labels, previous_legal_wait_s, mode=mode,
            recovery_motion_share=recovery_motion_share, recovery_wait_s=recovery_wait_s)
    weights, report = original_weights(features, labels, previous_legal_wait_s,
        mode='recovery_phase', recovery_motion_share=recovery_motion_share,
        recovery_wait_s=recovery_wait_s)
    labels = np.asarray(labels)
    waiting = np.asarray(previous_legal_wait_s)
    resumed = (labels != 0) & (waiting >= recovery_wait_s - 1e-6)
    classes = []
    for phase, selected in [('ordinary_motion', (labels != 0) & ~resumed),
            ('resumed_motion', resumed)]:
        budget = weights[selected].sum()
        actions = np.unique(labels[selected])
        for action in actions:
            indices = selected & (labels == action)
            weights[indices] *= (budget / len(actions)) / weights[indices].sum()
            classes.append(dict(phase=phase, action=int(action), labels=int(indices.sum()),
                allocated_loss_weight=float(weights[indices].sum())))
    assert np.isclose(weights.sum(), 1.) and (weights > 0).all()
    assert np.isclose(weights[labels == 0].sum(), .5)
    return weights, dict(report, mode=MODE, balanced_movement_classes=True,
        stop_budget_and_duplicate_weighting_unchanged=True,
        class_budgets=classes, no_action_override=True)
