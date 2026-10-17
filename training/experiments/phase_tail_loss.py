"""Focus ordinary-motion errors while preserving each phase's gradient mass."""
import torch


def phase_tail_loss(cross_entropy, labels, previous_legal_wait_s, weights,
        *, focus_share=.5, alpha=1.):
    if (cross_entropy.ndim != 1 or labels.shape != cross_entropy.shape
            or weights.shape != cross_entropy.shape
            or previous_legal_wait_s.shape != cross_entropy.shape
            or not 0 <= focus_share <= 1 or not alpha > 0
            or not torch.isfinite(cross_entropy).all()
            or not torch.isfinite(weights).all() or not (weights > 0).all()):
        raise ValueError('Finite vector losses, positive weights and valid tail parameters required')
    stop = labels == 0
    resumed = ~stop & (previous_legal_wait_s >= 2. - 1e-6)
    ordinary = ~stop & ~resumed
    for selected, expected in [(stop, .5), (ordinary, .25), (resumed, .25)]:
        if not selected.any() or abs(float(weights[selected].sum()) - expected) > 1e-6:
            raise ValueError('Keep stop/ordinary/resumed gradient budgets.5/.25/.25')
    base = (cross_entropy * weights).sum()
    if focus_share == 0:
        return base
    mass = weights[ordinary].sum()
    ordinary_mean = (cross_entropy[ordinary] * weights[ordinary]).sum()
    ordinary_tail = mass / alpha * torch.logsumexp(
        alpha * cross_entropy[ordinary] + (weights[ordinary] / mass).log(), dim=0)
    return base + focus_share * (ordinary_tail - ordinary_mean)
