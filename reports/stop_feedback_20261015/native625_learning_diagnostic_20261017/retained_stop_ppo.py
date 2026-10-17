"""Experimental PPO update guard; does not change observations or permissions.

The live pilot is not modified. This class is prepared for a subsequent
experiment: an actor update must retain every admitted reference STOP label.
It backtracks the proposed Adam parameter step, or restores both actor and
actor optimizer on rejection. Critic updates remain valid and are retained.
Reference retention is a data constraint, not a physical safety certificate.
"""
import copy

import torch

from experiments.native625_onpolicy import Native625OnPolicy


class RetainedStopOnPolicy(Native625OnPolicy):
    def __init__(self, actor, stop_observations, learning_rate=1e-4,
            temperature=2., maximum_backtracks=12, check_batch_size=128):
        super().__init__(actor, learning_rate=learning_rate, temperature=temperature)
        stops = torch.as_tensor(stop_observations, dtype=torch.float32).detach().clone()
        if stops.ndim != 2 or stops.shape[1] != 625 or not len(stops) or not torch.isfinite(stops).all():
            raise ValueError('Finite admitted native625 STOP observations are required')
        if not 0 <= maximum_backtracks <= 24 or check_batch_size < 1:
            raise ValueError('Bounded backtracking and positive check batches are required')
        self.register_buffer('reference_stop_observations', stops)
        self.maximum_backtracks = int(maximum_backtracks)
        self.check_batch_size = int(check_batch_size)
        if self.stop_mismatches():
            raise ValueError('Initial actor already violates reference STOP retention')

    @torch.no_grad()
    def stop_mismatches(self):
        wrong = 0
        for start in range(0, len(self.reference_stop_observations), self.check_batch_size):
            commands = self.actor(self.reference_stop_observations[start:start+self.check_batch_size])
            wrong += int((commands.abs().amax(1) > 0.).sum())
        return wrong

    def update(self, *args, **kwargs):
        # Retain the optimizer state as well as parameters: a rejected Adam
        # step must not leave momentum or step counters from an unexecuted update.
        parameters = dict(self.actor.head.named_parameters())
        before = {name: value.detach().clone() for name, value in parameters.items()}
        optimizer_before = copy.deepcopy(self.actor_optimizer.state_dict())
        statistics = super().update(*args, **kwargs)
        attempted = bool(statistics['actor_updated'])
        statistics.update(reference_stop_guard=True, actor_update_attempted=attempted,
            proposed_stop_mismatches=0, accepted_actor_step_fraction=1. if attempted else 0.,
            backtrack_checks=0, actor_optimizer_restored=False)
        if not attempted:
            return statistics
        wrong = self.stop_mismatches()
        statistics['proposed_stop_mismatches'] = wrong
        if not wrong:
            return statistics
        proposed = {name: value.detach().clone() for name, value in parameters.items()}
        for count in range(1, self.maximum_backtracks+1):
            fraction = 2. ** -count
            with torch.no_grad():
                for name, value in parameters.items():
                    value.copy_(before[name] + fraction*(proposed[name]-before[name]))
            statistics['backtrack_checks'] = count
            if not self.stop_mismatches():
                # Keeping the new Adam moments with a shortened parameter
                # step is equivalent to reducing this update's learning rate.
                statistics['accepted_actor_step_fraction'] = fraction
                return statistics
        with torch.no_grad():
            for name, value in parameters.items():
                value.copy_(before[name])
        self.actor_optimizer.load_state_dict(optimizer_before)
        statistics.update(actor_updated=False, accepted_actor_step_fraction=0.,
            actor_optimizer_restored=True)
        assert not self.stop_mismatches()
        return statistics
