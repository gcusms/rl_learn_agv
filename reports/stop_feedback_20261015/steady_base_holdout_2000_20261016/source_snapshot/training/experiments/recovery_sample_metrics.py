"""Read-only counts of actual recovery decisions; no policy or RNG operations."""
from collections import Counter


class RecoverySampleMetrics:
    def __init__(self):
        self.decisions = 0
        self.recovery = Counter()
        self.reasons = Counter()
        self.wait_run = 0
        self.longest_wait_run = 0
        self.probability_sums = Counter()

    def record(self, observation, mask, action, probabilities):
        if len(observation) != 300 or len(mask) != 12 or len(probabilities) != 12:
            raise ValueError('Expected native 300-field observation and 12 choices')
        if not mask[action]:
            raise ValueError('Sampled action is outside its recorded native mask')
        self.decisions += 1
        alternatives = any(v > .5 for v in observation[235:244] + observation[267:269])
        behind = (observation[294] > .5 and observation[229] >= .5
                  and observation[232] < -.2
                  and sum(v*v for v in observation[216:218]) > .025**2)
        non_goal = observation[264] < .5
        support = alternatives and (behind or (non_goal and
                   (observation[250] < .5 or observation[266] >= .04)))
        if not support:
            self.wait_run = 0
            return
        self.recovery['decisions'] += 1
        self.recovery['wait' if action == 0 else 'pure_turn' if action in (4, 5)
                      else 'translation'] += 1
        self.recovery[f'action_{action}'] += 1
        reason = max(range(16), key=lambda i: observation[250+i])
        self.reasons[str(reason)] += 1
        self.probability_sums['wait'] += probabilities[0]
        self.probability_sums['left_turn'] += probabilities[4]
        self.probability_sums['right_turn'] += probabilities[5]
        self.wait_run = self.wait_run + 1 if action == 0 else 0
        self.longest_wait_run = max(self.longest_wait_run, self.wait_run)

    def end_episode(self):
        self.wait_run = 0

    def summary(self):
        count = self.recovery['decisions']
        return dict(total_decisions=self.decisions, recovery_counts=dict(self.recovery),
                    previous_reason_index_counts=dict(self.reasons),
                    longest_recovery_wait_run=self.longest_wait_run,
                    mean_recovery_probabilities={k: v/count for k, v in self.probability_sums.items()} if count else {},
                    limitation='Actual training samples only; action masks and exploration frequency do not prove sustained safe recovery.')
