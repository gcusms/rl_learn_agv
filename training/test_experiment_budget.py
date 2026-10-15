import unittest
from unittest.mock import patch
from neural_navigation.experiment_budget import ExperimentBudget


class BudgetTests(unittest.TestCase):
    def test_submillisecond_remainder_does_not_busy_wait(self):
        budget=ExperimentBudget(.25);budget.started=0.;budget.cpu_started=0.
        with patch('neural_navigation.experiment_budget.time.process_time',return_value=.01),patch('neural_navigation.experiment_budget.time.monotonic',return_value=.039),patch('neural_navigation.experiment_budget.time.sleep') as sleep:
            budget.tick();sleep.assert_not_called()

    def test_real_quota_debt_is_slept(self):
        budget=ExperimentBudget(.25);budget.started=0.;budget.cpu_started=0.
        with patch('neural_navigation.experiment_budget.time.process_time',return_value=.02),patch('neural_navigation.experiment_budget.time.monotonic',side_effect=[0.,.0801]),patch('neural_navigation.experiment_budget.time.sleep') as sleep:
            budget.tick();sleep.assert_called_once_with(.08)


if __name__=='__main__':unittest.main()
