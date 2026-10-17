"""Run the unchanged fixed regression with explicitly selected fresh perception."""
import experiments.evaluate_quality_regression as evaluator
from experiments.fresh_perception_runtime import FreshPerceptionRuntime


if __name__ == '__main__':
    evaluator.QualitySensorNavigationRuntime = FreshPerceptionRuntime
    evaluator.main()
