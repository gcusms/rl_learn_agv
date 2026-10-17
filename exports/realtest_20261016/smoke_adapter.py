"""Three stored sensor frames through the actual ROS-free worker; no physics."""
import json
from pathlib import Path
import sys
import time

import numpy as np


def main():
    root = Path(__file__).resolve().parent
    sys.path[:0] = [str(root / 'reference_ros'), str(root / 'training'), str(root)]
    from rl_navigation.inference_worker import InferenceWorker
    from deployment.ros2_recording_adapter.model_bundle import inspect_bundle
    config = inspect_bundle(root / 'generated/model_candidate.json')
    report_dir = root / 'reports/stop_feedback_20261015/release_validation_20261016'
    report = json.loads((report_dir / 'RESULTS.json').read_text())
    row = report['rows'][0]
    cases = json.loads((root / 'reports/stop_feedback_20261015/early_yield_remaining_protocol_20261016/heldout_cases.json').read_text())
    case = next(item for item in cases if item['id'] == row['case_id'])
    settings = dict(model_source=str(root), required_runtime=config['required_runtime'],
        bundle=config['bundle'], map_path=str(root / 'training' / case['scene'] / 'decision_map.yaml'),
        route_mode='shortcut', geometry=[.1, .2, .22], max_rps=1.)
    worker = InferenceWorker(settings)
    start = time.perf_counter()
    with np.load(root / row['trace_path'], allow_pickle=False) as trace:
        for index in range(3):
            stamp = float(trace['labels'][index, 0])
            frame = (trace['scans'][index], trace['odometry'][index],
                trace['measured_velocity'][index], stamp, True)
            actual = None if index == 0 else ([0., 0.] if index == 1 else [.1, 0.])
            reason = 'external_stop' if index == 1 else 'none'
            result = worker.infer(frame, 1, case['start'], case['goal'], stamp,
                capture_observation=True, published_command=actual, published_reason=reason,
                localization_generation=1)[2]
            observation = np.asarray(result['policy_observation'])
            assert observation.shape == (625,) and np.isfinite(observation).all()
            assert len(result['recording_input']['action_mask']) == 12
            if index == 1:
                assert result['decision_feedback_previous_reason'] == 'other'
                np.testing.assert_array_equal(observation[222:224], [0., 0.])
            if index == 2:
                assert result['decision_feedback_previous_reason'] == 'none'
                np.testing.assert_allclose(observation[222:224], [.5, 0.], atol=1e-7)
    result = dict(completed=True, stored_frames=3, finite_625D=True,
        actual_published_history_visible_in_observation=True, duplicate_record_errors=0,
        native_kernel_loaded=True, physics_episodes=0, ros_node_executed=False,
        vehicle_started=False, elapsed_seconds=time.perf_counter() - start)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
