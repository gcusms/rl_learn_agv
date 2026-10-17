"""Verify the completed actual95 recovery against its saved learned actor."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from neural_navigation.masked_actions import ACTIONS, masks_from_observation


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'reports/stop_feedback_20261015'
MODEL = ROOT / 'models/stop_feedback_20261015/actual95_transfer_bc400_s231016'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (OUT / 'RESULTS.json').exists():
        raise FileExistsError('Do not overwrite the actual95 model conformance evidence')
    torch.set_num_threads(1)
    row = json.loads((OUT / 'candidate_row.json').read_text())
    parent = json.loads((OUT / 'parent_row.json').read_text())
    assert row['safe_success'] and row['physical_arrival'] and not row['false_arrival']
    assert not row['collisions'] and not row['physical_guards'] and not row['timeout']
    assert not parent['safe_success'] and parent['timeout']
    assert row['prefix_trace_sha256'] == parent['prefix_trace_sha256']
    trace = BASE / 'actual95_learned_validation_20261016/physical/w11_training_stall_t95.npz'
    assert digest(trace) == row['trace_sha256']
    manifest = json.loads((MODEL / 'bc_portable/manifest.json').read_text())
    for name, expected in manifest['files'].items():
        assert digest(MODEL / 'bc_portable' / name) == expected
    source = BASE / 'training_stall_source_20261016/train_failure_w11_e7.npz'
    assert digest(source) == row['prefix_trace_sha256']
    with np.load(trace, allow_pickle=False) as z:
        model_frames = z['model_controlled'].astype(bool)
        n = row['frozen_prefix_frames']
        assert n == 475
        np.testing.assert_array_equal(model_frames, np.arange(len(model_frames)) >= n)
        assert int(model_frames.sum()) == row['model_controlled_frames'] == 100
        x = z['observations'][model_frames].copy()
        choices = z['choices'][model_frames].copy()
        masks = z['masks'][model_frames].astype(bool)
        with np.load(source, allow_pickle=False) as s:
            for name in ['scans', 'odometry', 'measured_velocity']:
                np.testing.assert_allclose(z[name][:n], s[name][:n], rtol=0, atol=2e-8, equal_nan=True)
            np.testing.assert_allclose(z['labels'][:n + 1, 1:4], s['labels'][:n + 1, 1:4], rtol=0, atol=2e-8)
            np.testing.assert_array_equal(z['choices'][:n], s['choices'][:n])
    np.testing.assert_array_equal(masks_from_observation(x[:, :282]), masks)
    assert masks[np.arange(len(choices)), choices].all()
    actor = torch.jit.load(str(MODEL / 'bc_portable/actor.pt'), map_location='cpu').eval()
    with torch.no_grad():
        predicted = actor(torch.from_numpy(x)).numpy()
    np.testing.assert_array_equal(predicted, ACTIONS[choices])
    inputs = {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__),
        OUT / 'candidate_row.json', OUT / 'parent_row.json', trace, source,
        MODEL / 'policy_bc.zip', MODEL / 'bc_portable/actor.pt', MODEL / 'bc_portable/manifest.json']}
    result = dict(completed=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), inputs=inputs,
        actual_failure_prefix_physical_and_sensor_parity_verified=True,
        physical_prefix_frames=n, learned_control_frames=100,
        portable_actor_exactly_reproduces_all_post_release_actions=True,
        no_post_release_timed_or_expert_action_substitution=True,
        all_post_release_chosen_actions_legal=True,
        parent_timeout=True, candidate_safe_success=True,
        candidate_post_release_seconds=row['post_release_seconds'],
        parent_legal_wait_frames=parent['post_release_wait_with_choices_frames'],
        candidate_legal_wait_frames=row['post_release_wait_with_choices_frames'],
        physical_arrival_xy_error_m=row['final_xy_error_m'],
        no_new_physical_or_ppo_steps=True, current_original18_and_fresh12_acceptance_completed=False,
        actual106_escape_not_proven=True, matched_training_control_not_yet_finished=True,
        scope='One reused actual95 development recovery. Not unseen-map, complete-mission or real-robot acceptance.',
        deployment_approved=False, goal_achieved=False)
    (OUT / 'RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'inputs'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
