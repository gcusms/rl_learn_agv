"""Check wait-conditioned sampling without an optimizer or physical step."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from experiments.masked_navigation import PortableMaskedActor
from experiments.stop_recovery_policy import StopRecoveryPPO
from experiments.train_prefix_recovery import frozen_base_digest
from experiments.wait_calibrated_recovery import (
    WaitCalibratedRecoveryPolicy, LEGAL_WAIT_INDEX, SCHEMA,
)
from neural_navigation.masked_actions import ACTIONS


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'reports/stop_feedback_20261015'
PARENT = ROOT / 'models/stop_feedback_20261015/early_transfer_bc800_s231015'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    if (OUT / 'RESULTS.json').exists():
        raise FileExistsError('Do not overwrite a completed or failed audit')
    torch.set_num_threads(1)
    paths = [PARENT / 'policy_bc.zip', PARENT / 'demonstrations.npz', Path(__file__),
        ROOT / 'training/experiments/wait_calibrated_recovery.py',
        ROOT / 'training/experiments/proactive_recovery.py',
        ROOT / 'training/experiments/stop_recovery_policy.py',
        BASE / 'early_transfer_verified_w11_20261016/w11_training_stall_t106.npz',
        BASE / 'actual_t106_action9_probe_20261016/actual_t106_action9_4frames.npz']
    inputs = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    old = StopRecoveryPPO.load(PARENT / 'policy_bc.zip', device='cpu')
    model = StopRecoveryPPO.load(PARENT / 'policy_bc.zip', device='cpu',
        custom_objects={'policy_class': WaitCalibratedRecoveryPolicy})
    old.policy.set_training_mode(False)
    model.policy.set_training_mode(False)
    old_state, new_state = old.policy.state_dict(), model.policy.state_dict()
    assert old_state.keys() == new_state.keys()
    assert all(torch.equal(v, new_state[k]) for k, v in old_state.items())
    old_optimizer, new_optimizer = old.policy.optimizer.state_dict(), model.policy.optimizer.state_dict()
    assert old_optimizer['param_groups'] == new_optimizer['param_groups']
    for index, history in old_optimizer['state'].items():
        assert history.keys() == new_optimizer['state'][index].keys()
        for k, v in history.items():
            other = new_optimizer['state'][index][k]
            assert torch.equal(v, other) if torch.is_tensor(v) else v == other
    frozen = frozen_base_digest(model.policy)
    assert frozen == frozen_base_digest(old.policy)
    assert frozen == 'e5502406c4ee8e42c4da09dd6affa022cb76e7f9b01316b1eaf0d2551431938e'
    with np.load(PARENT / 'demonstrations.npz', allow_pickle=False) as z:
        demo_x = z['observations'].copy()
        demo_masks = z['masks'].copy()
        demo_choices = z['choices'].copy()
    batches = [demo_x]
    masks = [demo_masks]
    provenance = []
    for path in paths[-2:]:
        with np.load(path, allow_pickle=False) as z:
            keep = z['model_controlled'].astype(bool)
            raw = z['observations'][keep]
            observations = np.c_[raw[:, :300], raw[:, 303:]].astype(np.float32)
            batches.append(observations)
            masks.append(z['masks'][keep].astype(bool))
            provenance.append(dict(path=str(path.relative_to(ROOT)), actual_frames=int(keep.sum())))
    x = torch.from_numpy(np.concatenate(batches))
    mask = np.concatenate(masks)
    with torch.no_grad():
        old_dist = old.policy.get_distribution(x, action_masks=mask).distribution
        new_dist = model.policy.get_distribution(x, action_masks=mask).distribution
        old_probs, probs = old_dist.probs, new_dist.probs
        features = model.policy.mlp_extractor.policy_net(x)
        base, active = model.policy.action_net.parts(features)
        temperature = model.policy.action_net.temperature(features, base, active)
        np.testing.assert_array_equal(old_probs.argmax(-1), probs.argmax(-1))
        np.testing.assert_array_equal(old.policy.recovery_support(x), model.policy.recovery_support(x))
        unchanged = temperature == 1.
        torch.testing.assert_close(old_probs[unchanged], probs[unchanged], rtol=0, atol=0)
        stop_labels = demo_choices == 0
        assert int(stop_labels.sum()) == 55
        assert bool(unchanged[:len(demo_x)][stop_labels].all())
        torch.testing.assert_close(old_probs[:len(demo_x)][stop_labels],
            probs[:len(demo_x)][stop_labels], rtol=0, atol=0)
        only_stop = ~mask[:, 1:].any(-1)
        assert only_stop.any()
        np.testing.assert_array_equal(probs[only_stop, 0], np.ones(int(only_stop.sum())))
        np.testing.assert_array_equal(probs.numpy()[~mask], np.zeros(int((~mask).sum())))
        t106_begin = len(demo_x)
        eligible = (active & (x[:, LEGAL_WAIT_INDEX] * 5. >= 2.)
            & torch.from_numpy(mask[:, 9]) & (torch.from_numpy(mask).sum(-1) == 2))
        selected = torch.nonzero(eligible & (torch.arange(len(x)) >= t106_begin), as_tuple=False).flatten()[0]
        probe = x[selected:selected + 1]
        probe_mask = mask[int(selected):int(selected) + 1]
        before = float(old_probs[selected, 9])
        after = float(probs[selected, 9])
        assert before < 1e-6 and .02 < after < .2
        torch.manual_seed(231016)
        repeated = probe.repeat(1024, 1)
        repeated_mask = np.repeat(probe_mask, 1024, axis=0)
        actions, values, old_logp = model.policy(repeated, deterministic=False, action_masks=repeated_mask)
        reevaluated_values, logp, entropy = model.policy.evaluate_actions(repeated, actions,
            action_masks=repeated_mask)
        torch.testing.assert_close(logp, old_logp, rtol=0, atol=1e-6)
        torch.testing.assert_close(reevaluated_values, values, rtol=0, atol=1e-6)
        assert torch.isfinite(logp).all() and torch.isfinite(entropy).all()
        draws = np.bincount(actions.numpy(), minlength=12)
        assert 20 <= draws[9] <= 200 and draws[0] > draws[9]
        assert set(actions.tolist()) == {0, 9}
        portable = PortableMaskedActor(model.policy).eval()
        scripted = torch.jit.trace(portable, x[:128])
        expected = ACTIONS[probs.argmax(-1).numpy()]
        np.testing.assert_array_equal(scripted(x).numpy(), expected)
        scripted.save(str(OUT / 'calibrated_navigation_actor.pt'))
    model.recovery_contract = model.policy.recovery_contract()
    model.exploration_schema = SCHEMA
    model.exploration_contract = model.recovery_contract
    model.feedback_migration = dict(model.feedback_migration,
        recovery_contract=model.recovery_contract,
        calibration_parent_checkpoint_sha256=digest(PARENT / 'policy_bc.zip'),
        calibration_changed_weights=False, calibration_new_ppo_steps=0,
        calibration_exact_rollout_resume=False)
    checkpoint = OUT / 'policy_calibrated_initialization.zip'
    model.save(checkpoint)
    loaded = StopRecoveryPPO.load(checkpoint, device='cpu')
    assert isinstance(loaded.policy, WaitCalibratedRecoveryPolicy)
    assert loaded.recovery_contract == loaded.policy.recovery_contract()
    with torch.no_grad():
        restored = loaded.policy.get_distribution(probe, action_masks=probe_mask).distribution.probs
        torch.testing.assert_close(restored, probs[selected:selected + 1], rtol=0, atol=0)
    report = dict(completed=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        schema=SCHEMA, inputs=inputs, original_actual_frame_sources=provenance,
        checked_input_frames=len(x), calibrated_frames=int((temperature > 1).sum()),
        greedy_choices_unchanged=True, recovery_support_unchanged=True,
        unchanged_temperature_frames=int(unchanged.sum()), necessary_stop_labels_preserved=55,
        only_stop_and_illegal_probability_checks_passed=True,
        frozen_base_sha256=frozen, all_weights_and_adam_states_identical=True,
        actual_stall_motion_probability_before=before, actual_stall_motion_probability_after=after,
        actual_stall_observed_legal_wait_s=float(probe[0, LEGAL_WAIT_INDEX] * 5.),
        sampling_counts_1024=draws.tolist(), sampled_and_reevaluated_log_probability_parity=True,
        portable_greedy_parity_frames=len(x), saved_checkpoint_round_trip_passed=True,
        candidate_initialization_checkpoint=str(checkpoint.relative_to(ROOT)),
        candidate_initialization_sha256=digest(checkpoint), new_physical_steps=0,
        new_optimizer_updates=0, new_ppo_steps=0,
        admitted_for_physical_experiment=False, admitted_for_ppo=False,
        unproven_requirements=['A complete safe escape from the actual stalled state.',
            'Learned deterministic recovery after training.',
            'Full normal-navigation, necessary-wait, avoidance and docking regression.'],
        deployment_approved=False, goal_achieved=False)
    (OUT / 'RESULTS.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'inputs'}, indent=2), flush=True)


if __name__ == '__main__':
    try:
        audit()
    except Exception as error:
        (OUT / 'FAILURE.json').write_text(json.dumps(dict(error=repr(error),
            goal_achieved=False, new_ppo_steps=0), indent=2) + '\n')
        raise
