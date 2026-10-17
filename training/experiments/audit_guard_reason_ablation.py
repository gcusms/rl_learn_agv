"""Read-only audit of learned choices; no counterfactual is executed or labeled."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from experiments.context_release_actor import ContextReleaseActor
from experiments.guard_reason_release_actor import GuardReasonReleaseActor
from experiments.long_risk_actor import digest
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS, masks_from_observation


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'


def counts(values):
    values, numbers = np.unique(values, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(values, numbers)}


def main(args):
    output, model, control = [getattr(args, k).resolve() for k in ['output', 'model', 'control']]
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.03, memory_mib=1024, directory=output)
    paths = [
        BASE / 'context_broad_diagnostic_20261016/physical/hard3_w11_training_stall_t106.npz',
        BASE / 'context_broad_diagnostic_20261016/physical/hard3_legal_wait_parent_t95.npz',
        BASE / 'context_broad_diagnostic_20261016/physical/hard3_legal_wait_parent_t90.npz',
        BASE / 'context_physics_diagnostic_20261016/physical/w11_training_stall_t95.npz',
        BASE / 'context_physics_diagnostic_20261016/physical/w11_training_stall_t90.npz',
        BASE / 'context_physics_diagnostic_20261016/physical/w11_actual95_wait2_4.npz']
    fitted = json.loads((model / 'RESULTS.json').read_text())
    assert fitted['completed'] and digest(model / 'checkpoint.pt') == fitted['checkpoint_sha256']
    actor = GuardReasonReleaseActor(torch.jit.load(str(model / 'parent_actor.pt'), map_location='cpu')).eval()
    actor.load_state_dict(torch.load(model / 'checkpoint.pt', map_location='cpu', weights_only=False)['actor'])
    previous = ContextReleaseActor(torch.jit.load(str(control / 'parent_actor.pt'), map_location='cpu')).eval()
    previous.load_state_dict(torch.load(control / 'checkpoint.pt', map_location='cpu', weights_only=False)['actor'])
    inputs = [Path(__file__), model / 'checkpoint.pt', model / 'RESULTS.json',
        control / 'checkpoint.pt', *paths,
        ROOT / 'training/experiments/guard_reason_release_actor.py']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    rows = []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            controlled = z['model_controlled'].astype(bool)
            observations = z['observations'][controlled].copy()
            masks = z['masks'][controlled].copy()
        np.testing.assert_array_equal(masks_from_observation(observations[:, :282]), masks)
        assert np.all(observations[:, 303:399].reshape(-1, 12, 8).sum(2) == 1.)
        x = torch.from_numpy(observations)
        with torch.no_grad():
            features, baseline = actor.features(x)
            support = actor.support(x)
            predictions, before = actor(x).numpy(), previous(x).numpy()
            logits = actor.logits(x).masked_fill(~torch.from_numpy(masks), -1e8)
            probability = logits.softmax(1)
            # Disable just the new projection's input; frozen parent, masks,
            # history and old312features stay fixed. This is an offline shadow.
            shadow_features = features.clone(); shadow_features[:, 312:] = 0.
            shadow_logits = (actor.head(shadow_features) + 2. * features[:, 76:88]).masked_fill(
                ~torch.from_numpy(masks), -1e8)
            shadow_choice = shadow_logits.argmax(1)
            shadow_action = torch.where(support[:, None], actor.actions[shadow_choice], baseline).numpy()
        def choices(actions):
            distances = np.abs(actions[:, None] - ACTIONS[None]).max(2)
            result = distances.argmin(1)
            assert (distances.min(1) < 2e-6).all()
            assert masks[np.arange(len(result)), result].all()
            return result
        chosen, old, shadow = [choices(a) for a in [predictions, before, shadow_action]]
        legal_motion = masks[:, 1:].any(1)
        rows.append(dict(trace=str(path.relative_to(ROOT)), frames=len(x),
            active_frames=int(support.sum()), legal_motion_frames=int(legal_motion.sum()),
            only_stop_frames=int((~legal_motion).sum()), control_choices=counts(old),
            candidate_choices=counts(chosen), offline_new_reason_disabled_choices=counts(shadow),
            old_legal_wait_frames=int(((old == 0) & legal_motion).sum()),
            candidate_legal_wait_frames=int(((chosen == 0) & legal_motion).sum()),
            changed_decisions=int((old != chosen).sum()),
            head_only_reason_ablation_changed_decisions=int((shadow != chosen).sum()),
            candidate_minimum_masked_stop_probability=float(probability[:, 0].min()),
            all_decisions_currently_legal=True, no_actions_executed=True))
        budget.tick()
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected
    budget.tick()
    report = dict(completed=True, at=datetime.now(timezone.utc).isoformat(), inputs=hashes,
        candidate_source_fit_passed=fitted['exported_for_physical_validation'], rows=rows,
        no_new_physics_or_labels_or_training=True, deployment_approved=False,
        limitations=['Predictions on recorded inputs do not establish a safe closed-loop recovery.',
            'A changed shadow decision is not a valid training label or motion authorization.',
            'Failed late states still need complete physical continuation evidence.'],
        resource=budget.statistics())
    (output / 'RESULTS.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['model', 'control', 'output']:
        parser.add_argument('--' + key, type=Path, required=True)
    main(parser.parse_args())
