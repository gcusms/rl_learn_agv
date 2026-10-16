"""Inspect actual stopped observations without starting another simulation.

The same recorded observation/mask sequence is evaluated by several frozen
checkpoints. Cross-model choices are offline predictions, not navigation runs.
Same-checkpoint decisions are checked against each model's actual recorded
choices; no command is executed and no optimizer is updated.
"""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from experiments.proactive_recovery import obstacle_context
from experiments.stop_recovery_policy import StopRecoveryPPO, masks_tensor, usable
from experiments.train_prefix_recovery import frozen_base_digest
from neural_navigation.decision_feedback import FEEDBACK_REASONS, REASON_SLICE, STAT_SLICE
from neural_navigation.experiment_budget import ExperimentBudget


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'reports/stop_feedback_20261015'
MODELS = ROOT / 'models/stop_feedback_20261015'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantiles(values):
    values = np.asarray(values)
    if not values.size:
        return None
    return dict(zip(['minimum', 'p05', 'median', 'p95', 'maximum'],
                    map(float, np.quantile(values, [0, .05, .5, .95, 1]))))


def main():
    if (OUT / 'RESULTS.json').exists():
        raise ValueError('Do not overwrite a completed diagnostic')
    torch.set_num_threads(1)
    budget = ExperimentBudget(.05, memory_mib=2048, directory=OUT)
    inputs = {}
    traces = {}
    for name, folder, case_id in [
            ('rotation_option_t106', 'rotation_escape_t106_model_20261016', 'w11_training_stall_t106'),
            ('control_full_w11', 'w11_control2000_navigation_20261016', 'train_failure_w11_e7'),
            ('reference_full_w11', 'w11_recovery_bc800_navigation_20261016', 'train_failure_w11_e7')]:
        report_path = BASE / folder / 'RESULTS.json'
        report = json.loads(report_path.read_text())
        if not report['completed']:
            raise ValueError('Only completed physical reports are allowed')
        row, = [r for r in report['rows'] if r['case_id'] == case_id]
        path = BASE / folder / (case_id + ('' if row.get('recovery_test') else '_' + row['variant']) + '.npz')
        if digest(path) != row['trace_sha256']:
            raise ValueError('Physical trace changed')
        with np.load(path, allow_pickle=False) as z:
            actual = z['observations'].copy()
            if actual.shape[1] != 439:
                raise ValueError('Expected directional runtime 439-D observations')
            obs = np.concatenate([actual[:, :300], actual[:, 303:]], axis=1)
            masks = z['masks'].astype(bool)
            np.testing.assert_array_equal(masks_tensor(torch.from_numpy(obs)).numpy(), masks)
            issued_idle = np.max(abs(z['commands']), axis=-1) < 1e-8
            unfinished = np.linalg.norm(obs[:, 218:220] * 12., axis=-1) > .3
            selected = issued_idle & masks[:, 1:].any(-1) & unfinished
            post = z['model_controlled'].astype(bool) if 'model_controlled' in z.files else np.ones(len(obs), bool)
            selected &= post
            if name == 'rotation_option_t106':
                if post.sum() != 170 or selected.sum() != 170:
                    raise ValueError('The recorded 170-frame stopped continuation changed')
            stats = obs[:, 300+STAT_SLICE.start:300+STAT_SLICE.stop]
            reasons = obs[:, 300+REASON_SLICE.start:300+REASON_SLICE.stop].argmax(-1)
            source_summary = dict(frames=len(obs), selected_frames=int(selected.sum()),
                selection='model-controlled executed idle, motion permitted, estimated goal >0.3m',
                masks=dict(Counter(','.join(map(str, np.flatnonzero(m))) for m in masks[selected])),
                previous_reasons=dict(Counter(FEEDBACK_REASONS[r] for r in reasons[selected])),
                prior_legal_idle_seconds=quantiles(stats[selected, 3] * 5.),
                predictor_valid_frames=int((stats[selected, 4] > .5).sum()),
                scan_valid_frames=int((stats[selected, 7] > .5).sum()),
                reported_safe_success=row['safe_success'], reported_collisions=row['collisions'],
                reported_timeout=row['timeout'])
            traces[name] = dict(obs=obs, masks=masks, selected=selected,
                                choices=z['choices'].copy(), summary=source_summary)
        for f in [report_path, path]:
            inputs[str(f.relative_to(ROOT))] = digest(f)
        budget.tick()
    variants = dict(
        reference_bc=MODELS/'w11_recovery_bc800_s231015/policy_bc.zip',
        control2k=MODELS/'w11_control2000_s231015/policy.zip',
        retained2k=MODELS/'w11_retained2000_s231015/policy.zip',
        additional_bc=MODELS/'early_transfer_bc800_s231015/policy_bc.zip')
    results = {}; base_digests = []
    prediction_arrays = {}
    for variant, path in variants.items():
        model = StopRecoveryPPO.load(path, device='cpu')
        model.policy.set_training_mode(False)
        base_digests.append(frozen_base_digest(model.policy))
        inputs[str(path.relative_to(ROOT))] = digest(path)
        budget.tick()
        result = {}
        for name, trace in traces.items():
            selected = trace['selected']
            observations = trace['obs'][selected]
            masks = trace['masks'][selected]
            arrays = {k: [] for k in ['probabilities', 'support', 'usable', 'context',
                                      'baseline_choice', 'baseline_margin', 'learned_margin']}
            for first in range(0, len(observations), 125):
                x = torch.from_numpy(observations[first:first+125])
                m = masks[first:first+125]
                with torch.no_grad():
                    features = model.policy.mlp_extractor.policy_net(x)
                    baseline, support = model.policy.action_net.parts(features)
                    logits = model.policy.action_net(features)
                    probability = model.policy.get_distribution(x, m).distribution.probs
                    mask = torch.from_numpy(m)
                    baseline = baseline.masked_fill(~mask, -1e8)
                    learned = logits.masked_fill(~mask, -1e8)
                    values = dict(probabilities=probability.numpy(), support=support.numpy(),
                        usable=usable(x).numpy(), context=obstacle_context(x).numpy(),
                        baseline_choice=baseline.argmax(-1).numpy(),
                        baseline_margin=(baseline[:, 0]-baseline[:, 1:].amax(-1)).numpy(),
                        learned_margin=(learned[:, 0]-learned[:, 1:].amax(-1)).numpy())
                for key, value in values.items():
                    arrays[key].append(value)
                budget.tick()
            if not len(observations):
                result[name] = dict(selected_frames=0)
                continue
            arrays = {k: np.concatenate(v) for k, v in arrays.items()}
            probabilities = arrays['probabilities']
            if not np.isfinite(probabilities).all() or probabilities[~masks].any():
                raise ValueError('Invalid masked action distribution')
            choices = probabilities.argmax(-1)
            same_checkpoint = ((variant == 'reference_bc' and name in
                                ['rotation_option_t106', 'reference_full_w11'])
                               or (variant == 'control2k' and name == 'control_full_w11'))
            if same_checkpoint:
                np.testing.assert_array_equal(choices, trace['choices'][selected])
            entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-38))).sum(-1)
            result[name] = dict(selected_frames=len(observations),
                same_checkpoint_recorded_argmax_verified=same_checkpoint,
                learnable_support_frames=int(arrays['support'].sum()),
                usable_frames=int(arrays['usable'].sum()),
                obstacle_context_frames=int(arrays['context'].sum()),
                frozen_base_choices=dict(Counter(map(str, arrays['baseline_choice']))),
                predicted_choices=dict(Counter(map(str, choices))),
                stop_probability=quantiles(probabilities[:, 0]),
                # Summing motion entries preserves tiny probabilities even
                # when the float32 stop entry already rounds to exactly1.
                move_probability=quantiles(probabilities[:, 1:].sum(-1)),
                pure_right_turn_probability=quantiles(probabilities[:, 5]),
                creeping_right_turn_probability=quantiles(probabilities[:, 9]),
                entropy_nats=quantiles(entropy),
                frozen_base_stop_minus_best_motion_logit=quantiles(arrays['baseline_margin']),
                learned_stop_minus_best_motion_logit=quantiles(arrays['learned_margin']))
            prediction_arrays.update({variant+'__'+name+'__'+key: value for key, value in arrays.items()})
        results[variant] = result
        del model
        budget.tick()
    if len(set(base_digests)) != 1:
        raise ValueError('Frozen base differs between compared checkpoints')
    for filename in ['training/experiments/stop_recovery_policy.py',
                     'training/experiments/proactive_recovery.py',
                     'training/neural_navigation/decision_feedback.py']:
        inputs[filename] = digest(ROOT / filename)
    np.savez_compressed(OUT/'predictions.npz', **prediction_arrays)
    budget.tick()
    output = dict(completed=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        traces={name: t['summary'] for name, t in traces.items()}, predictions=results,
        inputs_sha256=inputs, source_sha256=digest(Path(__file__)),
        prediction_sha256=digest(OUT/'predictions.npz'),
        frozen_base_unchanged=True, frozen_base_sha256=base_digests[0],
        additional_environment_steps=0, optimizer_updates=0,
        current_runtime_or_checkpoint_changed=False, resource=budget.statistics(),
        limitations=['All inputs are reused development traces.',
            'Only explicitly marked same-checkpoint rows verify actual recorded decisions.',
            'Other checkpoints see another policy trajectory and possibly an altered mask; these are offline predictions, not independent navigation scores.',
            'A permitted motion need not be useful or safe for the complete mission.',
            'Probability saturation establishes an exploration obstacle, not a successful alternative maneuver.',
            'Motion mass is summed directly; float32(1-p_stop) can erase tiny positive probabilities. The initial audit and source are preserved separately.'],
        goal_achieved=False, deployment_approved=False)
    (OUT/'RESULTS.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(dict(completed=True,
        rotation={v: r['rotation_option_t106'] for v, r in results.items()},
        resource=output['resource']), indent=2), flush=True)


if __name__ == '__main__':
    main()
