"""Count actual long-wait training exposure, separately from initialization."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from neural_navigation.decision_feedback import STAT_SLICE
from neural_navigation.experiment_budget import ExperimentBudget


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
MODELS = ROOT / 'models/stop_feedback_20261015'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(observations, masks, labels, support=None):
    timer = observations[:, 300+STAT_SLICE.start+3]*5.
    available = masks[:, 1:].any(-1)
    if support is None:
        support = np.ones(len(labels), bool)
    groups = {}
    for name, selection in [
            ('all', np.ones(len(labels), bool)),
            ('prior_legal_idle_ge1s', timer >= 1.-1e-6),
            ('prior_legal_idle_ge2s', timer >= 2.-1e-6),
            ('prior_legal_idle_ge4s', timer >= 4.-1e-6)]:
        groups[name] = dict(frames=int(selection.sum()),
            supported_with_motion=int((selection & support & available).sum()),
            selected_stop=int((selection & (labels == 0)).sum()),
            selected_motion=int((selection & (labels != 0)).sum()),
            action_counts=dict(Counter(map(str, labels[selection]))))
    return dict(frames=len(labels), maximum_prior_idle_with_motion_s=float(timer.max()),
        groups=groups,
        labels_per_action=[int((labels == a).sum()) for a in range(12)],
        permitted_per_action=masks.sum(0).astype(int).tolist())


def main():
    destination = OUT / 'TRAINING_COVERAGE.json'
    if destination.exists():
        raise ValueError('Do not overwrite completed evidence')
    budget = ExperimentBudget(.05, memory_mib=1024, directory=OUT)
    inputs = {}; demonstrations = {}; physical = {}
    for name in ['w11_recovery_bc800_s231015', 'early_transfer_bc800_s231015']:
        p = MODELS/name/'demonstrations.npz'
        with np.load(p, allow_pickle=False) as z:
            demonstrations[name] = summarize(z['observations'], z['masks'].astype(bool), z['choices'])
        inputs[str(p.relative_to(ROOT))] = digest(p)
        budget.tick()
    for variant in ['control', 'retained']:
        folder = MODELS/('w11_'+variant+'2000_s231015')
        status = json.loads((folder/'status.json').read_text())
        if status['status'] != 'complete' or status['additional_ppo_steps'] != 2000:
            raise ValueError('Completed actual 2k PPO sampling required')
        observations = []; masks = []; choices = []; supports = []; labels = []
        for path in sorted((folder/'rollouts').glob('*.npz')):
            with np.load(path, allow_pickle=False) as z:
                observations.append(z['observations'][:, 0].copy())
                masks.append(z['action_masks'][:, 0].astype(bool))
                choices.append(z['sampled_actions'][:, 0, 0].astype(int))
                records = [json.loads(str(s)) for s in z['execution_diagnostics_json'][:, 0]]
                labels.extend(records)
                supports.extend(r['recovery_actor_supported'] for r in records)
            inputs[str(path.relative_to(ROOT))] = digest(path)
            budget.tick()
        observations = np.concatenate(observations); masks = np.concatenate(masks)
        choices = np.concatenate(choices); supports = np.asarray(supports, bool)
        if len(choices) != 2000 or not np.array_equal(choices, [r['sampled_choice'] for r in labels]):
            raise ValueError('Actual recorded sampled-action count/order changed')
        summary = summarize(observations, masks, choices, supports)
        timer = observations[:, 300+STAT_SLICE.start+3]*5.
        eligible = supports & masks[:, 1:].any(-1) & (timer >= 2.-1e-6)
        source_rows = {}
        for i, label in enumerate(labels):
            if not eligible[i]:
                continue
            case = json.loads(label['case_json'])
            source = case.get('source_training_case_id') or case.get('scene', 'routine')
            bucket = source_rows.setdefault(source, dict(frames=0, selected_stop=0, selected_motion=0))
            bucket['frames'] += 1
            bucket['selected_stop' if choices[i] == 0 else 'selected_motion'] += 1
        summary.update(initialization_frames_excluded=status['initialization_frames'],
            long_wait_training_loss_enabled=bool(status.get('long_wait_exploration')),
            temporally_eligible_supported_frames_by_source=source_rows,
            probability_threshold_not_evaluated_here=True)
        physical[variant] = summary
        inputs[str((folder/'status.json').relative_to(ROOT))] = digest(folder/'status.json')
    output = dict(completed=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        demonstrations=demonstrations, actual_ppo_samples=physical, inputs_sha256=inputs,
        source_sha256=digest(Path(__file__)),
        additional_environment_steps=0, optimizer_updates=0,
        physical_initialization_not_counted_as_training=True,
        limitations=['Demonstration labels and on-policy sampled actions are different kinds of evidence.',
            'Supported waits of at least2s are only the temporal part of the optional exploration-loss gate; stop probability>=0.95 would further restrict it.',
            'No assumption is made that any permitted movement would complete the task.',
            'No actual late-prefix training or physical continuation is performed by this audit.'],
        resource=budget.statistics(), goal_achieved=False, deployment_approved=False)
    destination.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(dict(completed=True,
        demonstration_long_wait_frames={k:v['groups']['prior_legal_idle_ge2s']['frames'] for k,v in demonstrations.items()},
        ppo_long_wait_supported={k:v['groups']['prior_legal_idle_ge2s']['supported_with_motion'] for k,v in physical.items()},
        resource=output['resource']), indent=2), flush=True)


if __name__ == '__main__':
    main()
