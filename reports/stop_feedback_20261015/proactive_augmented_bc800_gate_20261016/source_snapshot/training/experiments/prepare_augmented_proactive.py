"""Review every frozen training probe and select complete safe BC examples.

The unsuccessful probe stays in the source report and in the selection audit.
Selection uses physical outcomes; truth and offline intervention times never
become policy inputs. Existing normal navigation and waiting labels are kept.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.prepare_proactive_recovery import CHECKPOINT, Interface, PROJECT
from experiments.proactive_recovery import ProactiveRecoveryPolicy, contract
from experiments.recovery_demonstrations import convert_observations, load_demonstrations
from experiments.stop_recovery_policy import warm_start_recovery
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import COMMANDS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-cores', type=float, default=.35)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(args.cpu_cores, memory_mib=3072, directory=output)
    original = PROJECT/'reports/stop_feedback_20261015/proactive_protocol_20261016'
    source = PROJECT/'reports/stop_feedback_20261015/proactive_training_probes_20261016'
    report_path = source/'RESULTS.json'
    report = json.loads(report_path.read_text())
    cases = json.loads((original/'ppo_training_cases.json').read_text())
    expected = [c for c in cases if c['id'].startswith('train_failure_proactive_variation_')]
    if (not report['completed'] or len(expected)!=8 or len(report['rows'])!=8
            or len({r['case_id'] for r in report['rows']})!=8
            or {r['case_id'] for r in report['rows']}!={c['id'] for c in expected}
            or json.loads((source/'cases.json').read_text())!=expected
            or report['case_source_sha256']!=digest(original/'ppo_training_cases.json')
            or not report['not_a_trained_policy']):
        raise ValueError('All eight predeclared training probes, including failures, are required')
    declared = dict(windows=[dict(start_s=6.,end_s=7.4,choice=5),
        dict(start_s=20.,end_s=22.,choice=1)])
    if report['declaration']!=declared:
        raise ValueError('The offline collection recipe changed')
    model = warm_start_recovery(CHECKPOINT, Interface(), seed=231015,
        policy_class=ProactiveRecoveryPolicy)
    specs = json.loads((original/'demonstrations.json').read_text())
    old_obs, old_masks, old_choices, _ = load_demonstrations(specs, model.policy)
    if len(old_obs)!=1058:
        raise ValueError('Expected the unchanged original retention examples')
    decisions = []
    for row in report['rows']:
        identity = row['case_id']
        path = source/(identity+'.npz')
        if (row['specification']!=dict(id=identity,**declared)
                or digest(path)!=row['trace_sha256']):
            raise ValueError('Probe declaration or trace changed: '+identity)
        with np.load(path,allow_pickle=False) as data:
            masks = data['masks'].astype(bool)
            choices = data['choices'].astype(np.int64)
            if not masks[np.arange(len(choices)),choices].all():
                raise ValueError('Illegal source choice: '+identity)
            observations = convert_observations(data['observations'],masks,
                data['commands'],data['labels'][:,0])
            away = np.linalg.norm(observations[:,218:220]*12.,axis=-1)>.3
            if not np.allclose(data['commands'][away],COMMANDS[choices[away]],rtol=0,atol=2e-8):
                raise ValueError('Source command changed after selection: '+identity)
        safe = bool(row['safe_success'] and row['physical_arrival'] and not (
            row['false_arrival'] or row['collisions'] or row['physical_guards'] or row['timeout']))
        decision = dict(id=identity,selected=safe,trace_sha256=digest(path),
            safe_success=bool(row['safe_success']),physical_arrival=bool(row['physical_arrival']),
            collisions=row['collisions'],timeout=bool(row['timeout']),
            reason='complete_physical_safe_episode' if safe else 'failed_episode_retained_not_imitated')
        decisions.append(decision)
        if safe:
            specs.append(dict(report=str(report_path.relative_to(PROJECT)),id=identity))
        budget.tick()
    obs,masks,choices,sources = load_demonstrations(specs,model.policy)
    if (not np.array_equal(obs[:len(old_obs)],old_obs)
            or not np.array_equal(masks[:len(old_masks)],old_masks)
            or not np.array_equal(choices[:len(old_choices)],old_choices)):
        raise ValueError('Original movement or appropriate-wait examples were changed')
    write(output/'demonstrations.json',specs)
    write(output/'DATA.json',dict(completed=True,created_at=datetime.now().astimezone().isoformat(),
        original_frames=len(old_obs),new_frames=len(obs)-len(old_obs),total_frames=len(obs),
        source_report=str(report_path.relative_to(PROJECT)),source_report_sha256=digest(report_path),
        case_source_sha256=digest(original/'ppo_training_cases.json'),
        selection=decisions,selected_complete_episodes=sum(d['selected'] for d in decisions),
        excluded_failed_episodes=sum(not d['selected'] for d in decisions),
        all_failed_probe_records_retained=True,original_examples_preserved_exactly=True,
        action_counts={str(k):v for k,v in Counter(choices.tolist()).items()},source_episodes=sources,
        policy_contract=contract(),no_validation_conditions_used_as_examples=True,
        no_truth_or_offline_intervention_clock_as_policy_input=True,
        original_body_masks_guard_speed_reward_unchanged=True,resource=budget.statistics(),
        deployment_approved=False))
    write(output/'PLAN.json',dict(frozen_at=datetime.now().astimezone().isoformat(),
        preceding_goal_turn_classification='progress',
        preceding_evidence='Uploaded immutable transfer branch 0f32475; six checkpoints, seven native bundles and 108 checks passed; remote README and BC checkpoint verified byte-identical.',
        hypothesis='Additional complete training conditions can teach both early waiting and avoidance instead of fitting one seven-action turn.',
        observed_limit='Current proactive BC is 14/14 on reused development but 1/4 on full-reset different conditions, and w11 t90 still times out.',
        supervised_initialization_only=True,additional_ppo_steps=0,seed=231015,
        baseline='models/stop_feedback_20261015/proactive_bc_s231015_prepared',
        unchanged_architecture=True,maximum_initialization_updates=400,
        required_before_ppo=['Inspect fitted actions and probability margins separately for each source, movement and valid waiting.',
            'All original 14 full tasks, including the six directional tasks.',
            'All four different-condition full-reset tasks, including any regressed original success.',
            'Both original full prefix continuations and the actual w11 prefix family.',
            'Retain failed results. A safe scripted example is not a learned-policy success.'],
        original_deployment_not_replaced=True,deployment_approved=False))
    print(json.dumps(dict(output=str(output),total_frames=len(obs),selected=sum(d['selected'] for d in decisions),
        excluded=sum(not d['selected'] for d in decisions),action_counts=Counter(choices.tolist())),indent=2),flush=True)


if __name__=='__main__':
    main()
