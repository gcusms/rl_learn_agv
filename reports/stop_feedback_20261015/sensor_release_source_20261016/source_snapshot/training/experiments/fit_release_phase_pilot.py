"""Two bounded, paired supervised fits; offline probes never count as navigation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.train_release_actor import main as train
from experiments.release_actor import ReleaseActor
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.35, memory_mib=1536, directory=output)
    dataset = BASE / 'release_recovery_sources_v2_20261016/demonstrations.npz'
    result_path = dataset.parent / 'RESULTS.json'
    sources = json.loads(result_path.read_text())
    assert sources['completed'] and digest(dataset) == sources['dataset_sha256']
    risk = ROOT / 'models/stop_feedback_20261015/long_risk_bc1000_s231016'
    current = ROOT / 'models/stop_feedback_20261015/release_context_bc2000_s231016'
    validation = BASE / 'release_validation_20261016/RESULTS.json'
    row = next(r for r in json.loads(validation.read_text())['rows'] if r['case_id'] == 'long_risk_sensor_05_1')
    trace_path = ROOT / row['trace_path']
    assert digest(trace_path) == row['trace_sha256']
    files = [Path(__file__), dataset, result_path, validation, trace_path,
        current / 'checkpoint.pt', risk / 'checkpoint.pt', risk / 'parent_actor.pt',
        ROOT / 'training/experiments/train_release_actor.py',
        ROOT / 'training/experiments/release_fit_weights.py', ROOT / 'training/experiments/release_actor.py']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
    names = {'stop_move': 'release_data_control300_s231016', 'recovery_phase': 'release_phase300_s231016'}
    write(output / 'PLAN.json', dict(inputs=hashes, seed=231016, maximum_updates_per_model=300,
        models=names, common_data_frames=sources['frames'], common_risk_initialization=True,
        changed_factor_between_new_models='loss allocation and duplicate weighting only',
        old_candidate_retained=True, original_frozen_four_are_development=True,
        offline_probes_cannot_prove_safe_counterfactual_execution=True,
        new_ppo_steps=0, new_physics_episodes=0, deployment_approved=False))
    with np.load(trace_path, allow_pickle=False) as archive:
        observations, recorded, masks = (archive[k].copy() for k in ['observations', 'choices', 'masks'])
    x = torch.from_numpy(observations)
    rows = []
    for mode, name in names.items():
        destination = current.parent / name
        print(json.dumps(dict(status='fitting', mode=mode, output=str(destination))), flush=True)
        train(argparse.Namespace(dataset=dataset, risk_model=risk, output=destination,
            seed=231016, maximum_updates=300, loss_mode=mode,
            recovery_motion_share=.5, recovery_wait_seconds=2.))
        fit = json.loads((destination / 'RESULTS.json').read_text())
        summary = dict(mode=mode, model=str(destination.relative_to(ROOT)), fit=fit)
        if fit.get('exported_for_physical_validation'):
            actor = ReleaseActor(torch.jit.load(str(destination / 'parent_actor.pt'), map_location='cpu')).eval()
            actor.load_state_dict(torch.load(destination / 'checkpoint.pt', map_location='cpu', weights_only=False)['actor'])
            portable = torch.jit.load(str(destination / 'portable/actor.pt'), map_location='cpu').eval()
            with torch.no_grad():
                command = actor(x).numpy()
                np.testing.assert_array_equal(command, portable(x).numpy())
                pstop = actor.logits(x).masked_fill(~actor.masks(x), -1e8).softmax(1)[:, 0].numpy()
            choices = np.abs(command[:, None] - ACTIONS[None]).max(2).argmin(1)
            wait = (recorded == 0) & masks[:, 1:].any(1)
            summary['diagnostic_off_policy_probe'] = dict(frames=len(x),
                eager_portable_choices_identical=True,
                changed_choices=int((choices != recorded).sum()),
                recorded_waits_changed_to_motion=int((wait & (choices != 0)).sum()),
                stop_probability_on_recorded_wait_p50=float(np.median(pstop[wait])),
                no_counterfactual_execution=True)
        rows.append(summary)
        budget.tick()
        write(output / 'STATUS.json', dict(status='paired_fits', completed_models=len(rows), rows=rows))
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    write(output / 'RESULTS.json', dict(completed=True, at=datetime.now(timezone.utc).isoformat(), rows=rows,
        resource=budget.statistics(), new_ppo_steps=0, new_physics_episodes=0,
        old_candidate_unchanged=True, goal_achieved=False, deployment_approved=False))
    write(output / 'STATUS.json', dict(status='complete', completed_models=len(rows)))
    (output / 'driver_snapshot.py').write_text(Path(__file__).read_text())
    print(json.dumps(dict(completed=True, models=len(rows), report=str(output)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
