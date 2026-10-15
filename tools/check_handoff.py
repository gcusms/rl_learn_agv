"""Check a fresh checkout without ROS, a display, a GPU or motor publishers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models/localization_mapping_20261015/ray_consistent_candidate/portable'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--build-native', action='store_true', help='Also rebuild the optional native parity reference')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    try:
        os.nice(19)
    except (AttributeError, OSError):
        pass
    sys.path.insert(0, str(ROOT / 'training'))
    sys.path.insert(0, str(ROOT))
    os.environ['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'training'), str(ROOT)])
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from neural_navigation.goal_pose_runtime import check_files, resolve_runtime

    pinned = json.loads((ROOT / 'docs/handoff/FILES.json').read_text())['files']
    for name, expected in pinned.items():
        if digest(ROOT / name) != expected:
            raise ValueError('Handoff file mismatch: ' + name)
    manifest = json.loads((MODEL / 'manifest.json').read_text())
    check_files(MODEL, manifest)
    card = json.loads((MODEL / 'MODEL_CARD.json').read_text())
    if digest(MODEL / 'manifest.json') != card['manifest_sha256']:
        raise ValueError('Model card manifest mismatch')
    for name, expected in card['runtime_sources_sha256'].items():
        if digest(ROOT / 'training/neural_navigation' / name) != expected:
            raise ValueError('Candidate runtime mismatch: ' + name)

    frozen = ROOT / 'training/assets/localization_replay_20261015'
    tasks = json.loads((frozen / 'TASKS.json').read_text())
    if len(tasks) != 28:
        raise ValueError('Missing frozen simulation tasks')
    for case in tasks:
        if not (ROOT / 'training' / case['scene'] / 'decision_map.yaml').is_file():
            raise ValueError('Missing physical map: ' + case['scene'])
        with np.load(frozen / (case['id'] + '.npz')) as data:
            if data['scan'].shape[1] != 220:
                raise ValueError('Scan contract mismatch')
    real = ROOT / 'training/assets/real_recording_20261014'
    with (real / 'record/decisions.jsonl').open() as stream:
        decisions = [json.loads(line)['data'] for line in stream]
    if len(decisions) != 1516 or len({r['mission_id'] for r in decisions}) != 22:
        raise ValueError('Real recording partition mismatch')

    data = np.load(frozen / 'short_00.npz')
    case = tasks[0]
    initial = data['prior']
    goal = [*initial[:2], float(initial[2] + .5)]
    runtime_type = resolve_runtime(manifest['required_runtime'])
    if runtime_type is None:
        raise ValueError('Unknown runtime contract')
    runtime = runtime_type(MODEL, ROOT / 'training' / case['scene'] / 'decision_map.yaml',
                           initial, goal, route_mode='shortcut')
    result = runtime.step(data['scan'][0], data['odom'][0], data['velocity'][0], float(data['t'][0]))
    if result['policy_observation_dimensions'] != 303 or not np.isfinite(result['command']).all():
        raise ValueError('Directional runtime smoke check failed')
    if abs(runtime.core.goal_yaw - goal[2]) > 1e-9:
        raise ValueError('Directional goal lost')

    from neural_navigation.decision_feedback import contract as feedback_contract
    from experiments.stop_feedback_ppo import StopFeedbackPPO
    feedback_models = []
    for folder, schema in [('ppo_4000_s221015', 'stop_decision_feedback_v1'),
                           ('ppo_v2_4000_s221015', 'stop_decision_feedback_v2'),
                           ('ppo_explore_4000_s221015', 'stop_decision_feedback_v2')]:
        feedback = ROOT / 'models/stop_feedback_20261015' / folder
        feedback_bundle = feedback / 'portable'
        feedback_manifest = json.loads((feedback_bundle / 'manifest.json').read_text())
        if feedback_manifest['observation_contract'] != feedback_contract(schema):
            raise ValueError('Archived feedback semantics mismatch: ' + folder)
        for name, expected in feedback_manifest['files'].items():
            if name != 'localization_kernel.so' and digest(feedback_bundle / name) != expected:
                raise ValueError('Feedback component mismatch: ' + folder + '/' + name)
        checkpoint = StopFeedbackPPO.load(feedback / 'policy.zip', device='cpu')
        if checkpoint.num_timesteps != 4000 or checkpoint.observation_space.shape != (436,):
            raise ValueError('Feedback training checkpoint mismatch: ' + folder)
        if getattr(checkpoint, 'feedback_schema', 'stop_decision_feedback_v1') != schema:
            raise ValueError('Feedback checkpoint semantics mismatch: ' + folder)
        exploration = getattr(checkpoint, 'exploration_schema', 'none')
        expected_exploration = 'sensor_temperature_v1' if folder == 'ppo_explore_4000_s221015' else 'none'
        if exploration != expected_exploration:
            raise ValueError('Feedback exploration semantics mismatch: ' + folder)
        exported = torch.jit.load(str(feedback_bundle / 'actor.pt'), map_location='cpu').eval()
        with torch.no_grad():
            command = exported(torch.zeros((1, 439)))
        if command.shape != (1, 2) or not torch.isfinite(command).all():
            raise ValueError('Feedback TorchScript interface mismatch: ' + folder)
        feedback_models.append(dict(folder=folder, schema=schema, exploration_schema=exploration, additional_steps=4000,
                                    checkpoint_loaded=True, torchscript_loaded=True))
        del checkpoint, exported

    if args.build_native:
        from tools.prepare_native_bundle import build
        native = build(ROOT / 'models/localization_mapping_20261015/temporal_native_candidate/portable',
                       output / 'native_reference')
        os.environ['HANDOFF_NATIVE_BUNDLE'] = str(native)
        for entry in feedback_models:
            bundle = ROOT / 'models/stop_feedback_20261015' / entry['folder'] / 'portable'
            feedback_manifest = json.loads((bundle / 'manifest.json').read_text())
            rebuilt_feedback = build(bundle, output / entry['folder'])
            feedback_runtime = resolve_runtime(feedback_manifest['required_runtime'])(
                rebuilt_feedback, ROOT / 'training' / case['scene'] / 'decision_map.yaml',
                initial, goal, route_mode='shortcut')
            feedback_result = feedback_runtime.step(data['scan'][0], data['odom'][0],
                                                   data['velocity'][0], float(data['t'][0]))
            if (feedback_result['policy_observation_dimensions'] != 439
                    or not np.isfinite(feedback_result['command']).all()):
                raise ValueError('Rebuilt feedback runtime smoke check failed: ' + entry['folder'])
            entry['native_runtime_checked'] = True
    tests = ['training.test_measured_odometry', 'training.test_verified_static_localizer',
             'training.test_fast_static_localizer', 'training.test_global_relocalizer',
             'training.test_decision_feedback', 'experiments.test_stop_feedback_ppo',
             'training.test_navigation_runtime', 'experiments.test_audit_decision_waits',
             'experiments.test_feedback_rollout_trace', 'experiments.test_feedback_alternatives',
             'experiments.test_feedback_exploration', 'experiments.test_feedback_recovery']
    completed = subprocess.run([sys.executable, '-m', 'unittest', '-v', *tests],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (output / 'tests.txt').write_text(completed.stdout)
    print(completed.stdout)
    if completed.returncode:
        raise RuntimeError('Implementation checks failed; see tests.txt')
    report = dict(completed=True, pinned_files_checked=len(pinned), model_files_checked=len(manifest['files']),
        simulation_tasks=len(tasks), real_decisions=len(decisions), real_segments=22,
        observation_dimensions=303, goal_yaw_retained=True, smoke_command=result['command'],
        feedback_ppo_steps=4000, feedback_training_dimensions=436, feedback_runtime_dimensions=439,
        feedback_checkpoint_loaded=True, feedback_torchscript_loaded=True,
        feedback_native_runtime_checked=args.build_native,
        feedback_models=feedback_models,
        native_reference_rebuilt=args.build_native, implementation_tests_returncode=completed.returncode,
        python=sys.version, torch=torch.__version__, cuda_available=torch.cuda.is_available(),
        scope='Integrity, input contracts, one-frame runtime and implementation checks; no closed-loop or real accuracy acceptance.',
        deployment_approved=False)
    (output / 'RESULTS.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
