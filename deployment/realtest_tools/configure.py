"""Create location-specific model configs without modifying archived models."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


BUNDLES = {
    'candidate': 'models/stop_feedback_20261015/release_context_bc2000_s231016/portable',
    'baseline': 'models/quality_recovery_20261014/portable',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--kind', choices=BUNDLES, default='candidate')
    parser.add_argument('--bundle', type=Path, help='Optional separately rebuilt native bundle')
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root), str(root / 'training')]
    from deployment.ros2_recording_adapter.model_bundle import inspect_bundle
    bundle = args.bundle.resolve() if args.bundle else root / BUNDLES[args.kind]
    manifest = json.loads((bundle / 'manifest.json').read_text())
    expected = 625 if args.kind == 'candidate' else 300
    if manifest['observation_contract']['dimensions'] != expected:
        raise ValueError('Bundle does not match --kind')
    source_identity = hashlib.sha256((root / 'SOURCE.json').read_bytes()).hexdigest()
    config = dict(branch='realtest-20261016-local-snapshot',
        branch_commit='uncommitted-source-sha256:' + source_identity,
        bundle=str(bundle),
        required_runtime=manifest.get('required_runtime',
            'experiments.quality_sensor_runtime.QualitySensorNavigationRuntime'),
        sensor_core=manifest.get('sensor_core', 'native'),
        manifest_sha256=hashlib.sha256((bundle / 'manifest.json').read_bytes()).hexdigest(),
        observation_dimensions=expected, action_choices=12, route_mode='shortcut', mode='shadow')
    output = root / 'generated' / ('model_' + args.kind + '.json')
    output.parent.mkdir(exist_ok=True)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(config, indent=2) + '\n')
    try:
        inspect_bundle(temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(output)
    print('model_source=' + str(root))


if __name__ == '__main__':
    main()
