"""Validate the selected recovery bundle without changing source or weights."""
import hashlib
import json
from pathlib import Path


def inspect_bundle(config_path):
    config = json.loads(Path(config_path).read_text())
    bundle = Path(config['bundle'])
    manifest_bytes = (bundle/'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if config.get('manifest_sha256') and hashlib.sha256(manifest_bytes).hexdigest() != config['manifest_sha256']:
        raise ValueError('Model manifest checksum mismatch')
    # The native 2026-10-14 export predates runtime metadata in the portable
    # manifest. Its release uses the quality runtime, not the time-consistent
    # core required by the older recovery bundle. Never infer from dimension.
    sensor_core = manifest.get('sensor_core', 'native')
    runtimes = {
        'native': 'experiments.quality_sensor_runtime.QualitySensorNavigationRuntime',
        'time-consistent': 'experiments.time_consistent_tracking.TimeConsistentFreshRuntime',
        'odom-goal-pose': 'experiments.goal_pose_navigation.GoalPoseRuntime',
    }
    runtime = runtimes.get(sensor_core)
    if runtime is None or config['required_runtime'] != runtime:
        raise ValueError('Model runtime contract mismatch')
    if manifest.get('required_runtime', runtime) != runtime:
        raise ValueError('Model runtime contract mismatch')
    if config.get('sensor_core', sensor_core) != sensor_core:
        raise ValueError('Model sensor core mismatch')
    if sensor_core == 'native' and not config.get('manifest_sha256'):
        raise ValueError('Native model requires a pinned manifest checksum')
    if manifest['observation_contract']['dimensions'] != config['observation_dimensions']:
        raise ValueError('Observation dimension mismatch')
    if len(manifest['action_contract']['choices']) != config['action_choices']:
        raise ValueError('Action contract mismatch')
    quality = json.loads((bundle/'quality_observation.json').read_text())
    expected_quality=manifest.get('quality_prefix_contract') if sensor_core=='odom-goal-pose' else manifest['observation_contract']
    if quality != expected_quality:
        raise ValueError('Quality feature contract mismatch')
    if sensor_core=='odom-goal-pose':
        goal_contract=json.loads((bundle/'goal_pose_observation.json').read_text())
        if (manifest['observation_contract']!=goal_contract
            or goal_contract.get('schema')!='quality_goal_pose_v1' or goal_contract.get('dimensions')!=303):
            raise ValueError('Goal-pose feature contract mismatch')
        if 'goal_pose_observation.json' not in manifest['files']:
            raise ValueError('Missing goal-pose contract hash')
        if (bundle/'localization_contract.json').exists() and (
            'localization_contract.json' not in manifest['files'] or
            json.loads((bundle/'localization_contract.json').read_text()).get('schema')!='odom_anchored_static_support_v1'):
            raise ValueError('Localization contract mismatch')
    if not manifest.get('requires_quality_feature_generator'):
        raise ValueError('Quality feature generator required')
    for required in ('actor.pt','localizer.pt','router.pt','hardware.json','quality_observation.json'):
        if required not in manifest['files']:
            raise ValueError('Missing bundle hash: '+required)
    for name, expected in manifest['files'].items():
        if Path(name).name != name:
            raise ValueError('Invalid bundle filename')
        if hashlib.sha256((bundle/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Model bundle checksum mismatch: '+name)
    return dict(config, sensor_core=sensor_core, actor_sha256=manifest['files']['actor.pt'],
                deployment_approved=manifest.get('deployment_approved',False),
                sensor_core_parameters=manifest.get('sensor_core_parameters', {}))
