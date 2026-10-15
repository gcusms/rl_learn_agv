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
        'learned-terminal-goal-pose': 'experiments.docking_navigation.DockingRuntime',
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
    directional_core=sensor_core in ('odom-goal-pose','learned-terminal-goal-pose')
    hardware=json.loads((bundle/'hardware.json').read_text())
    from neural_navigation.masked_actions import validate_hardware_action
    if hardware.get('policy_action')!=manifest['action_contract']:
        raise ValueError('Hardware/manifest action contract mismatch')
    validate_hardware_action(hardware)
    if (bundle/'uncertainty_contract.json').exists():
        from neural_navigation.weak_geometry_uncertainty import contract as uncertainty_contract
        if 'uncertainty_contract.json' not in manifest['files'] or json.loads((bundle/'uncertainty_contract.json').read_text())!=uncertainty_contract():
            raise ValueError('Uncertainty contract mismatch')
    if (bundle/'routing_contract.json').exists():
        from neural_navigation.routing_contract import contract as routing_contract
        if 'routing_contract.json' not in manifest['files'] or json.loads((bundle/'routing_contract.json').read_text())!=routing_contract():
            raise ValueError('Routing acceptance contract mismatch')
    expected_quality=manifest.get('quality_prefix_contract') if directional_core else manifest['observation_contract']
    if quality != expected_quality:
        raise ValueError('Quality feature contract mismatch')
    if directional_core:
        goal_contract=json.loads((bundle/'goal_pose_observation.json').read_text())
        if (manifest['observation_contract']!=goal_contract
            or goal_contract.get('schema')!='quality_goal_pose_v1' or goal_contract.get('dimensions')!=303):
            raise ValueError('Goal-pose feature contract mismatch')
        if 'goal_pose_observation.json' not in manifest['files']:
            raise ValueError('Missing goal-pose contract hash')
        if (bundle/'localization_contract.json').exists() and (
            'localization_contract.json' not in manifest['files'] or
            json.loads((bundle/'localization_contract.json').read_text()).get('schema') not in
                ('odom_anchored_static_support_v1','odom_anchored_static_support_v2','odom_anchored_static_support_v3','odom_anchored_static_support_v4','odom_anchored_static_support_v5','odom_anchored_static_support_v6','fast_static_tracking_v1','fast_static_tracking_native_v1','fast_static_tracking_v2','fast_static_tracking_native_v2','fast_static_tracking_temporal_v1','fast_static_tracking_temporal_native_v1')):
            raise ValueError('Localization contract mismatch')
        if (bundle/'localization_contract.json').exists() and json.loads((bundle/'localization_contract.json').read_text()).get('schema') in ('fast_static_tracking_native_v1','fast_static_tracking_native_v2','fast_static_tracking_temporal_native_v1') and 'localization_kernel.so' not in manifest['files']:
            raise ValueError('Missing native localization kernel hash')
    if sensor_core=='learned-terminal-goal-pose':
        if 'docking_contract.json' not in manifest['files']:
            raise ValueError('Missing terminal policy contract hash')
        docking=json.loads((bundle/'docking_contract.json').read_text())
        version_radius={'learned_terminal_actor_v1':.8,'learned_terminal_actor_v2':.10}
        expected_radius=version_radius.get(docking.get('schema'))
        if (docking!=manifest.get('docking_contract') or expected_radius is None
            or docking.get('activation_distance_m')!=expected_radius
            or docking.get('observation_dimensions')!=303 or docking.get('actions')!=12
            or docking.get('teacher_at_runtime') is not False):
            raise ValueError('Terminal policy contract mismatch')
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
