"""Causal replay of the recorded runtime; inspect, never replace, its route.

Offline connectivity is diagnostic evidence only. The optional existing
learned decoder runs on a shallow copy and cannot affect replayed actions.
"""
import copy
from collections import deque
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OUT = BASE / 'sensor01_route_diagnosis_20261017'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def connectivity(free, start, target, directions, allow_start=False):
    h, w = free.shape
    inside = lambda p: 0 <= p[0] < h and 0 <= p[1] < w
    if not inside(start) or not inside(target):
        return dict(connected=False, reason='outside')
    if not free[target]:
        return dict(connected=False, reason='target_blocked')
    if not free[start] and not allow_start:
        return dict(connected=False, reason='start_blocked')
    queue = deque([start]); distance = {start: 0}
    while queue:
        r, c = queue.popleft()
        if (r, c) == target:
            return dict(connected=True, steps=distance[(r, c)], visited=len(distance))
        for dr, dc in directions:
            nr, nc = r + int(dr), c + int(dc)
            if not inside((nr, nc)) or not free[nr, nc] or (nr, nc) in distance:
                continue
            # Conservative graph diagnostic: do not cross blocked corners.
            if dr and dc and not (free[r, nc] and free[nr, c]):
                continue
            distance[(nr, nc)] = distance[(r, c)] + 1
            queue.append((nr, nc))
    return dict(connected=False, reason='disconnected', visited=len(distance))


def main():
    assert not OUT.exists(), 'Fresh diagnostic output required'
    OUT.mkdir()
    # A collected predecessor is terminal. Never restart it or alter its data.
    memory_path = BASE / 'hybrid_memory_pair_preflight_20261017/RESULTS.json'
    memory = read(memory_path)
    assert memory['completed'] and memory['new_physical_episodes'] == 2
    job_path = BASE / 'v2_behavior_retention_validation_20261017/jobs/observed6_pre_ppo_holdout_sensor_01_candidate.json'
    job = read(job_path); case = job['spec']; bundle = Path(job['bundle'])
    trace_path = Path(job['trace'])
    previous_plan = read(BASE / 'v2_behavior_retention_validation_20261017/PLAN.json')
    expected_trace = 'd68e0d9db6b9f308f79d80d04a8036b834e11741c3b19645cd253a0313ef22eb'
    assert digest(trace_path) == expected_trace
    paths = [Path(__file__), job_path, trace_path, memory_path]
    # Verify the source files that generated the original trace are unchanged.
    for name, expected in previous_plan['inputs'].items():
        if name.startswith(('training/neural_navigation/', 'training/navrl/')):
            assert digest(ROOT / name) == expected, name
            paths.append(ROOT / name)
    manifest = read(bundle / 'manifest.json')
    paths += [bundle / 'manifest.json'] + [bundle / name for name in manifest['files']]
    scene = ROOT / 'training' / case['scene']
    paths += [p for p in scene.iterdir() if p.is_file()]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in set(paths)}
    write(OUT / 'PLAN.json', dict(at=now(), inputs=hashes,
        source_case=case, frozen_bundle=str(bundle.relative_to(ROOT)),
        exact_recorded_sensor_stream=True, model_weights_unchanged=True,
        original_runtime_actions_remain_authoritative=True,
        shadow_existing_decoder_never_changes_live_runtime=True,
        offline_graph_connectivity_not_policy_input_or_runtime_fallback=True,
        new_physical_episodes=0, new_ppo_steps=0, new_optimizer_updates=0,
        soft_cpu_cores=.35, rss_budget_mib=2800, hard_service_memory_mib=3072,
        goal_achieved=False, deployment_approved=False))
    write(OUT / 'STATUS.json', dict(status='loading_runtime', at=now()))
    import numpy as np
    import torch
    from neural_navigation.aligned_release_runtime import AlignedReleaseRuntime
    from neural_navigation.core import LearnedRoutingMixin
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.masked_actions import ACTIONS, masks_from_observation
    from neural_navigation.route_efficiency import efficient_learned_route
    torch.set_num_threads(1)
    budget = ExperimentBudget(.35, 2800, OUT)
    trace = np.load(trace_path, allow_pickle=False)
    runtime = AlignedReleaseRuntime(bundle, scene / 'decision_map.yaml',
        np.asarray(case['start']) + np.asarray(case.get('initial_prior_offset', [0, 0, 0])),
        case['goal'], 'shortcut')
    budget.tick()
    route_rows = []; arrays = {}; frame_rows = []
    original = LearnedRoutingMixin._neural_route

    def inspect_route(core):
        original(core)
        # This is the raw candidate before the RoomRoutingMixin applies its
        # existing shortcut and retention rules.
        dyn = core.laser_fusion.layer
        start = core.cell(core.estimate[:2]); target = core.cell(core.goal)
        gm = np.zeros_like(core.free); gm[target] = 1.
        pm = np.zeros_like(core.free); pm[start] = 1.
        current = np.stack([core.free * (1-dyn), dyn, gm, pm])
        static = np.stack([core.free, np.zeros_like(dyn), gm, pm])
        if core.start_inflation_overlap:
            current[0, start[0], start[1]] = core.free[start]
            current[1, start[0], start[1]] = 0.
        key = np.stack([current, static]).tobytes()
        assert key in core.router_cache, 'Must inspect the exact current Q field'
        q, reach = core.router_cache[key]
        raw = core.predicted_path.copy()
        observed = core.laser_fusion.observed_hard
        hard = core.laser_fusion.routing_hard
        occupancy = (core.free > 0) & ~hard
        shadow = copy.copy(core)
        shadow.static_visibility_cache = dict(core.static_visibility_cache)
        shadow_path = efficient_learned_route(shadow, q, occupancy, raw)
        # No state or route from the shadow is installed on the runtime.
        np.testing.assert_array_equal(core.predicted_path, raw)
        r, c = start; seen = set(); decode_reason = 'step_limit'; blocked = []
        for step in range(160):
            if not (0 <= r < core.grid_h and 0 <= c < core.grid_w):
                decode_reason = 'outside'; break
            if (r, c) in seen:
                decode_reason = 'loop'; break
            seen.add((r, c))
            if (r, c) == target:
                decode_reason = 'goal'; break
            direction = int(np.argmax(q[:, r, c]))
            dr, dc = core.router.directions[direction]
            r += dr; c += dc
            if 0 <= r < core.grid_h and 0 <= c < core.grid_w and not occupancy[r, c]:
                blocked.append([int(r), int(c)])
        record = dict(time_s=float(core.elapsed), start=list(start), target=list(target),
            start_static_free=bool(core.free[start] > 0),
            start_observed_hard=bool(observed[start]), start_routing_hard=bool(hard[start]),
            escape_start=bool(core.escape_start), start_inflation_overlap=bool(core.start_inflation_overlap),
            routing_contract=getattr(core, 'routing_contract', None),
            raw_complete=bool(len(raw)>1 and np.linalg.norm(raw[-1]-core.goal)<.12),
            raw_endpoint_error_m=float(np.linalg.norm(raw[-1]-core.goal)), raw_vertices=len(raw),
            greedy_decode_end=decode_reason, decoded_blocked_cells=blocked,
            current_probability=float(core.current_probability), static_probability=float(core.static_probability),
            static_connectivity=connectivity(core.free > 0, start, target, core.router.directions),
            measured_connectivity=connectivity((core.free > 0) & ~observed, start, target, core.router.directions),
            hard_connectivity=connectivity(occupancy, start, target, core.router.directions),
            neural_input_connectivity=connectivity(current[0] > 0, start, target, core.router.directions),
            shadow_complete=bool(len(shadow_path)>1 and np.linalg.norm(shadow_path[-1]-core.goal)<.12),
            shadow_usable=bool(shadow._route_usable(shadow_path)),
            shadow_candidate_count=int(shadow.route_candidate_count),
            shadow_endpoint_error_m=float(np.linalg.norm(shadow_path[-1]-core.goal)),
            shadow_path_length_m=float(np.linalg.norm(np.diff(shadow_path, axis=0), axis=1).sum()),
            measured_cells=int(observed.sum()), routing_hard_cells=int(hard.sum()),
            forecast_cells=int(core.laser_fusion.forecast_cells))
        route_rows.append(record)
        if core.elapsed >= 11.8:
            prefix=f'route_{len(route_rows):03d}_'
            arrays.update({prefix+'q':q.copy(),prefix+'free':core.free.copy(),
                prefix+'dynamic_layer':dyn.copy(),prefix+'observed_hard':observed.copy(),
                prefix+'routing_hard':hard.copy(),prefix+'raw_path':raw.copy(),
                prefix+'shadow_path':shadow_path.copy(),prefix+'estimate':core.estimate.copy()})

    LearnedRoutingMixin._neural_route = inspect_route
    try:
        exact_observations = 0
        for index, label in enumerate(trace['labels']):
            result = runtime.step(trace['scans'][index], trace['odometry'][index],
                trace['measured_velocity'][index], float(label[0]), capture_observation=True)
            obs = np.asarray(result['policy_observation'], np.float32)
            exact_observations += int(np.array_equal(obs, trace['observations'][index]))
            np.testing.assert_allclose(obs, trace['observations'][index], rtol=0, atol=2e-6)
            np.testing.assert_array_equal(masks_from_observation(obs[:282]), trace['masks'][index])
            choice = int(np.max(abs(ACTIONS-np.asarray(result['proposed_action'])),axis=1).argmin())
            assert choice == int(trace['choices'][index]), (index, choice)
            np.testing.assert_allclose(result['command'], trace['commands'][index], rtol=0, atol=2e-8)
            np.testing.assert_allclose(runtime.core.estimate, label[4:7], rtol=0, atol=2e-8)
            core = runtime.core
            route = trace['route_diagnostics'][index]
            np.testing.assert_allclose([core.plan_revision,len(core.path),*core.path[-1]],route[:4],rtol=0,atol=2e-8)
            assert core.replan_reason == str(trace['route_reasons'][index])
            frame_rows.append(dict(frame=index,time_s=float(label[0]),action=choice,
                legal_actions=np.flatnonzero(trace['masks'][index]).tolist(),
                stop_reason=result['stop_reason'],route_complete=core.diagnostics()['route_complete'],
                route_switch_reason=core.replan_reason,
                current_rejection_reasons=result['current_action_rejection_reasons']))
            budget.tick()
            if index % 10 == 0:
                write(OUT/'STATUS.json',dict(status='sensor_replay',at=now(),frame=index+1,total_frames=len(trace['labels'])))
        assert 'mujoco' not in sys.modules, 'This audit must not load physics'
        for name, expected in hashes.items():
            assert digest(ROOT/name)==expected,name
        np.savez_compressed(OUT/'ROUTER_DIAGNOSTICS.npz',**arrays)
        report=dict(completed=True,at=now(),source_case=case['id'],frames=len(frame_rows),
            exact625_observation_frames=exact_observations,
            replay_obs_max_tolerance=2e-6,mask_action_command_pose_route_parity_passed=True,
            routes=route_rows,frames_detail=frame_rows,resource=budget.statistics(),
            all_input_hashes_reverified=len(hashes),new_physical_episodes=0,
            new_ppo_steps=0,new_optimizer_updates=0,weights_or_runtime_route_changed=False,
            offline_shadow_not_closed_loop_success=True,goal_achieved=False,deployment_approved=False)
        write(OUT/'RESULTS.json',report)
        write(OUT/'STATUS.json',dict(status='complete',at=now(),frames=len(frame_rows)))
        print(json.dumps({k:v for k,v in report.items() if k not in ('routes','frames_detail')},ensure_ascii=False),flush=True)
    finally:
        LearnedRoutingMixin._neural_route = original


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OUT.exists():
            write(OUT/'FAILURE.json',dict(at=now(),error=repr(error),no_auto_restart=True))
        raise
