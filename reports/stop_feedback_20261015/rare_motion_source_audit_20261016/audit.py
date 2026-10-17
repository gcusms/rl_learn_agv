"""Find matching constrained motion in already complete-safe reference traces."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
cpu, wall = time.process_time(), time.monotonic()
tasks = json.loads((ROOT / 'reports/stop_feedback_20261015/context_feedback_longfit_protocol_20261016/tasks.json').read_text())
seen = set()
rows = []
for group, references in tasks['reference'].items():
    for reference in references:
        if not reference['safe_success']:
            continue
        assert reference['physical_arrival'] and not any(reference[k] for k in
            ['collisions', 'false_arrival', 'physical_guards', 'timeout'])
        trace = reference.get('trace_path')
        if not trace:
            continue
        path = ROOT / trace
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert sha == reference['trace_sha256'], trace
        if sha in seen:
            continue
        seen.add(sha)
        with np.load(path, allow_pickle=False) as z:
            x, masks, choices = [z[k] for k in ['observations', 'masks', 'choices']]
            assert x.shape[1] >= 439
            assert masks[np.arange(len(choices)), choices].all()
            control = z['model_controlled'].astype(bool) if 'model_controlled' in z.files else np.ones(len(x), bool)
            if reference.get('recovery_test') and 'model_controlled' not in z.files:
                control = z['labels'][:, 0] >= reference['release_sensor_timestamp_s'] - 1e-8
            far = np.linalg.norm(x[:, 218:220] * 12, axis=1) > .3
            valid = (x[:, 435] > .5) & (x[:, 438] > .5)
            selected = control & far & valid & (choices != 0)
            exact09 = masks[:, 0] & masks[:, 9] & (masks.sum(1) == 2)
            sparse = masks[:, 1:].sum(1) <= 2
            laser_reason = x[:, 409] > .5
            long_wait = x[:, 434] * 5 >= 2 - 1e-6
            near = x[:, 144:216].min(1) * 12 < .25
            matches = selected & exact09 & (choices == 9)
            exact09_context = control & far & valid & exact09
            indices = np.flatnonzero(matches)
            rows.append(dict(group=group, case_id=reference['case_id'], trace=trace,
                trace_sha256=sha, observation_dimensions=x.shape[1], complete_safe_reference=True,
                eligible_moving_frames=int(selected.sum()),
                exact0_9_valid_context_frames=int(exact09_context.sum()),
                exact0_9_stop_frames=int((exact09_context & (choices == 0)).sum()),
                exact0_9_motion9_frames=int(matches.sum()),
                exact0_9_motion9_after_laser_stop=int((matches & laser_reason).sum()),
                exact0_9_motion9_after_long_wait=int((matches & long_wait).sum()),
                exact0_9_motion9_near_scan=int((matches & near).sum()),
                sparse_motion9_frames=int((selected & sparse & (choices == 9)).sum()),
                examples=[dict(index=int(i), sensor_time=float(z['labels'][i, 0]),
                    nearest_scan_m=float(x[i, 144:216].min() * 12),
                    previous_legal_wait_s=float(x[i, 434] * 5),
                    previous_reason_index=int(x[i, 399:419].argmax())) for i in indices[:10]]))
        # This audit is small and runs alongside the bounded physics/fit jobs.
        delay=(time.process_time()-cpu)/.02-(time.monotonic()-wall)
        while delay > .002:
            time.sleep(min(.1, delay))
            delay=(time.process_time()-cpu)/.02-(time.monotonic()-wall)
result=dict(completed=True, unique_complete_safe_traces=len(rows),
    exact0_9_context_frames=sum(r['exact0_9_valid_context_frames'] for r in rows),
    exact0_9_stop_frames=sum(r['exact0_9_stop_frames'] for r in rows),
    matched_traces=sum(r['exact0_9_motion9_frames'] > 0 for r in rows),
    matching_frames=sum(r['exact0_9_motion9_frames'] for r in rows),
    matching_after_long_wait=sum(r['exact0_9_motion9_after_long_wait'] for r in rows),
    matching_after_laser_stop=sum(r['exact0_9_motion9_after_laser_stop'] for r in rows),
    rows=rows, no_training_or_new_physics_or_labels=True,
    future_admission_requires_causal625D_replay_and_semantics_checks=True,
    limitations=['Reference cohorts are development; adding sources makes them training regressions.',
        'A matched legal set is not proof of physical similarity or actual106 recovery.'],
    seconds=time.monotonic()-wall, cpu_seconds=time.process_time()-cpu)
(OUT/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k != 'rows'}))
