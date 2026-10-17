"""Measure short/long forecast disagreement on completed recorded inputs."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from forecast_alignment import contract, fuse_current_frame
from neural_navigation.experiment_budget import ExperimentBudget


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'reports/stop_feedback_20261015'
HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = HERE / 'RECORDED_FORECAST_AUDIT.json'
    assert not output.exists()
    report_path = BASE / 'native625_local_speedup_retention_20261017/RESULTS.json'
    report = json.loads(report_path.read_text())
    assert report['completed'] and report['new_physical_episodes'] == 40
    rows = [r for g in report['groups'].values() for r in g['rows']] + report['frozen_rows']
    assert len(rows) == 40
    budget = ExperimentBudget(.1, memory_mib=550, directory=HERE)
    paths = [Path(__file__), HERE / 'forecast_alignment.py', HERE / 'test_forecast_alignment.py', report_path,
             ROOT / 'training/neural_navigation/long_risk_observation.py',
             ROOT / 'training/neural_navigation/long_motion_risk.py',
             ROOT / 'training/neural_navigation/predictive_constraint.py',
             ROOT / 'training/neural_navigation/experiment_budget.py']
    result_rows, examples = [], []
    for row in rows:
        path = ROOT / row['trace_path']
        assert digest(path) == row['trace_sha256']
        paths.append(path)
        with np.load(path, allow_pickle=False) as saved:
            observations = saved['observations']
            diagnostics = saved['constraint_diagnostics']
            labels = saved['labels']
            masks, choices = saved['masks'], saved['choices']
            assert observations.shape[1] == 625 and diagnostics.shape[1] == 15
            old = observations[:, 467:481]
            short = diagnostics[:, 3:]
            count = dict(frames=len(old), supported_short_contact_frames=0,
                         long_missing_short_contact_action_pairs=0, stop_contact_missing_frames=0,
                         stop_contact_missing_while_long_valid=0, stop_contact_missing_only_stop_legal=0)
            for frame in range(len(old)):
                finite = np.isfinite(short[frame])
                valid = bool(observations[frame, 435] > .5 and observations[frame, 438] > .5)
                fused = fuse_current_frame(old[frame], short[frame], int(diagnostics[frame, 0]),
                    float(labels[frame, 0]), float(labels[frame, 0]), input_valid=valid, scan_valid=valid)
                if valid and finite.any():
                    count['supported_short_contact_frames'] += 1
                    missing = finite & (old[frame, :12] == 0.)
                    count['long_missing_short_contact_action_pairs'] += int(missing.sum())
                    if missing[0]:
                        count['stop_contact_missing_frames'] += 1
                        count['stop_contact_missing_while_long_valid'] += int(old[frame, 12] > .5)
                        count['stop_contact_missing_only_stop_legal'] += int(not masks[frame, 1:].any())
                        if row['case_id'] == 'pre_ppo_holdout_sensor_01':
                            examples.append(dict(case_id=row['case_id'], variant=row['variant'],
                                frame=frame, sensor_time_s=float(labels[frame, 0]),
                                original_stop_long_score=float(old[frame, 0]),
                                original_long_valid=bool(old[frame, 12] > .5),
                                guard_stop_contact_s=float(short[frame, 0]),
                                fused_stop_score=float(fused[0]), choice=int(choices[frame]),
                                legal_choices=np.flatnonzero(masks[frame]).tolist()))
                    assert np.all(fused[:12][finite] + 1e-7 >= 1.-short[frame, finite]/5.2)
                if frame % 100 == 0:
                    budget.tick()
        result_rows.append(dict(case_id=row['case_id'], variant=row['variant'], **count))
    totals = {k: sum(r[k] for r in result_rows) for k in result_rows[0] if k not in ['case_id', 'variant']}
    assert any(e['variant'] == 'candidate' and e['legal_choices'] == [0] for e in examples)
    budget.tick()
    result = dict(completed=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        original_retention_reason=report['reason'], rows=result_rows, totals=totals,
        failing_sensor01_examples=examples, prototype_contract=contract(),
        evidence='Saved same-frame current guard TTCs and long history; no network is evaluated on changed semantics',
        no_new_physics=True, new_ppo_steps=0, new_optimizer_updates=0,
        no_model_or_runtime_files_changed=True, not_a_collision_repair_or_success_claim=True,
        no_heldout_traces_admitted_to_training=True, goal_achieved=False, deployment_approved=False,
        inputs={str(p.relative_to(ROOT)): digest(p) for p in set(paths)}, resource=budget.statistics())
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(completed=True, totals=totals, failing_examples=len(examples), resource=budget.statistics())))


if __name__ == '__main__':
    main()
