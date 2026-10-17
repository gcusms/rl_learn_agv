"""Independent comparison of saved initialization sensors and physical traces."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent.parent / 'deployment_acceptance_supplement_20261017'


def main():
    records = []
    for job_path in sorted((OUT / 'jobs').glob('*.json')):
        if job_path.stem.endswith('_RESULT'):
            continue
        job = json.loads(job_path.read_text())
        spec = job['spec']
        row = json.loads(Path(job['output']).read_text())
        n = row['frozen_prefix_frames']
        actual_path = Path(job['trace'])
        assert hashlib.sha256(actual_path.read_bytes()).hexdigest() == row['trace_sha256']
        checks = {}
        with np.load(actual_path, allow_pickle=False) as actual, np.load(ROOT / spec['trace'], allow_pickle=False) as ref:
            np.testing.assert_array_equal(actual['choices'][:n], ref['choices'][:n])
            np.testing.assert_allclose(actual['labels'][:n + 1, :4], ref['labels'][:n + 1, :4], rtol=0, atol=2e-8)
            assert not actual['model_controlled'][:n].any()
            assert actual['model_controlled'][n:].all()
            for name in actual.files:
                if name not in ref.files or name == 'model_controlled':
                    continue
                a, b = actual[name], ref[name]
                if a.ndim == 0 or b.ndim == 0 or a.shape[1:] != b.shape[1:] or min(len(a), len(b)) < n:
                    continue
                if a.dtype.kind in 'biuf' and b.dtype.kind in 'biuf':
                    matched = bool(np.array_equal(a[:n], b[:n], equal_nan=True))
                    checks[name] = dict(prefix_frames=n, exact=matched)
            # Full sensor/odometry arrays are reported individually; timing counters are not parity criteria.
            essential = ['observations', 'scans', 'odometry', 'measured_velocity', 'commands', 'masks']
            assert all(key in checks for key in essential), (spec['id'], checks)
            for key in essential:
                assert checks[key]['exact'], (spec['id'], key)
            records.append(dict(case_id=spec['id'], prefix_frames=n,
                                actual_trace_fields=actual.files, reference_trace_fields=ref.files,
                                shared_array_checks=checks, essential_exact_arrays=essential,
                                physics_pose_prefix_within_2e_8_m_rad=True,
                                prefix_commands_initialization_only=True,
                                all_after_release_frames_learned_controlled=True,
                                not_a_full_mission_score=True))
    assert len(records) == 2
    value = dict(completed=True, prefix_sensor_and_physical_parity_verified=True,
                 records=records, new_ppo_steps=0, new_training_updates=0,
                 no_localization_real_car_or_generalization_claim=True,
                 deployment_approved=False)
    (OUT / 'TRACE_AUDIT.json').write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
