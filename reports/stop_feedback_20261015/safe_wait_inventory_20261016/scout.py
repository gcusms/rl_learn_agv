"""Find verified, model-controlled late-wait recoveries in existing safe traces."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT/'reports/stop_feedback_20261015'


def main():
    if (OUT/'RESULTS.json').exists():
        raise FileExistsError('Do not replace an inventory')
    sources = [
        'w11_recovery_bc800_navigation_20261016',
        'w11_recovery_bc800_prefixes_20261016',
        'w11_recovery_bc800_w11_20261016',
        'w11_recovery_bc800_fresh_20261016',
        'w11_control2000_navigation_20261016',
        'w11_control2000_prefixes_20261016',
        'w11_control2000_w11_20261016',
        'w11_control2000_fresh_20261016',
        'w11_retained2000_navigation_20261016',
        'w11_retained2000_prefixes_20261016',
        'w11_retained2000_w11_20261016',
        'w11_retained2000_fresh_20261016',
        'proactive_augmented_ppo8000_navigation_20261016',
        'proactive_augmented_ppo8000_prefixes_20261016',
        'proactive_augmented_ppo8000_w11_20261016',
        'proactive_augmented_ppo8000_fresh_20261016',
        'early_transfer_diagnostic_navigation_20261016',
        'early_transfer_diagnostic_reference_20261016',
    ]
    inputs = {str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    rows = []
    skipped = []
    seen = set()
    total_safe = 0
    safe_frames = 0
    for name in sources:
        folder = BASE/name
        path = folder/'RESULTS.json'
        if not path.exists():
            skipped.append(dict(folder=name,reason='not_started'))
            continue
        raw = path.read_bytes()
        report = json.loads(raw)
        if not report['completed']:
            skipped.append(dict(folder=name,reason='incomplete_at_inventory'))
            continue
        inputs[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        for row in report['rows']:
            if (not row['safe_success'] or row['collisions'] or row['false_arrival']
                    or row.get('physical_guards',0) or not row.get('protocol_valid',True)):
                continue
            suffix = '' if row.get('recovery_test') else '_'+row['variant']
            path = folder/(row['case_id']+suffix+'.npz')
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != row['trace_sha256']:
                raise ValueError('A completed physical trace changed')
            inputs[str(path.relative_to(ROOT))] = digest
            if digest in seen:
                continue
            seen.add(digest)
            with np.load(path,allow_pickle=False) as data:
                obs = data['observations']
                masks = data['masks'].astype(bool)
                choices = data['choices'].astype(int)
                commands = data['commands']
                times = data['labels'][:,0]
                controlled = data['model_controlled'].astype(bool) if row.get('recovery_test') else np.ones(len(obs),bool)
                if (obs.shape!=(len(times),439) or masks.shape!=(len(times),12)
                        or not np.isfinite(obs).all() or not masks[np.arange(len(times)),choices].all()):
                    raise ValueError('Feedback observation/action contract changed')
                encoded = np.c_[np.ones(len(obs),bool),obs[:,235:244]>.5,obs[:,267:269]>.5]
                np.testing.assert_array_equal(encoded,masks)
                usable = masks[:,1:].any(-1) & (obs[:,435]>.5) & (obs[:,438]>.5) & (
                    np.linalg.norm(obs[:,218:220]*12.,axis=-1)>.3)
                long_wait = obs[:,434]*5. >= 2.-1e-6
                selected = controlled & usable & long_wait
                moves = selected & (choices!=0) & (np.max(abs(commands),axis=-1)>1e-8)
                total_safe += 1
                safe_frames += int(controlled.sum())
                if selected.any():
                    rows.append(dict(report=name,case_id=row['case_id'],variant=row['variant'],
                        trace=str(path.relative_to(ROOT)),trace_sha256=digest,
                        recovery_test=bool(row.get('recovery_test')),
                        prior_legal_wait_minimum_s=2.,safe_episode_seconds=row['seconds'],
                        qualifying_frames=int(selected.sum()),motion_frames=int(moves.sum()),
                        motion_indices=np.flatnonzero(moves).tolist(),
                        motion_sensor_times_s=times[moves].tolist(),
                        motion_actions=choices[moves].tolist(),
                        qualifying_stop_frames=int((selected & (choices==0)).sum())))
    result = dict(completed=True,at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        unique_safe_traces=total_safe,actual_model_controlled_safe_frames=safe_frames,
        traces_with_prior_legal_wait=len(rows),
        traces_with_model_motion_after_prior_legal_wait=sum(r['motion_frames']>0 for r in rows),
        motion_frames_after_prior_legal_wait=sum(r['motion_frames'] for r in rows),
        rows=rows,skipped=skipped,inputs=inputs,
        minimum_prior_legal_wait_seconds=2.,source_guard='Unmodified standard evaluated model bundles only; rotation diagnostics excluded.',
        scripted_prefix_frames_excluded=True,new_physical_steps=0,new_ppo_steps=0,
        not_yet_admitted_as_demonstration=True,
        limitations=['Episode success alone does not prove every intermediate action is optimal.',
            'Different policies/routes are correlated development data, not independent generalization trials.',
            'Inspect complete recovery and necessity of any wait before creating labels.'],
        goal_achieved=False,deployment_approved=False)
    (OUT/'RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='inputs'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
