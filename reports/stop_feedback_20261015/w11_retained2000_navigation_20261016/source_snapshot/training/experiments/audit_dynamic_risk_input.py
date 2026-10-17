"""Compare declared actor forecasts with the actual constraint forecasts.

Disagreement is a diagnostic: the models have different horizons and shape
uncertainty, so it does not establish which forecast is physically correct.
No feature, action or execution contract is changed here.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    root=Path('reports/localization_mapping_20261015/blind_repair_20261015')
    bundle=Path('models/localization_mapping_20261015/contour_mask_candidate/portable')
    trace_path=root.parent/'anticipatory_ppo_20261015_132151/paired_12/motion_profile_015_constraints_before_ppo.npz'
    quality=json.loads((bundle/'quality_observation.json').read_text())
    if quality['schema']!='quality_prediction_append_v1' or quality['prefix_dimensions']!=282:
        raise ValueError('Pinned quality feature semantics required')
    commands=np.asarray(quality['commands_m_s_rad_s'])
    expected=np.array([[0,0],[.2,0],[.1,0],[.05,0],[0,.5],[0,-.5],
        [.05,.5],[.05,-.5],[.01,.5],[.01,-.5],[.2,.5],[.2,-.5]])
    np.testing.assert_array_equal(commands,expected)
    z=np.load(trace_path);actor=z['observations'][:,282:294];contacts=z['constraint_diagnostics'][:,3:]
    if actor.shape!=contacts.shape:raise ValueError('Risk action ordering mismatch')
    # Normalize both times on the actor's declared scale solely for comparison.
    scale=quality['horizon_s']+quality['period_s']
    constraint=np.where(np.isfinite(contacts),1-contacts/scale,0)
    disagreement=(constraint-actor)>.25
    rows=[]
    for index in np.flatnonzero(disagreement.any(axis=1)):
        rows.append(dict(time_s=float(z['labels'][index,0]),
            legal_moving=int(z['masks'][index,1:].sum()),
            quality_risk=actor[index].tolist(),constraint_contact_s=[None if not np.isfinite(t) else float(t) for t in contacts[index]],
            normalized_constraint_risk=constraint[index].tolist(),
            underrepresented_actions=np.flatnonzero(disagreement[index]).tolist()))
    report=dict(scope='Forecast consistency diagnostic on a frozen known failure, not physical risk calibration',
        source_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        actor_source='Map-frame laser-fusion tracks, quality contour/innovation processing',
        guard_source='Odometry-frame cumulative tracks with retained shape/velocity hypotheses',
        common_input='Both read the same degraded laser scan; their forecast states differ',
        frames=len(actor),disagreement_frames=len(rows),
        stop_risk_disagreement_frames=int(disagreement[:,0].sum()),rows=rows)
    output=root/'dynamic_risk_input_audit.json';output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
