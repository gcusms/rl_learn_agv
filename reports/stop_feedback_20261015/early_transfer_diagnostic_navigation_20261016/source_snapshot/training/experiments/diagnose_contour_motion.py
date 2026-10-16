"""Causal sensor shadow of versioned dynamic constraints, without sim truth."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.sensors import LidarSpec
from neural_navigation.predictive_constraint import PredictiveConstraint,cumulative_contract,registered_contract


def main():
    p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpu-cores',type=float,default=.1);a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    hardware=json.loads((a.bundle/'hardware.json').read_text());z=np.load(a.trace)
    core=SimpleNamespace(lidar=LidarSpec(hardware['lidar']),odometry_sanity_failure=False)
    versions=dict(cumulative=cumulative_contract(),registered=registered_contract())
    predictors={key:PredictiveConstraint(config,hardware['robot_footprint_m'],.3) for key,config in versions.items()}
    budget=ExperimentBudget(a.cpu_cores);rows=[];parity=True
    for i,label in enumerate(z['labels']):
        core.elapsed=float(label[0]);core.odom=z['odometry'][i];core.scan=z['scans'][i];core.measured_velocity=z['measured_velocity'][i]
        row=dict(time_s=core.elapsed,original_choice=int(z['choices'][i]),original_allowed=np.flatnonzero(z['masks'][i]).tolist())
        for name,predictor in predictors.items():
            predictor.prepare(core)
            tracks=[]
            for track in predictor.tracks:
                if np.linalg.norm(np.mean(track['points'],axis=0)-core.odom[:2])>1.6:continue
                tracks.append(dict(id=int(track['id']),hits=int(track['hits']),measurement_age_s=float(track['age']),
                    velocity_m_s=np.asarray(track['velocity']).tolist(),source=track['source'],
                    retained_without_measurement=bool(track['retained_without_measurement']),
                    contour_uncertainty_m=float(track['contour_position_uncertainty_m'])))
            row[name]=dict(predictor.diagnostics(core),tracks=tracks)
            if name=='cumulative':
                actual=np.where(np.isfinite(predictor.contacts),predictor.contacts,np.nan)
                parity=parity and bool(np.allclose(actual,z['constraint_diagnostics'][i,3:],rtol=0,atol=1e-8,equal_nan=True))
        rows.append(row);budget.tick()
    result=dict(scope='Causal sensor-only shadow; not closed-loop safety acceptance',
        original_v2_contact_parity=parity,trace=str(a.trace.resolve()),rows=rows,resource=budget.statistics())
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(original_v2_contact_parity=parity,frames=len(rows),resource=result['resource']),indent=2))


if __name__=='__main__':main()
