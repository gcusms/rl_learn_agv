"""Sensor-only shadow of an unexported measured-history geometry prototype."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from neural_navigation.contour_completion import CompletedContourPrediction
from neural_navigation.predictive_constraint import PredictiveConstraint,registered_contract
from neural_navigation.sensors import LidarSpec
from neural_navigation.experiment_budget import ExperimentBudget


def main():
    root=Path('reports/localization_mapping_20261015/blind_repair_20261015')
    trace=root.parent/'anticipatory_ppo_20261015_132151/paired_12/motion_profile_015_constraints_before_ppo.npz'
    hardware=json.loads(Path('models/localization_mapping_20261015/contour_mask_candidate/portable/hardware.json').read_text())
    z=np.load(trace);core=SimpleNamespace(lidar=LidarSpec(hardware['lidar']),odometry_sanity_failure=False)
    versions={name:PredictiveConstraint(registered_contract(True),hardware['robot_footprint_m'],.3)
        for name in ('v4','completion_prototype')}
    versions['completion_prototype'].contours=CompletedContourPrediction()
    budget=ExperimentBudget(.1);rows=[]
    for i,label in enumerate(z['labels']):
        core.elapsed=float(label[0]);core.odom=z['odometry'][i];core.scan=z['scans'][i]
        core.measured_velocity=z['measured_velocity'][i]
        row=dict(time_s=core.elapsed)
        for name,predictor in versions.items():
            predictor.prepare(core)
            row[name]=dict(rejected=[j for j,c in enumerate(predictor.commands) if predictor.rejection(core,c)],
                tracks=[dict(id=int(t['id']),source=t['source'],age=float(t['age']),
                    uncertainty_m=float(t['contour_position_uncertainty_m']),points=len(t['points']),
                    completed=bool(t.get('completed_from_measured_history',False)))
                    for t in predictor.tracks if np.linalg.norm(np.mean(t['points'],axis=0)-core.odom[:2])<1.6])
            budget.tick()
        rows.append(row)
    output=root/'completion_sensor_shadow.json'
    output.write_text(json.dumps(dict(scope='Unexported prototype shadow; not a V4 contract or closed-loop result',
        trace=str(trace),rows=rows,resource=budget.statistics()),indent=2)+'\n')
    for row in rows:
        if 7.3<row['time_s']<9.1:print(json.dumps(row),flush=True)


if __name__=='__main__':main()
