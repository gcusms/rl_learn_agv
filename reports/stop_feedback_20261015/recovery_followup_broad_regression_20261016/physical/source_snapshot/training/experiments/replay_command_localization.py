"""Diagnose localization on fixed executed commands; not a navigation score."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from navrl.geometry import ROOT
from navrl.map import StaticMap
from neural_navigation.odom_anchored_localizer import OdometryAnchoredLocalizer
from neural_navigation.sensors import LidarSpec
from experiments.command_odometry_input import CommandOdometryInput


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variant',default='selected_8k');a=p.parse_args();torch.set_num_threads(1)
    cases=json.loads((a.evaluation/'cases.json').read_text())
    lidar=LidarSpec(json.loads((a.bundle/'hardware.json').read_text())['lidar']);rows=[]
    config=json.loads((a.bundle/'localization_contract.json').read_text())
    for case in cases:
        with np.load(a.evaluation/(case['id']+'_'+a.variant+'.npz')) as z:
            labels=z['labels'];scans=z['scans'];commands=z['commands']
        localizer=OdometryAnchoredLocalizer(StaticMap(ROOT/case['scene']/'decision_map.yaml',radius=.30),
            lidar,a.bundle/'localizer.pt',config)
        localizer.reset(np.array(case['start'])+case.get('initial_prior_offset',[0,0,0]),None)
        odom=CommandOdometryInput();reasons=Counter();errors=[];support=[];accepted=[]
        for i,(t,scan,command) in enumerate(zip(labels,scans,commands)):
            pose=localizer.update(odom.pose,scan,float(t[0]));error=float(np.linalg.norm(pose[:2]-t[1:3]))
            errors.append(error);support.append(localizer.static_support_fraction);reasons[localizer.correction_reason]+=1
            if localizer.scan_update_accepted:accepted.append(error)
            if i+1<len(labels):odom.advance(command,labels[i+1,0]-t[0])
        row=dict(case=case['id'],reasons=dict(reasons),xy_p95_m=float(np.percentile(errors,95)),end_error_m=errors[-1],
            support_median=float(np.median(support)),accepted_error_p95_m=float(np.percentile(accepted,95)) if accepted else None,
            end_std=localizer.std.tolist());rows.append(row);print(json.dumps(row),flush=True)
    a.output.write_text(json.dumps(dict(completed=True,scope='fixed executed-command localization replay, not closed-loop navigation',rows=rows),indent=2))


if __name__=='__main__':main()
