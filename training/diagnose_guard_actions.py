"""Probe a saved sensor/guard snapshot; truth is used only for report labels.

This does not send commands, alter the policy, or prove moving-obstacle safety.
"""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from navrl.map import StaticMap
from neural_navigation.runtime import EmergencyStop
from neural_navigation.sensors import LidarSpec


def main():
    p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh diagnostic output required')
    data=json.loads(a.trace.read_text());hardware=json.loads((a.bundle/'hardware.json').read_text())
    scene=Path(data['case']['scene'])
    if not scene.is_absolute():scene=Path(__file__).resolve().parent/scene
    static_map=StaticMap(scene/'decision_map.yaml',radius=.3)
    results=[];commands=[[v,w] for v in [0.,.02,.05,.1,.2] for w in [-.5,-.25,0.,.25,.5]]
    for index in [50,100,200,400,650]:
        if index>=len(data['samples']):continue
        r=data['samples'][index];d=r['diagnostics']
        core=SimpleNamespace(elapsed=r['timestamp'],odom=np.array(r['odometry']),estimate=np.array(r['estimate']),
                             scan=np.array([np.nan if v is None else v for v in r['scan']]),lidar=LidarSpec(hardware['lidar']),
                             localizer=SimpleNamespace(std=np.array(d['localization_std'])),odometry_sanity_failure=d['odometry_sanity_failure'],
                             static_map=static_map,diagnostics=lambda:d)
        guard=EmergencyStop(hardware['robot_footprint_m'],memory_config=hardware.get('near_field_memory'))
        if guard.memory:
            for entry in r['guard_memory']:
                point=np.array(entry['point']);key=tuple(np.floor(point/guard.memory.config.get('resolution_m',.04)).astype(int))
                guard.memory.entries[key]=(point,entry['last_seen'])
        probes=[]
        for command in commands:
            executed,reason=copy.deepcopy(guard).check(core,command,core.elapsed,core.elapsed)
            probes.append(dict(requested=command,executed=executed.tolist(),reason=reason,
                               fully_accepted=bool(np.allclose(executed,command,atol=1e-9,rtol=0))))
        results.append(dict(index=index,timestamp=r['timestamp'],policy_request=r['model_command'],policy_execution=r['command'],
                            original_reason=r['stop_reason'],world_gap_report_only=r['world_min_gap_m'],probes=probes))
    a.output.write_text(json.dumps(dict(case_id=data['case']['id'],samples=results,
                                     limitation='Counterfactual snapshot only, static measured endpoints and sample-based guard, not a new navigation controller or closed-loop proof'),indent=2))
    for r in results:print(r['index'],'accepted',[p['requested'] for p in r['probes'] if p['fully_accepted'] and any(p['requested'])])


if __name__=='__main__':main()
