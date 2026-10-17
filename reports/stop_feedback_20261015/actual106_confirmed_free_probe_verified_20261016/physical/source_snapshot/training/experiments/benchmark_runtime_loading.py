"""Fresh-process local load comparison, not a target-board latency claim."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


PROBE='''import json,sys,time,resource
started=time.perf_counter()
import torch
torch.set_num_threads(1)
after_torch=time.perf_counter()
if sys.argv[1]=='portable_import':
    from neural_navigation.goal_pose_runtime import DockingRuntime
else:
    from experiments.docking_navigation import DockingRuntime
after_import=time.perf_counter()
from navrl.geometry import ROOT
from navrl.map import StaticMap
import numpy as np
map_path=ROOT/'assets/navigation_stage2_dataset/validation_00/decision_map.yaml'
m=StaticMap(map_path,radius=.3)
pose=[*m.xy(np.argwhere(m.free)[0]),0.]
runtime=DockingRuntime(sys.argv[2],map_path,pose,pose,'shortcut')
after_construct=time.perf_counter()
runtime.core.set_goal([*pose[:2],.5])
print(json.dumps(dict(mode=sys.argv[1],torch_import_s=after_torch-started,module_import_s=after_import-after_torch,construct_s=after_construct-after_import,goal_update_s=time.perf_counter()-after_construct,total_s=time.perf_counter()-started,rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,training_imported=any(x in sys.modules for x in ('gymnasium','stable_baselines3','sb3_contrib','mujoco')))))
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=[]
    for mode in ('legacy_training_import','portable_import'):
        result=subprocess.run([sys.executable,'-c',PROBE,mode,str(a.bundle.resolve())],check=True,capture_output=True,text=True,env=os.environ.copy())
        rows.append(json.loads(result.stdout.strip().splitlines()[-1]))
    a.output.write_text(json.dumps(dict(scope='local CPU, fresh Python processes, cached filesystem; not Jetson measurement',rows=rows),indent=2))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
