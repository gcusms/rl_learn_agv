"""Native sensor-stream check: one fusion update per frame, no routing duplicate."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from navrl.geometry import ROOT
from neural_navigation.stress_env import StressNavigationEnv
from experiments.fresh_perception_runtime import FreshPerceptionRuntime


def main():
    parser=argparse.ArgumentParser()
    for name in ('bundle','trace','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Fresh output required')
    torch.set_num_threads(1)
    case=json.loads(args.trace.read_text())['case']
    scene=ROOT/case.get('scene','assets/room_navigation_scripted')
    env=StressNavigationEnv(case=case,components=args.bundle,scene_dir=scene,route_mode='shortcut',max_steps=700)
    rows=[]
    try:
        env.reset(seed=case['seed'],options={k:case[k] for k in ('start','goal','initial_prior_offset') if k in case})
        prior=np.asarray(case['start'])+np.asarray(case.get('initial_prior_offset',[0,0,0]))
        runtime=FreshPerceptionRuntime(args.bundle,scene/'decision_map.yaml',prior,case['goal'],'shortcut')
        previous=runtime.core.perception_update_count
        for index in range(30):
            result=runtime.step(env.scan,env.observed_odometry,env.measured_velocity,env.elapsed,capture_observation=True)
            count=runtime.core.perception_update_count
            # Reset has no sensor scan; every real frame creates one update.
            assert count-previous == 1, (index,previous,count)
            assert result['perception_update_time']==runtime.core.elapsed
            runtime.core._routing_dynamic_layer()
            assert runtime.core.perception_update_count==count
            rows.append(dict(timestamp=float(env.elapsed),core_time=float(runtime.core.elapsed),
                fusion_updates=count,last_route_time=float(runtime.core.last_route_time)))
            previous=count
            v,w=result['command']
            env.step(np.array([v/(.2 if v>=0 else .08),w/.5]))
            assert not env.collision_events and not env.hard_guard_activations
    finally:
        env.close()
    args.output.write_text(json.dumps(dict(verified=True,frames=len(rows),rows=rows,
        limitation='Thirty native frames in one development case; cadence check only, not navigation acceptance.'),indent=2))
    print('Verified native frame cadence:',len(rows),flush=True)


if __name__=='__main__':main()
