"""Board-reusable single-stream inference benchmark; no physics/ROS processes.

Use actual board results, not desktop results, to accept an embedded deployment.
"""
import argparse
import json
import platform
import resource
import time
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from navrl.geometry import ROOT, LidarProfile
from navrl.map import StaticMap
from neural_navigation.models import LearnedLocalizer, NeuralRouter


def measure(operation, count):
    for _ in range(5): operation()
    elapsed=[]
    for _ in range(count):
        started=time.perf_counter();operation();elapsed.append((time.perf_counter()-started)*1000)
    return {'p50_ms':float(np.percentile(elapsed,50)), 'p95_ms':float(np.percentile(elapsed,95)),
            'p99_ms':float(np.percentile(elapsed,99)), 'max_ms':float(max(elapsed))}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--components',type=Path,required=True)
    parser.add_argument('--map',type=Path,default=ROOT/'assets/room_navigation_scripted/decision_map.yaml')
    parser.add_argument('--iterations',type=int,default=50)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.iterations<5 or args.threads<1:parser.error('iterations >=5 and threads >=1 required')
    if args.output.exists():parser.error('Use a fresh output report')
    torch.set_num_threads(args.threads)
    static_map=StaticMap(args.map,radius=.3);lidar=LidarProfile()
    localizer=LearnedLocalizer(static_map,lidar,args.components/'localizer.pt')
    checkpoint=torch.load(args.components/'router.pt',map_location='cpu',weights_only=True)
    router=NeuralRouter(checkpoint['iterations']);router.load_state_dict(checkpoint['state_dict']);router.eval()
    policy=PPO.load(args.policy,device='cpu')
    h,w=static_map.free.shape;free=static_map.free[2:h//4*4:4,2:w//4*4:4].astype(np.float32)
    x=np.zeros((2,4,*free.shape),np.float32);x[:,0]=free
    r,c=np.argwhere(free)[-1];x[:,2,r,c]=1.
    r,c=np.argwhere(free)[0];x[:,3,r,c]=1.
    inputs=torch.from_numpy(x)
    # Synthetic valid endpoints exercise feature construction; no accuracy claim.
    scan=np.full(lidar.beams,2.,np.float64)
    observation=np.zeros(policy.observation_space.shape,np.float32)
    def pose():
        localizer.reset([1.2,1.2,0.],None);return localizer.update(np.zeros(3),scan)
    def route():
        with torch.no_grad():return router(inputs)
    def action():return policy.predict(observation,deterministic=True)
    def combined():pose();route();action()
    report={'platform':platform.platform(),'machine':platform.machine(),'torch':torch.__version__,
            'threads':args.threads,'iterations':args.iterations,'grid_shape':list(free.shape),
            'localizer':measure(pose,args.iterations),'router_uncached':measure(route,args.iterations),
            'policy':measure(action,args.iterations),'combined_uncached':measure(combined,args.iterations),
            'peak_process_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'parameter_counts':{'localizer':sum(p.numel() for p in localizer.network.parameters()),
                                'router':sum(p.numel() for p in router.parameters()),
                                'ppo_policy_including_value_head':sum(p.numel() for p in policy.policy.parameters())},
            'limitations':['Synthetic observations; timing only, not navigation accuracy',
                           'Excludes scan transport, temporal map fusion, route extraction, visibility checks and motor IO',
                           'Process RSS includes training libraries loaded by SB3, not a stripped deployment runtime',
                           'Results apply only to the machine executing this command; not extrapolated to Orin Nano']}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
