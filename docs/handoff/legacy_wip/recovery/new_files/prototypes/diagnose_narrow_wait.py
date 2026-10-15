"""Read-only replay diagnostics: route rejection, native probes and actor scores.
Counterfactual probes never enter the executed simulation commands.
"""
import argparse,copy,hashlib,json,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from neural_navigation.core import RoomRoutingMixin
from neural_navigation.static_footprint import static_footprint_clear
from prototypes.narrow_alignment import AlignmentNarrowCore
from prototypes.narrow_passage import rotation_clear
from prototypes.narrow_training_env import NarrowTrainingEnv
from prototypes.narrow_observation import COMMANDS
from prototypes.train_narrow_long import CASES

original_route=RoomRoutingMixin._neural_route

def capture_proposal(core):
    original_route(core)
    if isinstance(core,AlignmentNarrowCore):core.diagnostic_proposal=core.predicted_path.copy()


def probes(core,guard,commands,force_route_complete=False):
    view=copy.copy(core)
    if force_route_complete:
        diagnostics=core.diagnostics().copy();diagnostics['route_complete']=True
        view.diagnostics=lambda:diagnostics
    snapshot=copy.deepcopy(guard)
    if snapshot.memory:
        points,uncertainty=snapshot.memory.update(core)
        class Prepared:
            def update(self,unused):return points,uncertainty
            def reset(self):pass
        snapshot.memory=Prepared()
    result=[]
    for command in commands:
        executed,reason=copy.copy(snapshot).check(view,command,core.elapsed,core.elapsed,True,False)
        result.append(dict(command=np.asarray(command).tolist(),executed=executed.tolist(),reason=reason,
            accepted=bool(np.allclose(executed,command,rtol=0,atol=1e-9))))
    return result


def route_details(core):
    path=getattr(core,'diagnostic_proposal',core.predicted_path).copy()
    bad=[];yaw=float(core.estimate[2])
    for index,(a,b) in enumerate(zip(path[:-1],path[1:])):
        delta=b-a;length=np.linalg.norm(delta)
        if length<1e-8:continue
        heading=float(np.arctan2(delta[1],delta[0]))
        static=core._static_visible(a,b);segment=core._segment_clear(a,b)
        turn=rotation_clear(core.static_map,a,yaw,heading,core.footprint,core.margin)
        if not static or not segment or not turn:
            bad.append(dict(index=index,a=a.tolist(),b=b.tolist(),length_m=float(length),heading_rad=heading,
                from_yaw_rad=yaw,static_sweep_clear=bool(static),segment_clear=bool(segment),corner_turn_clear=bool(turn)))
        yaw=heading
    # See if skipping a tiny first decoded edge can retain the same candidate
    # tail while satisfying every original geometric and sensor route check.
    valid_skips=[]
    for index in range(1,min(12,len(path))):
        candidate=np.vstack([core.estimate[:2],path[index:]])
        if core._route_usable(candidate):valid_skips.append(index)
    direct=np.array([core.estimate[:2],core.goal])
    return dict(proposal_first_points=path[:8].tolist(),proposal_points=len(path),
        proposal_reaches_goal=bool(len(path)>1 and np.linalg.norm(path[-1]-core.goal)<.12),
        first_rejections=bad[:5],valid_skipped_prefix_indices=valid_skips,
        direct_route_usable=bool(core._route_usable(direct)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run',type=Path,default=Path('training/runs/narrow_long_50000_v2_20261013'))
    args=p.parse_args()
    if args.output.exists():p.error('Fresh output required')
    args.output.mkdir(parents=True);torch.set_num_threads(1)
    model=MaskablePPO.load(args.run/'policy.zip',device='cpu')
    scene=args.run/'scenes/gap_150';bundle=Path('models/recovery_20261012/portable')
    status=dict(status='running',started_unix=time.time(),model_sha256=hashlib.sha256((args.run/'policy.zip').read_bytes()).hexdigest(),
        diagnostic_only=True,counterfactual_commands_executed=False,cases=[])
    def save():
        tmp=args.output/'RESULTS.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(args.output/'RESULTS.json')
    save();RoomRoutingMixin._neural_route=capture_proposal
    try:
        for case in CASES[:4]:
            env=NarrowTrainingEnv(scene,bundle,seed=case['seed'],max_steps=250)
            rows=[];snapshots=[];counts=Counter()
            original=json.loads((args.run/'evaluation_050000'/f"{case['id']}.json").read_text())['samples']
            try:
                observation,_=env.reset(options=dict(start=case['start'],goal=case['goal'],case_seed=case['seed']))
                for index in range(80):
                    mask=env.action_masks();choice,_=model.predict(observation,action_masks=mask,deterministic=True);choice=int(choice)
                    core=env.runtime.core
                    counts['frames']+=1;counts[core.narrow_route_state]+=1
                    counts['only_stop']+=int(not mask[1:].any())
                    counts['stop_with_alternatives']+=int(choice==0 and mask[1:].any())
                    counts['forward_available']+=int(np.any(mask&(COMMANDS[:,0]>0)))
                    counts['fine_available']+=int(mask[12:].any())
                    counts['fine_chosen']+=int(choice>=12)
                    if index in (0,10,20,40,60,79):
                        x=torch.from_numpy(observation)[None]
                        with torch.no_grad():
                            logits=model.policy.mlp_extractor.forward_actor(x)[0].numpy()
                            net=model.policy.mlp_extractor.policy_net
                            teacher=net.teacher_head(net.teacher(x[:,:300]))[0].numpy()
                            probabilities=model.policy.get_distribution(x,action_masks=mask[None]).distribution.probs[0].numpy()
                        native=probes(core,env.runtime.guard,COMMANDS)
                        no_route=probes(core,env.runtime.guard,COMMANDS,True)
                        smaller=np.array([[.01,0],[.02,0],[0,.025],[0,-.025],[0,.05],[0,-.05],[.02,.05],[.02,-.05]])
                        snapshots.append(dict(frame=index,timestamp=float(core.elapsed),choice=choice,
                            estimate=core.estimate.tolist(),truth=env.env.truth.tolist(),std=core.localizer.std.tolist(),
                            waypoint_local=(observation[216:218]*2).tolist(),corridor_features=observation[308:314].tolist(),
                            current_static_pose_clear=bool(static_footprint_clear(core.static_map,core.estimate,core.footprint,core.margin)),
                            pose_error_m=float(np.linalg.norm(core.estimate[:2]-env.env.truth[:2])),
                            state=core.narrow_route_state,route_complete=core.diagnostics()['route_complete'],
                            route=route_details(core),legal_choices=np.flatnonzero(mask).tolist(),
                            logits=logits.tolist(),teacher_logits=teacher.tolist(),probabilities=probabilities.tolist(),
                            native_probes=native,diagnostic_route_bypass_probes=no_route,
                            smaller_probes=probes(core,env.runtime.guard,smaller),
                            smaller_route_bypass_probes=probes(core,env.runtime.guard,smaller,True)))
                    observation,reward,done,truncated,info=env.step(choice)
                    row=dict(choice=choice,pose=env.runtime.core.estimate.tolist(),command=info['commanded_velocity'])
                    expected=original[index]
                    if choice!=expected['choice'] or not np.allclose(row['pose'],expected['pose'],rtol=0,atol=1e-9):
                        raise RuntimeError(f"Replay diverged: {case['id']} frame {index}")
                    rows.append(row)
                    if done or truncated:break
                report=dict(case=case,counts=dict(counts),replay_matches_original=True,
                    travel_m=float(env.env.travel),snapshots=snapshots,samples=rows)
                path=args.output/f"{case['id']}.json";path.write_text(json.dumps(report,indent=2))
                status['cases'].append(dict(case_id=case['id'],counts=dict(counts),replay_matches_original=True,
                    report=str(path)));save();print(json.dumps(status['cases'][-1]),flush=True)
            finally:env.close()
        status.update(status='complete',finished_unix=time.time())
    except BaseException as error:status.update(status='failed',error=repr(error));raise
    finally:RoomRoutingMixin._neural_route=original_route;save()

if __name__=='__main__':main()
