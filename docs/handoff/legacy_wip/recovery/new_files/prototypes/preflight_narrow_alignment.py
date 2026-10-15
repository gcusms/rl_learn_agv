"""Bounded native-physics screening; neither trains nor replaces an actor."""
import argparse, hashlib, json, time
from pathlib import Path
from collections import Counter
import mujoco
import numpy as np
import torch
from navrl.map import StaticMap
from neural_navigation.stress_env import StressNavigationEnv
from neural_navigation.static_footprint import static_footprint_clear
from neural_navigation.masked_actions import ACTIONS, COMMANDS, masks_from_observation
from prototypes.narrow_passage import generate as straight_scenes, rotation_clear
from prototypes.narrow_junctions import generate as junction_scenes
from prototypes.anchored_narrow_runtime import AnchoredNarrowRuntime
from prototypes.narrow_alignment import AlignmentNarrowRuntime
from prototypes.narrow_observation import augment, contract


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bundle',type=Path,default=Path('models/recovery_20261012/portable'))
    p.add_argument('--steps',type=int,default=110);args=p.parse_args()
    if args.output.exists() or not 1<=args.steps<=700:p.error('Fresh output and 1..700 steps required')
    args.output.mkdir(parents=True);torch.set_num_threads(1)
    hardware=json.loads((args.bundle/'hardware.json').read_text())
    source=list(Path('prototypes').glob('*narrow*.py'))+[Path('training/neural_navigation/core.py'),Path('training/neural_navigation/runtime.py')]
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in source}
    for f in source:
        dest=args.output/'source_snapshot'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(f.read_bytes())
    report=dict(status='running',started_unix=time.time(),geometry=[],cases=[],source_sha256=hashes,
        actor_sha256=hashlib.sha256((args.bundle/'actor.pt').read_bytes()).hexdigest(),
        new_observation_contract=contract(),training_started=False,deployment_approved=False,
        limitation='Short development screening only; pivot rotation failure does not prove a rolling turn impossible. Fine actions are probed, not executed by the old 12-action actor.')
    def save():
        temp=args.output/'RESULTS.tmp';temp.write_text(json.dumps(report,indent=2,allow_nan=False));temp.replace(args.output/'RESULTS.json')
    save()
    try:
        for scene in junction_scenes(args.output/'junctions',hardware):
            directory=Path(scene['scene']);model=mujoco.MjModel.from_xml_path(str(directory/'physical_scene.xml'))
            m=StaticMap(directory/'decision_map.yaml',radius=.3)
            row=dict(scene,xml_loaded=True,body_count=model.nbody,
                start_clear=static_footprint_clear(m,scene['start'],hardware['robot_footprint_m'],.04),
                pivot_quarter_turn_clear=rotation_clear(m,scene['pivot'],np.pi/2,0,hardware['robot_footprint_m'],.04))
            report['geometry'].append(row);save()
        scene=straight_scenes(args.output/'straight',hardware)[0]
        case=dict(id='narrow_150_alignment_screen',category='narrow_development',scene=str(Path(scene['scene']).resolve()),seed=113150,start=scene['start'],goal=scene['goal'])
        for name,cls in [('anchored',AnchoredNarrowRuntime),('alignment_information',AlignmentNarrowRuntime)]:
            env=StressNavigationEnv(case=case,components=args.bundle,scene_dir=Path(case['scene']),route_mode='shortcut',max_steps=args.steps)
            rows=[];states=Counter();errors=[];success=False;fine_available=0;corridor_supported=0;probe_frames=0
            try:
                env.reset(seed=case['seed'],options=dict(start=case['start'],goal=case['goal']))
                runtime=cls(args.bundle,Path(case['scene'])/'decision_map.yaml',case['start'],case['goal'],'shortcut')
                for index in range(args.steps):
                    result=runtime.step(env.scan,env.observed_odometry,env.measured_velocity,env.elapsed,capture_observation=True)
                    obs=np.asarray(result['policy_observation'],np.float32);mask=masks_from_observation(obs[:282])
                    distance=np.max(np.abs(ACTIONS-result['proposed_action']),axis=1);choice=int(distance.argmin())
                    if distance[choice]>2e-6 or not mask[choice]:raise ValueError('Unknown or masked actor action')
                    if choice and result['stop_reason']!='estimated_goal_reached':np.testing.assert_allclose(result['command'],COMMANDS[choice],atol=2e-8,rtol=0)
                    diagnostic=runtime.core.diagnostics();state=diagnostic.get('narrow_route_state','legacy');states[state]+=1
                    row=dict(timestamp=float(env.elapsed),choice=choice,command=result['command'],estimate=runtime.core.estimate.tolist(),route_state=state,stop_reason=result['stop_reason'])
                    if name=='alignment_information' and index%5==0:
                        # An additional read-only observation cannot change live memory/guard state.
                        before=runtime.guard.reverse_started
                        extended=augment(obs,runtime.core,runtime.guard,env.elapsed)
                        assert runtime.guard.reverse_started==before
                        scan_lost=augment(obs,runtime.core,runtime.guard,env.elapsed,scan_valid=False)
                        assert not scan_lost[300:308].any()
                        stale=augment(obs,runtime.core,runtime.guard,env.elapsed,received_timestamp=env.elapsed+1)
                        assert not stale[300:308].any()
                        row['fine_allowed']=extended[300:308].astype(bool).tolist();row['scan_corridor']=extended[308:314].tolist()
                        fine_available+=bool(extended[300:308].any());corridor_supported+=bool(extended[308]);probe_frames+=1
                    rows.append(row);errors.append(float(np.linalg.norm(runtime.core.estimate[:2]-env.truth[:2])))
                    if result['stop_reason']=='estimated_goal_reached':success=np.linalg.norm(env.truth[:2]-env.goal)<=.2;break
                    env.step(np.array([result['command'][0]/.2,result['command'][1]/.5]))
                    if env.collision_events or env.hard_guard_activations:break
                summary=dict(variant=name,frames=len(rows),simulated_seconds=float(env.elapsed),travel_m=float(env.travel),
                    safe_success=bool(success and not env.collision_events and not env.hard_guard_activations),collisions=env.collision_events,
                    physical_guards=env.hard_guard_activations,screening_limit_reached=bool(len(rows)==args.steps and not success),
                    route_states=dict(states),localization_error_p95_m=float(np.quantile(errors,.95)),
                    fine_probe_frames=probe_frames,fine_action_available_frames=fine_available,scan_corridor_supported_frames=corridor_supported)
                (args.output/f'{name}.json').write_text(json.dumps(dict(case=case,samples=rows,result=summary),allow_nan=False))
                report['cases'].append(summary);save();print(json.dumps(summary),flush=True)
            finally:env.close()
        report.update(status='complete',finished_unix=time.time(),source_hashes_unchanged=all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in hashes.items()))
    except BaseException as error:report.update(status='failed',error=repr(error));raise
    finally:save()

if __name__=='__main__':main()
