"""Native-physics validation of repaired adapters, without additional training."""
import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from prototypes.narrow_training_env import NarrowTrainingEnv
from prototypes.narrow_ppo import PortableFineActor
from prototypes.narrow_observation import masks,ACTIONS,contract
from prototypes.train_narrow_long import CASES


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-steps',type=int,default=250)
    p.add_argument('--cases',nargs='+',choices=[c['id'] for c in CASES],default=[c['id'] for c in CASES])
    p.add_argument('--run',type=Path,default=Path('training/runs/narrow_long_50000_v2_20261013'))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'status.json').exists():p.error('Evaluation output already used')
    if len(set(args.cases))!=len(args.cases):p.error('Duplicate evaluation cases')
    cases=[next(c for c in CASES if c['id']==name) for name in args.cases]
    torch.set_num_threads(1)
    model=MaskablePPO.load(args.run/'policy.zip',device='cpu')
    actor=PortableFineActor(model.policy).eval()
    scene=args.run/'scenes/gap_150';bundle=Path('models/recovery_20261012/portable')
    names=['prototypes/narrow_training_env.py','prototypes/narrow_ppo.py','prototypes/narrow_fine_runtime.py',
        'prototypes/narrow_recovery_guard.py','prototypes/narrow_alignment.py','prototypes/anchored_narrow_runtime.py',
        'prototypes/narrow_observation.py','prototypes/narrow_passage.py','training/neural_navigation/core.py','prototypes/evaluate_wait_repair.py',
        'prototypes/surface_localizer.py','prototypes/entrance_staging.py','prototypes/entrance_guard.py','training/neural_navigation/models.py']
    sources={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in names}
    for name in names:
        target=args.output/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(Path(name).read_bytes())
    status=dict(status='running',phase='native_evaluation',started_unix=time.time(),
        additional_training_steps=0,adapter_changes=True,source_sha256=sources,
        parent_model_sha256=hashlib.sha256((args.run/'policy.zip').read_bytes()).hexdigest(),
        published_actor_sha256=hashlib.sha256((bundle/'actor.pt').read_bytes()).hexdigest(),
        cases=[],requested_case_order=args.cases,deployment_approved=False,checks={})
    def save():
        tmp=args.output/'status.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(args.output/'status.json')
    save()
    env=None
    try:
        env=NarrowTrainingEnv(scene,bundle,seed=cases[0]['seed'],max_steps=args.max_steps)
        for case in cases:
            rows=[];counts=Counter();started=time.monotonic()
            obs,_=env.reset(options=dict(start=case['start'],goal=case['goal'],case_seed=case['seed']))
            if not status['checks']:
                exported=torch.jit.trace(actor,torch.from_numpy(obs)[None])
                exported.save(str(args.output/'actor_adapter_untrained.pt'))
                (args.output/'observation_action_contract.json').write_text(json.dumps(contract(),indent=2))
                model.save(args.output/'policy_adapter_untrained.zip')
                status['checks']['exported']=True
            for index in range(args.max_steps):
                if index%25==0:
                    status.update(current_case=case['id'],current_frame=index);save()
                mask=env.action_masks();choice,_=model.predict(obs,action_masks=mask,deterministic=True);choice=int(choice)
                with torch.no_grad():out=exported(torch.from_numpy(obs)[None])[0].numpy()
                np.testing.assert_allclose(out,ACTIONS[choice],rtol=0,atol=2e-8)
                core=env.runtime.core;state=core.narrow_route_state
                counts[state]+=1;counts['only_stop']+=int(not mask[1:].any())
                counts['stop_with_alternatives']+=int(choice==0 and mask[1:].any())
                counts['forward_available']+=int(np.any(mask&(ACTIONS[:,0]>0)))
                before=obs[216:218].tolist();path=core.predicted_path[:6].tolist()
                obs,reward,done,truncated,info=env.step(choice)
                if info['nonzero_choice_changed_by_guard']:raise RuntimeError('Training/execution mask disagreement')
                rows.append(dict(frame=index,choice=choice,pose=core.estimate.tolist(),command=info['commanded_velocity'],
                    route_state_before=state,waypoint_local_normalized_before=before,legal_choices=np.flatnonzero(mask).tolist(),
                    route_prefix_before=path,previous_stop_reason=env.runtime.context.last_reason,
                    scan_update_accepted=core.localizer.scan_update_accepted,map_agreement=float(core.localizer.map_agreement),
                    truth_for_scoring_only=env.env.truth.tolist(),pose_error_m=float(np.linalg.norm(core.estimate[:2]-env.env.truth[:2]))))
                rows[-1].update(policy_target_kind=core.policy_target_kind,
                    surface_residual_m=core.localizer.surface_residual_m,
                    surface_support=core.localizer.surface_support,observable_rank=core.localizer.observable_rank,
                    localization_rejection=core.localizer.rejection_reason)
                if done or truncated:break
            before=json.loads((args.run/'evaluation_050000'/f"{case['id']}.json").read_text())['result']
            result=dict(info,case_id=case['id'],frames=len(rows),counts=dict(counts),
                elapsed_wall_s=time.monotonic()-started,timeout=bool(truncated),
                original_success=bool(before['verified_mission_success']),original_travel_m=float(before['travel_m']),
                maximum_pose_error_m=max(r['pose_error_m'] for r in rows))
            (args.output/f"{case['id']}.json").write_text(json.dumps(dict(case=case,samples=rows,result=result),indent=2))
            status['cases'].append(result);save();print('CASE',json.dumps(result),flush=True)
        status.update(status='complete',phase='finished',finished_unix=time.time(),
            source_hashes_unchanged=all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in sources.items()),
            published_actor_unchanged=hashlib.sha256((bundle/'actor.pt').read_bytes()).hexdigest()==status['published_actor_sha256'])
        status['checks'].update(export_parity_every_frame=True,selected_executed_action_parity=True)
    except BaseException as error:
        status.update(status='failed',phase='failed',finished_unix=time.time(),error=repr(error));raise
    finally:
        if env is not None:env.close()
        save()


if __name__=='__main__':main()
