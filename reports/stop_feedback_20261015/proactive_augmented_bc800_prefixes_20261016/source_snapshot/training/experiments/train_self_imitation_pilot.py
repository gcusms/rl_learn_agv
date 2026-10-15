"""Bounded offline self-imitation pilot on verified RL training experience.

No environment steps are claimed during auxiliary updates; deterministic
closed-loop replay decides whether to proceed with regression diagnostics.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from experiments.quality_masked_navigation import QualityAnchoredPPO
from experiments.self_imitation import validate_demonstration,losses
from experiments.train_quality_navigation import export,evaluate,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--demonstration',type=Path,required=True);p.add_argument('--source-trace',type=Path,required=True)
    p.add_argument('--training-log',type=Path,required=True);p.add_argument('--failure-pool',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=300)
    a=p.parse_args()
    if a.output.exists() or not 1<=a.updates<=1000:p.error('Fresh output and 1..1000 updates required')
    data=json.loads(a.demonstration.read_text());source_trace=json.loads(a.source_trace.read_text())
    if data['trace_sha256']!=digest(a.source_trace) or data['training_log_sha256']!=digest(a.training_log):
        p.error('Demonstration provenance hashes differ')
    records=[json.loads(x) for x in a.training_log.read_text().splitlines()]
    matches=[r for r in records if r.get('step')==data['training_step'] and 'worker' in r and 'training_case_json' in r]
    if len(matches)!=1 or json.loads(matches[0]['training_case_json'])!=data['case'] or source_trace['case']!=data['case']:
        p.error('Original training case mismatch')
    pool=json.loads(a.failure_pool.read_text())
    if pool.get('kind')!='training':p.error('Training-only templates required')
    a.checkpoint=a.checkpoint.resolve();a.output=a.output.resolve();torch.set_num_threads(1)
    model=QualityAnchoredPPO.load(a.checkpoint,device='cpu')
    if model.observation_space.shape!=(300,):raise ValueError('Quality actor required')
    obs,actions,masks,returns=validate_demonstration(data,model.gamma)
    if len(obs)!=len(source_trace['samples']):raise ValueError('Demonstration length differs from source')
    np.testing.assert_allclose(obs,[r['policy_observation'] for r in source_trace['samples']],atol=2e-6,rtol=0)
    np.testing.assert_array_equal(actions,[r['choice'] for r in source_trace['samples']])
    np.testing.assert_array_equal(masks,[r['mask'] for r in source_trace['samples']])
    tensors=[torch.as_tensor(x,device=model.device) for x in (obs,actions,masks,returns)]
    a.output.mkdir();components=a.output/'candidate';components.mkdir()
    for name in ('hardware.json','localizer.pt','router.pt','quality_observation.json'):
        shutil.copy2(a.checkpoint.parent/name,components/name)
    project=Path(__file__).resolve().parents[2];hashes={}
    for path in (project/'training').rglob('*.py'):
        relative=path.relative_to(project)
        if 'runs' in relative.parts:continue
        hashes[str(relative)]=digest(path);target=components/'code_snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    config=dict(parent_policy_sha256=digest(a.checkpoint),demonstration_sha256=digest(a.demonstration),
        source_trace_sha256=digest(a.source_trace),transitions=len(obs),requested_auxiliary_updates=a.updates,
        new_ppo_environment_steps=0,actor_critic_loss='positive-advantage SIL, Oh et al. ICML2018 eq2/3',
        value_coefficient=.1,reference_strength=2.,reference_kl_budget=.02,learning_rate=3e-5,
        source_sha256=hashes,guard_unchanged=True,reward_unchanged=True,deployment_approved=False,
        regression_seeds=list(range(938000,938008)),
        limitation='One recorded training episode; auxiliary SIL warm start, not full paper replay/prioritization or deployment.')
    (components/'config.json').write_text(json.dumps(config,indent=2))
    status=dict(status='running',phase='self_imitation',started_unix=time.time())
    def save():(a.output/'status.json').write_text(json.dumps(status,indent=2))
    save();model.reference_strength=2.;model.reference_kl_budget=.02;model.snapshot_reference()
    model.save(components/'policy_initial.zip');export(components/'policy_initial.zip',components,a.output/'initial_portable')
    for group in model.policy.optimizer.param_groups:group['lr']=3e-5
    model.policy.set_training_mode(True);accepted=0;rolled_back=0;history=[]
    try:
        for step in range(a.updates):
            actor,critic,stats=losses(model.policy,*tensors)
            if not stats['positive_advantage_states']:break
            reference,_=model.reference_loss(tensors[0],tensors[2])
            before={k:v.detach().clone() for k,v in model.policy.state_dict().items()}
            optimizer=copy.deepcopy(model.policy.optimizer.state_dict())
            loss=actor+.1*critic+2.*reference
            model.policy.optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(),model.max_grad_norm)
            model.policy.optimizer.step()
            with torch.no_grad():post,_=model.reference_loss(tensors[0],tensors[2])
            if not torch.isfinite(post) or post.item()>.02:
                model.policy.load_state_dict(before);model.policy.optimizer.load_state_dict(optimizer)
                rolled_back+=1;break
            accepted+=1
            if step%20==0 or step==a.updates-1:
                history.append(dict(update=accepted,actor_loss=float(actor.detach()),critic_loss=float(critic.detach()),
                    reference_kl=float(post),**stats))
                (a.output/'updates.json').write_text(json.dumps(history,indent=2))
        model.self_imitation_updates=accepted
        model.save(components/'policy.zip');export(components/'policy.zip',components,a.output/'portable')
        config.update(accepted_auxiliary_updates=accepted,rolled_back_updates=rolled_back)
        (components/'config.json').write_text(json.dumps(config,indent=2));results=[]
        for index,task in enumerate([data['case']]+config['regression_seeds']):
            status.update(phase='matched_diagnostic',completed_cases=len(results));save();pair={}
            for label,name in [('initial','policy_initial.zip'),('final','policy.zip')]:
                pair[label]=evaluate(components/name,components,pool['templates'],task,a.output/f'{label}_{index}.json')
            results.append(pair)
            (a.output/'MATCHED_RESULTS.json').write_text(json.dumps(dict(results=results,completed=len(results)==9),indent=2))
            if index==0 and not pair['final']['safe_success']:break
        status.update(status='complete',finished_unix=time.time(),history_task_solved=results[0]['final']['safe_success'],
            regression_completed=len(results)==9,development_gate_passed=len(results)==9 and all(x['final']['safe_success'] for x in results));save()
    except Exception as error:
        status.update(status='failed',error=str(error));save();raise


if __name__=='__main__':main()
