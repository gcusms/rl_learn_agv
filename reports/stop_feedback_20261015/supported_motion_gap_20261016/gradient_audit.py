"""Measure stop saturation and actor/critic gradients without optimizing."""
from datetime import datetime
import gc
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
import torch.nn.functional as F

from experiments.conditional_recovery_ppo import actor_terms
from experiments.stop_recovery_policy import StopRecoveryPPO


ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
MODELS=ROOT/'models/stop_feedback_20261015'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gradient_norms(loss,parameters):
    gradients=torch.autograd.grad(loss,[p for _,p in parameters],allow_unused=True,retain_graph=True)
    sums=dict(actor=0.,critic=0.)
    for (name,_),gradient in zip(parameters,gradients):
        if gradient is None:continue
        group='actor' if name.startswith(('mlp_extractor.policy_net.residual.','action_net.residual.')) else 'critic'
        sums[group]+=float(gradient.detach().double().square().sum())
    return {k:math.sqrt(v) for k,v in sums.items()}


def main():
    if (OUT/'GRADIENTS.json').exists():raise FileExistsError('Do not replace a completed audit')
    torch.set_num_threads(1)
    inputs={str(Path(__file__).relative_to(ROOT)):digest(Path(__file__))}
    rows=[]
    for name in ['w11_control2000_s231015','w11_retained2000_s231015']:
        for step in [1000,2000]:
            folder=MODELS/name
            checkpoint=folder/('policy_step_'+str(step)+'.zip')
            path=folder/'rollouts'/('rollout_'+str(step).zfill(6)+'.npz')
            for p in [checkpoint,path]:inputs[str(p.relative_to(ROOT))]=digest(p)
            model=StopRecoveryPPO.load(checkpoint,device='cpu')
            model.policy.set_training_mode(False)
            parameters=[(n,p) for n,p in model.policy.named_parameters() if p.requires_grad]
            with np.load(path,allow_pickle=False) as data:
                x=torch.from_numpy(data['observations'].reshape(-1,436).copy())
                masks=data['action_masks'].reshape(-1,12).astype(bool)
                choices=torch.from_numpy(data['sampled_actions'].reshape(-1).astype(np.int64))
                old=torch.from_numpy(data['old_log_prob'].reshape(-1).copy())
                advantages=torch.from_numpy(data['advantages'].reshape(-1).copy())
                returns=torch.from_numpy(data['returns'].reshape(-1).copy())
            values,logp,entropy=model.policy.evaluate_actions(x,choices,action_masks=masks)
            parity=float((logp-old).abs().max())
            if parity>5e-5:raise ValueError('Pre-update checkpoint does not reproduce stored likelihoods')
            pg,ent,kl,fraction,count=actor_terms(x,logp,old,advantages,entropy,.2,model.policy.recovery_support)
            vf=F.mse_loss(returns,values.flatten())
            total=pg+model.ent_coef*ent+model.vf_coef*vf
            retention_value=0.;retention_samples=0
            if getattr(model,'demonstration_retention_contract',None) is not None and count:
                from experiments.demonstration_retention import loss as retention_loss
                retained,retention_samples=retention_loss(model)
                retention_value=float(retained.detach())
                total=total+model.demonstration_retention_contract['coefficient']*retained
            norms=gradient_norms(total,parameters)
            combined=math.hypot(norms['actor'],norms['critic'])
            clip_factor=min(1.,model.max_grad_norm/(combined+1e-6))
            active=model.policy.recovery_support(x).numpy()
            late=active&(x[:,431].numpy()*5.>=2.-1e-6)
            rows.append(dict(model=name,pre_update_checkpoint_step=step,
                source_rollout=str(path.relative_to(ROOT)),diagnostic_full_rollout_frames=len(x),
                actual_minibatch_order_replayed=False,old_log_probability_max_abs_error=parity,
                recovery_actor_frames=count,previous_legal_wait_ge2s_frames=int(late.sum()),
                sampled_stop_on_late_frames=int((late&(choices.numpy()==0)).sum()),
                policy_gradient_loss=float(pg.detach()),entropy_loss=float(ent.detach()),
                value_loss=float(vf.detach()),retention_loss=retention_value,retention_samples=retention_samples,
                gradient_norms=norms,joint_gradient_clip_factor=clip_factor,
                actor_norm_after_joint_clip=norms['actor']*clip_factor,
                actor_policy_gradient_norms=gradient_norms(pg,parameters),
                actor_entropy_gradient_norms=gradient_norms(model.ent_coef*ent,parameters)))
            del model,x,values,logp,entropy,pg,ent,vf,total,parameters
            if retention_samples:del retained
            gc.collect()
    trace=ROOT/'reports/stop_feedback_20261015/w11_retained2000_w11_20261016/w11_training_stall_t106.npz'
    inputs[str(trace.relative_to(ROOT))]=digest(trace)
    with np.load(trace,allow_pickle=False) as data:
        i=int(np.flatnonzero(data['model_controlled'])[0]);raw=data['observations'][i]
        x=torch.from_numpy(np.r_[raw[:300],raw[303:]].astype(np.float32))[None]
        masks=torch.from_numpy(data['masks'][i].astype(bool))[None]
    stops=[]
    for name,filename in [('reference','w11_recovery_bc800_s231015/policy_bc.zip'),
            ('control','w11_control2000_s231015/policy.zip'),
            ('retained','w11_retained2000_s231015/policy.zip'),
            ('diagnostic_bc','early_transfer_bc800_s231015/policy_bc.zip')]:
        path=MODELS/filename;inputs[str(path.relative_to(ROOT))]=digest(path)
        model=StopRecoveryPPO.load(path,device='cpu');model.policy.set_training_mode(False)
        with torch.no_grad():
            features=model.policy.mlp_extractor.policy_net(x)
            base,supported=model.policy.action_net.parts(features)
            logits=model.policy.action_net(features).double().masked_fill(~masks,-1e8)
            motion=logits.masked_fill(~(masks&torch.arange(12)[None].ne(0)),-1e8)
            p=float(torch.exp(torch.logsumexp(motion,-1)-torch.logsumexp(logits,-1)))
            alternative=int(motion.argmax(-1))
            p8=float(torch.exp(torch.logsumexp(motion/8.,-1)-torch.logsumexp(logits/8.,-1)))
            stops.append(dict(model=name,legal_actions=np.flatnonzero(masks.numpy()[0]).tolist(),
                recovery_support=bool(supported[0]),frozen_base_stop_minus_motion=float(base[0,0]-base[0,alternative]),
                learned_stop_minus_motion=float(logits[0,0]-logits[0,alternative]),
                motion_probability_float64=p,expected_motion_draws_per_million_same_state=p*1e6,
                any_motion_chance_per250_same_state=-math.expm1(250*math.log1p(-p)),
                hypothetical_temperature8_motion_probability=p8,
                hypothetical_temperature_not_applied=True))
        del model;gc.collect()
    report=dict(completed=True,at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        whole_rollout_diagnostic_gradients=rows,actual_t106_same_input_saturation=stops,inputs=inputs,
        no_optimizer_step=True,no_saved_model_or_rng_modified=True,new_physical_steps=0,new_ppo_steps=0,
        scope='Stored likelihood parity and whole-rollout gradient diagnostic, not exact minibatch/optimizer replay. Temperature8 is an unapplied illustration.',
        goal_achieved=False,deployment_approved=False)
    (OUT/'GRADIENTS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='inputs'},indent=2))


if __name__=='__main__':main()
