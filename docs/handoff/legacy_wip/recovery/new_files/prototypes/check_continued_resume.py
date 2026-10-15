"""Audit learned recovery checkpoint preservation, without rollout or training."""
import json
from pathlib import Path
import torch
from sb3_contrib import MaskablePPO
from prototypes.check_continued_migration import InspectionInterface
from prototypes.continued_recovery_policy import warm_start_continued


def main():
    torch.set_num_threads(1)
    base=Path(__file__).resolve().parents[1]/'training/runs/safety_context_20261011'
    checkpoint=base/'focused_prior_control_8000/candidate/policy.zip'
    original=MaskablePPO.load(checkpoint,device='cpu')
    migrated=warm_start_continued(checkpoint,InspectionInterface(original.training_reward_scale))
    original_params=dict(original.policy.named_parameters())
    state_counts={'recovery':0,'critic':0}
    for name,param in migrated.policy.named_parameters():
        reference=original_params[name]
        torch.testing.assert_close(param,reference,atol=0,rtol=0)
        left=original.policy.optimizer.state.get(reference,{})
        right=migrated.policy.optimizer.state.get(param,{})
        assert left.keys()==right.keys()
        if left:
            role='recovery' if '.residual.' in name else 'critic'
            state_counts[role]+=1
        for key,value in left.items():
            if torch.is_tensor(value):torch.testing.assert_close(value,right[key],atol=0,rtol=0)
            else:assert value==right[key]
    assert state_counts=={'recovery':6,'critic':6}
    assert original.policy.recovery_prior_scale==migrated.policy.recovery_prior_scale
    for key in ('gamma','gae_lambda','n_steps','n_epochs','batch_size','ent_coef'):
        assert getattr(original,key)==getattr(migrated,key)
    assert not any(param.requires_grad for param in migrated.policy.mlp_extractor.policy_net.base.parameters())
    assert not any(param.requires_grad for param in migrated.policy.action_net.base.parameters())
    report=dict(source=str(checkpoint),all_learned_weights_exact=True,
        learned_optimizer_states_exact=state_counts,prior_scale_preserved=migrated.policy.recovery_prior_scale,
        ppo_hyperparameters_preserved=True,base_actor_frozen=True,
        old_training_rng_resume_claimed=False,training_or_simulation_executed=False,
        limitation='Preserves trained parameters and optimizer history only. Expanded support can change commands in newly enabled states; no recovery improvement or safety proved.')
    (base/'CONTINUED_TRAINED_CHECKPOINT_RESUME_CHECK.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
