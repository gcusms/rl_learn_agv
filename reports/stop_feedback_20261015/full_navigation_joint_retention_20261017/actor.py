"""Identity-initialized, trainable navigation residual on a frozen full actor.

Unlike the previous recovery-only support, ordinary observable navigation is
trainable too. Safety permission remains the original sensor-derived mask.
The existing Native625OnPolicy likelihood implementation can train this head.
"""
import torch
import torch.nn.functional as F

from neural_navigation.masked_actions import ACTIONS


def contract():
    return dict(schema='native625_full_navigation_residual_v1',
        observation_dimensions=625,feature_dimensions=408,
        head_dimensions=[408,32,32,12],
        parent='complete_prior625_actor_not_a_439dim_ancestor',
        initial_behavior='zero_output_residual_plus_2times_actual_parent_choice',
        support='valid_current_prediction_and_scan_motion_available_goal_distance_over_0.3m',
        ordinary_navigation_is_trainable=True,
        original_current_action_mask_unchanged=True,stop_always_selectable=True,
        invalid_stop_only_and_terminal_docking_parent_exact=True,
        no_map_coordinates_case_clock_or_teacher_in_features=True,
        observations_and_execution_contract_unchanged=True,
        compatible_onpolicy='experiments.native625_onpolicy.Native625OnPolicy',
        no_runtime_reference_sequence=True)


class FullNavigationResidualActor(torch.nn.Module):
    def __init__(self,parent):
        super().__init__()
        self.parent=parent.eval()
        for parameter in self.parent.parameters():parameter.requires_grad_(False)
        self.register_buffer('actions',torch.from_numpy(ACTIONS.copy()))
        self.head=torch.nn.Sequential(torch.nn.Linear(408,32),torch.nn.Tanh(),
            torch.nn.Linear(32,32),torch.nn.Tanh(),torch.nn.Linear(32,12))
        torch.nn.init.zeros_(self.head[-1].weight)
        torch.nn.init.zeros_(self.head[-1].bias)

    def masks(self,observations):
        return torch.cat([torch.ones_like(observations[:,:1],dtype=torch.bool),
            observations[:,235:244]>.5,observations[:,267:269]>.5],dim=1)

    def support(self,observations):
        return ((observations[:,435]>.5)&(observations[:,438]>.5)
            &self.masks(observations)[:,1:].any(dim=1)
            &(torch.linalg.vector_norm(observations[:,218:220]*12.,dim=1)>.3))

    def features(self,observations):
        baseline=self.parent(observations)
        choice=(baseline[:,None]-self.actions[None]).abs().amax(2).argmin(1)
        one_hot=F.one_hot(choice,num_classes=12).to(observations.dtype)
        # Keep the established 408D context order and parent-choice slice.
        risk=torch.cat([observations[:,439:481],self.masks(observations).to(observations.dtype),
            observations[:,399:419],observations[:,433:435],one_hot],dim=1)
        features=torch.cat([risk,observations[:,144:216],observations[:,481:625],
            observations[:,216:224],observations[:,303:399]],dim=1)
        return features,baseline

    def logits(self,observations):
        features,_=self.features(observations)
        return self.head(features)+2.*features[:,76:88]

    def forward(self,observations):
        features,baseline=self.features(observations)
        logits=self.head(features)+2.*features[:,76:88]
        chosen=logits.masked_fill(~self.masks(observations),-1e8).argmax(1)
        return torch.where(self.support(observations)[:,None],self.actions[chosen],baseline)
