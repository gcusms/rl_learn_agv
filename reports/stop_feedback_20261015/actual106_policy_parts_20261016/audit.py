"""Decompose saved actor decisions on unchanged and causal-memory replays.

Read-only: recorded commands generated both histories. This audit neither
executes independent proposals nor treats newly legal motion as useful motion.
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from experiments.stop_recovery_policy import StopRecoveryPPO, masks_tensor
from neural_navigation.decision_feedback import GROUPS, STAT_SLICE
from neural_navigation.experiment_budget import ExperimentBudget


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'reports/stop_feedback_20261015'
MODEL = ROOT / 'models/stop_feedback_20261015/actual95_transfer_bc400_s231016'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + '\n')


def main():
    if (OUT / 'PLAN.json').exists():
        raise FileExistsError('Do not overwrite an actor-parts audit')
    paths = [Path(__file__), MODEL / 'policy_bc.zip', MODEL / 'demonstrations.npz',
        BASE / 'actual106_memory_context_20261016/replay.npz',
        BASE / 'actual106_memory_context_verified_20261016/RESULTS.json',
        BASE / 'actual95_learned_validation_20261016/physical/w11_training_stall_t106.npz',
        ROOT / 'training/experiments/stop_recovery_policy.py',
        ROOT / 'training/neural_navigation/decision_feedback.py']
    inputs = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    write('PLAN.json', dict(inputs=inputs, read_only=True, new_physical_steps=0,
        new_optimizer_updates=0, new_ppo_steps=0, soft_cpu_cores=.15,
        deployment_approved=False, goal_achieved=False))
    torch.set_num_threads(1)
    budget = ExperimentBudget(.15, memory_mib=2048, directory=OUT)
    model = StopRecoveryPPO.load(MODEL / 'policy_bc.zip', device='cpu')
    model.policy.set_training_mode(False)
    budget.tick()
    with np.load(paths[3], allow_pickle=False) as z:
        replay = {k: z[k].copy() for k in z.files}
    with np.load(paths[5], allow_pickle=False) as z:
        original = {k: z[k].copy() for k in ['observations', 'masks', 'labels']}
    selected = replay['times'] >= 106. - 1e-8
    assert selected.sum() == 170
    source = dict(original=original['observations'][selected],
                  confirmed_free=replay['observations'][selected])
    summaries, arrays = {}, dict(times=replay['times'][selected])
    for name, observations in source.items():
        base_logits, residuals, supports, probabilities, masks = [], [], [], [], []
        for obs in observations:
            train = np.r_[obs[:300], obs[303:]].astype(np.float32)
            assert train.shape == (436,)
            x = torch.from_numpy(train[None])
            with torch.no_grad():
                features = model.policy.mlp_extractor.policy_net(x)
                base, support = model.policy.action_net.parts(features)
                head = model.policy.action_net
                residual = head.residual(features[:, head.base_width:-13])
                mask = masks_tensor(x)
                distribution = model.policy.get_distribution(x, mask.numpy()).distribution
                combined = base + support[:, None].to(base.dtype) * residual
                manual = combined.masked_fill(~mask, -1e8).softmax(-1)
                torch.testing.assert_close(manual, distribution.probs, atol=2e-6, rtol=2e-6)
            base_logits.append(base.numpy()[0])
            residuals.append(residual.numpy()[0])
            supports.append(bool(support[0]))
            masks.append(mask.numpy()[0])
            probabilities.append(distribution.probs.numpy()[0])
            budget.tick()
        base_logits, residuals, masks, probabilities = map(np.asarray,
            [base_logits, residuals, masks, probabilities])
        support = np.asarray(supports)
        choices = probabilities.argmax(1)
        legal_motion = masks[:, 1:].any(1)
        combined = base_logits + support[:, None] * residuals
        moving = np.where(masks, combined, -1e8).copy()
        moving[:, 0] = -1e8
        best_motion = moving.argmax(1)
        rows = []
        for i in sorted({0, 10, 20, 70, 120, 169, *np.flatnonzero(choices).tolist()}):
            stats = observations[i, 303+STAT_SLICE.start:303+STAT_SLICE.stop]
            k = int(best_motion[i])
            rows.append(dict(time_s=float(arrays['times'][i]), choice=int(choices[i]),
                support=bool(support[i]), legal=np.flatnonzero(masks[i]).tolist(),
                best_motion=k, probability_stop=float(probabilities[i, 0]),
                probability_best_motion=float(probabilities[i, k]),
                base_stop_minus_best_motion=float(base_logits[i, 0]-base_logits[i, k]),
                residual_stop_minus_best_motion=float(residuals[i, 0]-residuals[i, k]),
                previous_idle_with_choices_s=float(stats[3]*5.),
                prediction_valid=bool(stats[4]>.5), scan_valid=bool(stats[7]>.5),
                guard_groups={str(a): GROUPS[int(observations[i, 303+a*8:303+(a+1)*8].argmax())]
                              for a in [1, 3, 4, 5, 7, 9]}))
        summaries[name] = dict(frames=len(observations), supported=int(support.sum()),
            legal_motion_frames=int(legal_motion.sum()), stops=int((choices==0).sum()),
            choices={str(int(k)): int(v) for k, v in zip(*np.unique(choices, return_counts=True))},
            probability_stop_minimum=float(probabilities[:, 0].min()),
            probability_stop_median=float(np.median(probabilities[:, 0])), rows=rows)
        for key, value in dict(base_logits=base_logits, residuals=residuals, support=support,
                               masks=masks, probabilities=probabilities, choices=choices).items():
            arrays[name+'_'+key] = value
    np.testing.assert_array_equal(arrays['confirmed_free_choices'], replay['independent_model_proposals'][selected])
    np.savez_compressed(OUT / 'parts.npz', **arrays)
    for name, expected in inputs.items():
        assert digest(ROOT / name) == expected, name
    result = dict(completed=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        groups=summaries, all170_confirmed_free_choices_exactly_reproduced=True,
        parts_archive_sha256=digest(OUT / 'parts.npz'), resource=budget.statistics(),
        new_physical_steps=0, new_optimizer_updates=0, new_ppo_steps=0,
        hypotheses_tested=['Whether a closed residual gate explains stopping.',
            'Whether newly permitted actions are preferred by the saved actor.'],
        limitations=['Recorded commands generated the replay; proposals were not executed.',
            'A legal or probable command is not proof of a useful complete recovery.',
            'No feature ablation, teacher action or simulated truth entered the actor.'],
        deployment_approved=False, goal_achieved=False)
    write('RESULTS.json', result)
    print(json.dumps({k: {j: v for j, v in g.items() if j!='rows'}
                      for k, g in summaries.items()}), flush=True)


if __name__ == '__main__':
    os.chdir(ROOT)
    try:
        main()
    except Exception as error:
        write('FAILURE.json', dict(error=repr(error), automatically_restarted=False,
                                  goal_achieved=False, deployment_approved=False))
        raise
