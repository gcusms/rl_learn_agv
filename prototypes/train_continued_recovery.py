"""Opt-in continued recovery experiment; never replaces the deployed actor.

Run with PYTHONPATH=training:. and the same arguments as train_quality_navigation.
Keep this outside training/ until the live matched trials have completed.
"""
import hashlib
import json
from pathlib import Path
import sys
import shutil
from collections import Counter

import numpy as np
import torch
import experiments.train_alignment_recovery as alignment
import experiments.train_focused_recovery as focused
import experiments.train_quality_navigation as trainer
from prototypes.continued_recovery_policy import continued_support, warm_start_continued


class ContinuedProgress(alignment.AlignmentProgress):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counts = Counter()
        self.reasons = Counter()

    def _on_step(self):
        observations = np.asarray(self.model._last_obs)
        masks = np.asarray(self.locals['action_masks'])
        actions = np.asarray(self.locals['actions']).reshape(-1)
        # Use exactly the actor's float32 predicate, including boundary values.
        with torch.no_grad():
            support = continued_support(torch.as_tensor(observations,
                device=self.model.device),
                self.model.policy.continuation_alignment_cos).cpu().numpy()
        for observation, mask, action, enabled in zip(observations, masks, actions, support):
            if not mask[action]:
                raise RuntimeError('Sampled action violates its native mask')
            self.counts['total_decisions'] += 1
            if enabled:
                self.counts['recovery_decisions'] += 1
                self.counts[f'action_{int(action)}'] += 1
                self.reasons[str(int(np.argmax(observation[250:266])))] += 1
        result = super()._on_step()
        if self.num_timesteps % 250 == 0:
            self.save_counts()
            if shutil.disk_usage(self.output).free < 100*1024**2:
                self.model.save(self.output/'policy_low_disk.zip')
                raise RuntimeError('Low disk: checkpoint saved; stop instead of filling disk')
        return result

    def save_counts(self):
        (self.output/'continued_recovery_samples.json').write_text(json.dumps(dict(
            counts=dict(self.counts), previous_stop_reasons=dict(self.reasons),
            predicate='actual actor continued_support, original observation dtype',
            autonomous_recovery_proven=False), indent=2))

    def _on_training_end(self):
        super()._on_training_end()
        self.save_counts()


def main():
    arguments = sys.argv[1:]
    if '--resume-quality' not in arguments or '--scheduled-recovery' not in arguments:
        raise ValueError('Explicit --resume-quality and --scheduled-recovery required')
    output = Path(arguments[arguments.index('--output')+1])
    if output.exists():
        raise FileExistsError('Fresh output required; no experiment overwritten')
    if shutil.disk_usage(output.parent).free < 250*1024**2:
        raise RuntimeError('At least 250 MiB free disk required before this small experiment')
    provenance = output.with_name(output.name+'_prototype_provenance')
    # Generic trainer requires its output directory not to exist at entry.
    provenance.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    sources = {}
    for name in ('__init__.py', 'continued_recovery_policy.py', 'train_continued_recovery.py'):
        relative = Path('prototypes')/name
        data = (root/relative).read_bytes()
        sources[str(relative)] = hashlib.sha256(data).hexdigest()
        target = provenance/'prototype_code_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    record = dict(source_sha256=sources, captured_before_training=True,
                  experiment_output=str(output.resolve()),
                  status='prepared', deployment_approved=False)
    path = provenance/'PROTOTYPE_SOURCE.json'
    path.write_text(json.dumps(record, indent=2))
    alignment.ALIGNMENT_COST = 0.
    trainer.make_environment = focused.make_environment
    trainer.resume_quality = warm_start_continued
    trainer.RecoveryCoverageProgress = ContinuedProgress
    try:
        trainer.main()
        record['status'] = 'training_entry_returned'
    except BaseException:
        record['status'] = 'interrupted_or_failed'
        raise
    finally:
        config_path = output/'candidate/config.json'
        if config_path.exists():
            config = json.loads(config_path.read_text())
            config.update(continued_recovery_support=True,
                          continuation_alignment_cos=.7,
                          prototype_source_record=str(path.resolve()),
                          residual_disabled_when='actual continued_support predicate is false',
                          prior_scale_inherited_from_checkpoint=True,
                          fixed_maneuver=False, deployment_approved=False)
            config_path.write_text(json.dumps(config, indent=2))
        record['source_hashes_unchanged'] = all(
            hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
            for name, digest in sources.items())
        path.write_text(json.dumps(record, indent=2))
        if not record['source_hashes_unchanged']:
            raise RuntimeError('Prototype source changed during experiment; reject results')


if __name__ == '__main__':
    main()
