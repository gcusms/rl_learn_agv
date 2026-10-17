"""Freeze complete published rollouts before auditing an unfinished PPO run.

Copies a bounded prefix of actual samples and their original checkpoints and
warmups. It never reads an unfinished rollout or treats this as final training
or navigation acceptance, and does not modify the running process.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.audit_prefix_recovery_rollouts import audit
from neural_navigation.experiment_budget import ExperimentBudget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-run', type=Path, required=True)
    parser.add_argument('--through-steps', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-cores', type=float, default=.05)
    args = parser.parse_args()
    if args.through_steps < 250 or args.through_steps % 250:
        parser.error('A positive complete 250-step rollout boundary is required')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(args.cpu_cores, memory_mib=2048, directory=args.output)
    frozen = args.output / 'input_snapshot'
    (frozen / 'rollouts').mkdir(parents=True)
    (frozen / 'warmups').mkdir()
    status = json.loads((args.model_run / 'status.json').read_text())
    config = json.loads((args.model_run / 'config.json').read_text())
    copied = {}

    def copy(relative):
        if relative in copied:
            return
        source, target = args.model_run / relative, frozen / relative
        raw = source.read_bytes()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        copied[relative] = hashlib.sha256(raw).hexdigest()

    copy('config.json')
    copy('policy_initial.zip')
    copy('INITIALIZATION_PARITY.json')
    for end in range(250, args.through_steps + 1, 250):
        relative = f'rollouts/rollout_{end:06d}.npz'
        copy(relative)
        with np.load(frozen / relative, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata_json']))
            if meta['collected_steps'] != end or len(data['observations']) != 250:
                raise ValueError('A complete original rollout is required')
            for text in data['execution_diagnostics_json'][:, 0]:
                warmup = json.loads(text)['initialization_trace']
                if warmup is not None:
                    path = Path(warmup)
                    if path.is_absolute() or '..' in path.parts or path.parts[0] != 'warmups':
                        raise ValueError('Warmup must be relative to the actual run')
                    copy(str(path))
        if end % 1000 == 0:
            copy(f'policy_step_{end}.zip')
        budget.tick()
    result = audit(frozen)
    budget.tick()
    if result['sampled_frames'] != args.through_steps:
        raise ValueError('The frozen sample count changed')
    if not json.loads((frozen / 'INITIALIZATION_PARITY.json').read_text())['completed']:
        raise ValueError('Initialization parity was not established')
    result.update(captured_at=datetime.now().astimezone().isoformat(),
        source_model_run=str(args.model_run), source_status_at_capture=status,
        target_ppo_steps=config['steps'], audited_through_steps=args.through_steps,
        audit_input_sha256=copied, running_process_modified=False,
        complete_training_audit=False, navigation_acceptance=False,
        resource=budget.statistics(),
        limitations=['Published prefix of an unfinished run only; the final run requires a separate complete audit.',
                     'Causal stop explanations and exact sampling probabilities do not establish successful navigation.'])
    (args.output / 'RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'audit_input_sha256'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
