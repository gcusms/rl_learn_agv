"""One diagnostic episode; the memory factor is installed before any scan.

The portable source bundle stays unchanged. This overlay is an experiment,
not a portable runtime or a deployment candidate.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job', type=Path)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    import numpy as np
    import torch
    from experiments.evaluate_docking_navigation import evaluate
    from experiments.release_recovery import evaluate_recovery
    from hybrid_memory import HybridFreeEvidenceMemory, contract
    from neural_navigation.aligned_release_runtime import AlignedReleaseRuntime, RUNTIME_NAME
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.safety_memory import NearFieldMemory

    assert job['variant'] in ('control', 'hybrid')
    torch.set_num_threads(1)
    created = []

    class DiagnosticRuntime(AlignedReleaseRuntime):
        def __init__(self, *positional, **keywords):
            super().__init__(*positional, **keywords)
            assert type(self.guard.memory) is NearFieldMemory
            assert not self.guard.memory.entries and self.guard.memory.last_time is None
            self.memory_probe_counts = []
            self.memory_install_elapsed_s = float(self.core.elapsed)
            if job['variant'] == 'hybrid':
                self.guard.memory = HybridFreeEvidenceMemory(self.guard.memory.config)
            assert self.policy.guard is self.guard
            created.append(self)

        def step(self, *positional, **keywords):
            result = super().step(*positional, **keywords)
            self.memory_probe_counts.append(len(self.guard.memory.entries))
            return result

    bundle = Path(job['bundle'])
    manifest = json.loads((bundle/'manifest.json').read_text())
    assert manifest['required_runtime'] == RUNTIME_NAME
    budget = ExperimentBudget(.35, 2800, args.job.parent)
    target = 'experiments.release_recovery.resolve_runtime' if job['recovery'] else 'neural_navigation.goal_pose_runtime.resolve_runtime'
    function = evaluate_recovery if job['recovery'] else evaluate
    with patch(target, return_value=DiagnosticRuntime):
        row, trace = function(job['spec'], bundle, budget)
    assert len(created) == 1
    runtime = created[0]
    assert len(runtime.memory_probe_counts) == len(trace['choices'])
    trace['guard_memory_point_counts'] = np.asarray(runtime.memory_probe_counts, dtype=np.int32)
    path = Path(job['trace'])
    np.savez_compressed(path, **trace)
    row.update(variant=job['variant'], trace_path=str(path.relative_to(Path.cwd())),
        trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        frozen_source_model=job['bundle'], policy_weights_unchanged=True,
        diagnostic_memory_overlay=job['variant']=='hybrid',
        memory_installed_before_first_sensor_step=True,
        memory_install_elapsed_s=runtime.memory_install_elapsed_s,
        memory_contract=contract() if job['variant']=='hybrid' else {'schema':'original_near_field_memory'},
        fresh_process_per_case=True, new_ppo_steps=0, new_optimizer_updates=0,
        resource=budget.statistics(), deployment_approved=False)
    output = Path(job['output'])
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(row, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(output)
    print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
