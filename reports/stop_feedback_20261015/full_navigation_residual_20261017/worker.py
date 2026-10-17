"""One physical case in a fresh process; coordinator retains only JSON."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('job',type=Path);args=parser.parse_args()
    job=json.loads(args.job.read_text())
    import numpy as np
    import torch
    from experiments.evaluate_docking_navigation import evaluate
    from experiments.release_recovery import evaluate_recovery
    from neural_navigation.experiment_budget import ExperimentBudget
    torch.set_num_threads(1)
    budget=ExperimentBudget(.35,2800,args.job.parent)
    function=evaluate_recovery if job['recovery'] else evaluate
    row,trace=function(job['spec'],Path(job['bundle']),budget)
    trace_path=Path(job['trace']);np.savez_compressed(trace_path,**trace)
    row.update(trace_path=str(trace_path.relative_to(Path.cwd())),
        trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),variant=job['variant'],
        fresh_process_per_case=True,resource=budget.statistics())
    path=Path(job['output']);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(row,indent=2)+'\n');temp.replace(path)
    print(json.dumps(row),flush=True)


if __name__=='__main__':main()
