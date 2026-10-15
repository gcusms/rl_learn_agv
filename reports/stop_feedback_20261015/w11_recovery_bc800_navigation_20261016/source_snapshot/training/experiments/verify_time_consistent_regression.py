"""Verify complete time-consistent sensor-core regression artifacts."""
import argparse
import json
from pathlib import Path
import torch
from experiments.verify_quality_regression import verify


def main():
    parser=argparse.ArgumentParser()
    for name in ('run','bundle','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Fresh output required')
    torch.set_num_threads(1);result=verify(args.run,args.bundle)
    source=json.loads((args.run/'SOURCE_PROVENANCE.json').read_text())
    manifest=json.loads((args.bundle/'manifest.json').read_text())
    if source.get('runtime_variant')!='fresh_fusion_time_consistent_confirmation_filter' or not source.get('captured_before_launch'):
        raise ValueError('Expected time-consistent pre-launch source snapshot')
    if manifest.get('sensor_core')!='time-consistent':raise ValueError('Expected time-consistent model bundle')
    result.update(source_captured_before_launch=True,sensor_core='time-consistent',
        limitation='48 reused development tasks. Snapshot captured before launch; evaluator reports simulated collision and arrival labels. No semantic understanding, physical hardware or sealed final acceptance claim.')
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='per_case'}),flush=True)


if __name__=='__main__':main()
