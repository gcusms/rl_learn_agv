"""Same-model archive reconfirmation diagnostic; not a deployment entry."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import evaluate_sensor_navigation as evaluation
from experiments.evaluate_visibility_archive import ArchiveRuntime
from experiments.reconfirmed_archive import ReconfirmedArchivePredictor


class ReconfirmedRuntime(ArchiveRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.core.laser_fusion.predictor = ReconfirmedArchivePredictor(
            self.core.laser_fusion.predictor.config, self.core)

    def step(self, *args, **kwargs):
        result = super().step(*args, **kwargs)
        result['diagnostics']['scan_reconfirmation_resets'] = self.core.laser_fusion.predictor.reconfirmation_resets
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--traces', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.traces.exists():
        parser.error('Fresh report/traces required')
    manifest = json.loads(args.manifest.read_text())
    if manifest.get('kind') != 'development':
        parser.error('Development manifest only')
    source_dir = Path(__file__).parent
    names = ('visibility_archive.py', 'evaluate_visibility_archive.py',
             'reconfirmed_archive.py', 'evaluate_reconfirmed_archive.py')
    source_hashes = {name: hashlib.sha256((source_dir/name).read_bytes()).hexdigest() for name in names}
    torch.set_num_threads(1)
    evaluation.SensorNavigationRuntime = ReconfirmedRuntime
    results = []
    for case in manifest['cases']:
        result = evaluation.run_case((case, args.bundle.resolve(), 700, args.traces))
        results.append(result)
        print('CASE', json.dumps(result), flush=True)
    if source_hashes != {name: hashlib.sha256((source_dir/name).read_bytes()).hexdigest() for name in names}:
        raise RuntimeError('Experiment source changed during evaluation')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(completed=True, cases=results, tasks=len(results),
        safe_success=sum(x['safe_success'] for x in results), collisions=sum(x['collisions'] for x in results),
        timeouts=sum(x['timeout'] for x in results), source_sha256=source_hashes,
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        bundle_manifest=json.loads((args.bundle/'manifest.json').read_text()),
        experiment='visibility_archive_reconfirmed_v2', archive_s=4., forecast_age_limit_s=.8,
        reacquisition_observations_before_forecast=2, deployment_approved=False,
        limitation='Isolated sensor-runtime change, same actor and original guard. No extra training or deployment. '
        'Reassociation retains identity but requires one further scan for motion forecasting; reversal/stop gates still apply. '
        'Small known-failure set, no broad navigation/real-robot guarantee.'), indent=2))


if __name__ == '__main__':
    main()
