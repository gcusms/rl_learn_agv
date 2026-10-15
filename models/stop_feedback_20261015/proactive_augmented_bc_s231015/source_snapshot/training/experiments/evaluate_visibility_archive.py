"""Same-actor sensor-runtime experiment; no predictor deployment or training."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import evaluate_sensor_navigation as evaluation
from neural_navigation.runtime import SensorNavigationRuntime
from experiments.visibility_archive import VisibilityArchivePredictor


class ArchiveRuntime(SensorNavigationRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.core.laser_fusion.predictor
        if old is None:
            raise ValueError('Archive experiment requires a configured motion predictor')
        self.core.laser_fusion.predictor = VisibilityArchivePredictor(old.config, self.core)

    def step(self, *args, **kwargs):
        result = super().step(*args, **kwargs)
        predictor = self.core.laser_fusion.predictor
        result['diagnostics'].update(scan_archived_tracks=sum(self.core.elapsed-t['time'] > .8 for t in predictor.tracks),
            scan_archive_reassociations=predictor.reassociations,
            scan_archive_clear_removals=predictor.clear_removals)
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--traces', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.traces.exists():
        p.error('Fresh report and traces required')
    manifest = json.loads(args.manifest.read_text())
    if manifest.get('kind') != 'development':
        p.error('Development manifest only; final acceptance excluded')
    torch.set_num_threads(1)
    evaluation.SensorNavigationRuntime = ArchiveRuntime
    results = []
    for case in manifest['cases']:
        result = evaluation.run_case((case, args.bundle.resolve(), 700, args.traces))
        results.append(result)
        print('CASE', json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(completed=True, cases=results, tasks=len(results),
        safe_success=sum(x['safe_success'] for x in results), collisions=sum(x['collisions'] for x in results),
        timeouts=sum(x['timeout'] for x in results), manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        bundle_manifest=json.loads((args.bundle/'manifest.json').read_text()),
        experiment='visibility_archive_v1', archive_s=4., forecast_age_limit_s=.8,
        source_sha256={name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                       for name in ('visibility_archive.py', 'evaluate_visibility_archive.py')},
        deployment_approved=False, limitation='Same exported actor, static/localization/guard/robot limits; '
        'sensor-only predictor identity retention differs. Archive does not prolong stale forecasts. '
        'Case intervention rules are fixed, but policy motion affects actual actor placement. '
        'No causal attribution from a small closed-loop set and no real-robot calibration claim.'), indent=2))


if __name__ == '__main__':
    main()
