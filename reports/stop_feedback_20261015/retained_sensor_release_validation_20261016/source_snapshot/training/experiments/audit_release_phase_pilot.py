"""Audit completed paired traces and plot remaining waits without new physics."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from experiments.long_risk_actor import digest
from neural_navigation.decision_feedback import FEEDBACK_REASONS, REASON_SLICE
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS, COMMANDS, masks_from_observation


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'


def runs(wait, times, final_time):
    result = []
    edges = np.r_[times, final_time]
    start = None
    for i, active in enumerate(np.r_[wait, False]):
        if active and start is None:
            start = i
        elif not active and start is not None:
            result.append(dict(first_frame=start, end_frame_exclusive=i,
                start_s=float(edges[start]), end_s=float(edges[i]),
                duration_s=float(edges[i] - edges[start])))
            start = None
    return sorted(result, key=lambda r: r['duration_s'], reverse=True)


def load_trace(row):
    path = ROOT / row['trace_path']
    assert digest(path) == row['trace_sha256']
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.15, memory_mib=1536, directory=output)
    validation = BASE / 'release_phase_validation_20261016/RESULTS.json'
    baseline = BASE / 'release_validation_20261016/RESULTS.json'
    report = json.loads(validation.read_text())
    old = json.loads(baseline.read_text())
    assert report['completed'] and len(report['rows']) == 4
    phase_path = ROOT / 'models/stop_feedback_20261015/release_phase300_s231016/portable/actor.pt'
    control_path = ROOT / 'models/stop_feedback_20261015/release_data_control300_s231016/portable/actor.pt'
    inputs = {str(p.relative_to(ROOT)): digest(p)
        for p in [Path(__file__), validation, baseline, phase_path, control_path]}
    summaries, traces = [], {}
    for row in report['rows']:
        trace = load_trace(row)
        observations, choices = trace['observations'], trace['choices']
        masks = masks_from_observation(observations[:, :282])
        np.testing.assert_array_equal(masks, trace['masks'])
        assert masks[np.arange(len(masks)), choices].all()
        # The final inactive arrival row may have a motion proposal and a
        # zero execution, as in the original evaluator and source admission.
        final_arrival = bool(row['physical_arrival'] and row['stop_reasons'].get('estimated_goal_reached') == 1)
        away = np.arange(len(choices)) < len(choices) - int(final_arrival)
        np.testing.assert_allclose(trace['commands'][away], COMMANDS[choices[away]], rtol=0, atol=2e-8)
        path = phase_path if row['variant'] == 'recovery_phase' else control_path
        actor = torch.jit.load(str(path), map_location='cpu').eval()
        with torch.no_grad():
            x = torch.from_numpy(observations)
            np.testing.assert_array_equal(actor(x).numpy(), ACTIONS[choices])
            probability = actor.logits(x).masked_fill(~torch.from_numpy(masks), -1e8).softmax(1)[:, 0].numpy()
        wait = (choices == 0) & masks[:, 1:].any(1)
        assert int(wait.sum()) == row['voluntary_wait_frames']
        spans = runs(wait, trace['labels'][:, 0], row['seconds'])
        risk = observations[:, 439:481].reshape(-1, 3, 14)
        supported = (risk[:, :, 12] > .5).any(1)
        supported_no_contact = supported & (risk[:, :, :12] == 0).all((1, 2))
        reason_values = observations[:, 303 + REASON_SLICE.start:303 + REASON_SLICE.stop]
        np.testing.assert_array_equal(reason_values.sum(1), np.ones(len(choices)))
        reason_ids = reason_values.argmax(1)
        summary = dict(case_id=row['case_id'], variant=row['variant'], frames=len(choices),
            portable_decisions_reproduced=True, nonterminal_execution_parity=True,
            safe_success=row['safe_success'], seconds=row['seconds'],
            voluntary_wait_frames=int(wait.sum()), longest_voluntary_wait_s=spans[0]['duration_s'] if spans else 0.,
            longest_waits=spans[:3],
            wait_without_supported_risk_history=int((wait & ~supported).sum()),
            wait_with_supported_history_and_zero_contact_scores=int((wait & supported_no_contact).sum()),
            no_forecast_contact_is_not_a_counterfactual_safe_motion_label=True,
            previous_reasons_on_wait={reason:int((wait & (reason_ids == i)).sum())
                for i, reason in enumerate(FEEDBACK_REASONS) if (wait & (reason_ids == i)).any()},
            pstop_on_wait_p50=float(np.median(probability[wait])) if wait.any() else None)
        summaries.append(summary)
        traces[(row['variant'], row['case_id'])] = (row, trace, wait)
        budget.tick()
    known = 'long_risk_sensor_05_1'
    old_row = next(row for row in old['rows'] if row['case_id'] == known)
    old_trace = load_trace(old_row)
    control = traces[('stop_move', known)][1]
    parity = {key:bool(np.array_equal(old_trace[key], control[key], equal_nan=True))
        for key in ['observations', 'choices', 'commands', 'labels', 'scans', 'odometry', 'measured_velocity']}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), gridspec_kw={'height_ratios':[2, 1]}, constrained_layout=True)
    styles = [('Old candidate', old_row, old_trace, '#667085', '-'),
        ('Added-data control', traces[('stop_move', known)][0], control, '#c87925', '--'),
        ('Phase-balanced candidate', *traces[('recovery_phase', known)][:2], '#137c66', '-')]
    for line_index, (label, row, trace, color, line_style) in enumerate(styles):
        times = trace['labels'][:, 0]
        axes[0].plot(times, trace['labels'][:, 7], label=label, color=color, linestyle=line_style)
        wait = (trace['choices'] == 0) & trace['masks'][:, 1:].any(1)
        for span in runs(wait, times, row['seconds']):
            axes[1].broken_barh([(span['start_s'], span['duration_s'])], (line_index-.25, .5), facecolors=color)
    axes[0].axhline(.2, color='#aaa', linewidth=.8, linestyle=':')
    axes[0].set(ylabel='Physical distance to goal (m)', title='Same development condition; learned-only closed-loop navigation')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.2)
    axes[1].set(xlabel='Simulation time (s)', yticks=[0,1,2],
        yticklabels=['Old candidate', 'Added-data control', 'Phase-balanced candidate'],
        title='Waiting while motion choices are available')
    axes[1].grid(axis='x', alpha=.2)
    fig.savefig(output / 'WAIT_COMPARISON.svg')
    fig.savefig(output / 'WAIT_COMPARISON.png', dpi=140)
    plt.close(fig)
    for name, expected in inputs.items():
        assert digest(ROOT / name) == expected, name
    budget.tick()
    result = dict(completed=True, at=datetime.now(timezone.utc).isoformat(), input_sha256=inputs,
        rows=summaries, original_and_added_data_control_exact_trace_parity=parity,
        resource=budget.statistics(), new_physical_steps=0, new_training_steps=0,
        goal_achieved=False, deployment_approved=False)
    (output / 'RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'driver_snapshot.py').write_text(Path(__file__).read_text())
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
