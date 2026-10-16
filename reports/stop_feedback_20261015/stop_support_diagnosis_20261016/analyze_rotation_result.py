"""Inspect the completed physical rotation probe, including rejected requests."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import COMMANDS


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = OUT.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = OUT/'ROTATION_RESULT.json'
    if destination.exists():
        raise ValueError('Do not overwrite completed physical evidence')
    budget = ExperimentBudget(.05, memory_mib=1024, directory=OUT)
    protocol = BASE/'rotation_escape_t106_protocol_20261016/RESULTS.json'
    complete = json.loads(protocol.read_text())
    if not complete['completed']:
        raise ValueError('Full diagnostic must be complete')
    inputs = {str(protocol.relative_to(ROOT)):digest(protocol)}
    traces = {}; summaries = {}
    for variant, folder, row_key in [
            ('model_only','rotation_escape_t106_model_20261016','frozen_model'),
            ('timed_diagnostic','rotation_escape_t106_probe_20261016','timed_probe')]:
        row = complete[row_key]
        report_path = BASE/folder/'RESULTS.json'
        report = json.loads(report_path.read_text())
        row_id = row['specification']['id'] if variant == 'timed_diagnostic' else row['case_id']
        path = BASE/folder/(row_id+'.npz')
        if not report['completed'] or digest(path)!=row['trace_sha256']:
            raise ValueError('Complete report and physical trace hashes required')
        with np.load(path,allow_pickle=False) as z:
            trace = {name:z[name].copy() for name in z.files}
        n = row['frozen_prefix_frames']
        if n!=530 or row['prefix_trace_sha256']!=complete['frozen_model']['prefix_trace_sha256']:
            raise ValueError('The same verified530-frame physical prefix is required')
        masks = trace['masks'][n:]
        choices = trace['choices'][n:]
        commands = trace['commands'][n:]
        if not masks[np.arange(len(choices)),choices].all():
            raise ValueError('The diagnostic executed an inadmissible action')
        np.testing.assert_allclose(commands,COMMANDS[choices],rtol=0,atol=2e-8)
        delta_yaw = np.diff(np.unwrap(trace['labels'][n:,3]))
        summaries[variant] = dict(post_initialization_frames=len(choices),
            post_initialization_choices=dict(Counter(map(str,choices))),
            post_initialization_only_stop_frames=int((~masks[:,1:].any(-1)).sum()),
            post_initialization_command_stop_frames=int((choices==0).sum()),
            net_yaw_change_rad=float(delta_yaw.sum()),
            cumulative_absolute_yaw_change_rad=float(abs(delta_yaw).sum()),
            final_xy_error_m=row['final_xy_error_m'],total_travel_m=row['travel_m'],
            safe_success=row['safe_success'],collisions=row['collisions'],
            timeout=row['timeout'],physical_guards=row['physical_guards'],
            model_only=variant=='model_only')
        traces[variant] = trace
        for p in [report_path,path]:
            inputs[str(p.relative_to(ROOT))] = digest(p)
        budget.tick()
    for name in ['labels','scans','odometry','measured_velocity','choices','commands']:
        np.testing.assert_allclose(traces['model_only'][name][:530],
            traces['timed_diagnostic'][name][:530],rtol=0,atol=2e-8,equal_nan=True)
    interventions = complete['timed_probe']['intervention']
    if len(interventions)!=16 or any(v['requested_choice']!=5 for v in interventions):
        raise ValueError('The predeclared3.2-second action5 request window changed')
    requests = dict(frames=len(interventions),
        original_action_choice=5,command=COMMANDS[5].tolist(),
        legal_and_executed=sum(bool(v['replaced_reference']) for v in interventions),
        illegal_model_fallback=sum(not v['requested_choice_legal'] for v in interventions),
        full_physical_trial_is_not_model_only=True)
    outcomes = dict(completed=True,at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        same_sensor_and_physical_initialization_verified=True,
        summaries=summaries,request_window=requests,
        diagnostic_extra_total_travel_m=(summaries['timed_diagnostic']['total_travel_m']-
                                        summaries['model_only']['total_travel_m']),
        successful_recovery_demonstration_available=False,
        inputs_sha256=inputs,source_sha256=digest(Path(__file__)),
        additional_environment_steps=0,physical_cases_rerun=0,optimizer_updates=0,
        limitations=['One reused development state; not an unseen mission.',
            'The requested action is only executed when legal; fallback model decisions can change direction.',
            'Failure of this finite window does not prove the state is unreachable or every legal motion is useless.',
            'Reduced voluntary waiting here was accompanied by more forced stopping and no physical arrival.',
            'This failed complete trial must not be labelled a successful recovery demonstration.'],
        resource=budget.statistics(),goal_achieved=False,deployment_approved=False)
    destination.write_text(json.dumps(outcomes,indent=2)+'\n')
    print(json.dumps(dict(completed=True,summaries=summaries,
        request_window=requests,extra_travel_m=outcomes['diagnostic_extra_total_travel_m'],
        resource=outcomes['resource']),indent=2),flush=True)


if __name__=='__main__':
    main()
