"""Score cumulative-occlusion hypothesis on already recorded sensor-derived tracks."""
import argparse
import json
from pathlib import Path
import torch
from experiments.cumulative_contour_prediction import CumulativeContourPrediction,shadow_risks
from experiments.train_quality_navigation import digest
from neural_navigation.masked_actions import COMMANDS


def main():
    parser=argparse.ArgumentParser()
    for name in ('audit','hardware','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Fresh output required')
    torch.set_num_threads(1);audit=json.loads(args.audit.read_text());hardware=json.loads(args.hardware.read_text())
    predictor=CumulativeContourPrediction();rows=[]
    for row in audit['samples']:
        # Extract only measured fields; explicitly exclude diagnostic actor labels.
        raw=[{k:t[k] for k in ('id','time','center','velocity','hits','points')} for t in row['tracks']]
        processed=predictor.update(raw,row['timestamp'])
        risks,details=shadow_risks(processed,row['estimate'],row['measured_velocity'],row['timestamp'],COMMANDS,hardware['robot_footprint_m'])
        rows.append(dict(timestamp=row['timestamp'],recorded_choice=row['choice'],old_selected_risk=row['actual_risk_scores'][row['choice']],
            shadow_selected_risk=float(risks[row['choice']]),shadow_scores=risks.tolist(),tracks=details))
    report=dict(audit_sha256=digest(args.audit),hardware_sha256=digest(args.hardware),samples=rows,commands_issued=False,
        simulator_truth_used=False,limitation='One shadow failure trajectory, already processed measured tracks. Uncalibrated geometry/speed hypotheses; not new navigation behavior, risk certification or general reversal handling.')
    args.output.write_text(json.dumps(report,indent=2));print(json.dumps([r for r in rows if r['timestamp']>=8.]))


if __name__=='__main__':main()
