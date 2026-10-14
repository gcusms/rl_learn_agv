"""Archive only after training completes, then evaluate fixed physical tasks."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'training/runs/docking_staged_ppo_10k_20261014'
MODEL=ROOT/'models/docking_candidate_20261014'
EVAL=ROOT/'training/runs/docking_full_evaluation_20261014'

def main():
    while True:
        progress=RUN/'progress.json'
        if progress.exists() and json.loads(progress.read_text()).get('phase')=='complete':break
        if (RUN/'ERROR.txt').exists():raise RuntimeError((RUN/'ERROR.txt').read_text())
        time.sleep(10)
    MODEL.mkdir(exist_ok=False)
    for name in ('policy.zip','config.json','progress.json','source_snapshot.zip'):
        shutil.copy2(RUN/name,MODEL/name)
    shutil.copytree(RUN/'portable_10000',MODEL/'portable')
    shutil.copytree(MODEL/'portable',MODEL/'portable_v4_localization')
    from neural_navigation.odom_anchored_localizer import turn_response_contract
    bundle=MODEL/'portable_v4_localization'
    (bundle/'localization_contract.json').write_text(json.dumps(turn_response_contract(),indent=2))
    p=bundle/'manifest.json';manifest=json.loads(p.read_text())
    manifest['files']['localization_contract.json']=hashlib.sha256((bundle/'localization_contract.json').read_bytes()).hexdigest()
    manifest['experimental_localization']='odom_anchored_static_support_v4'
    p.write_text(json.dumps(manifest,indent=2))
    chain=['docking_conditioned_ppo_10k_20261014','docking_progress_fixed_10k_20261014','docking_staged_ppo_10k_20261014']
    lineage=[]
    for name in chain:
        source=ROOT/'training/runs'/name;target=MODEL/'training_lineage'/name;target.mkdir(parents=True)
        for f in ('config.json','progress.json','interruption.json','source_snapshot.zip'):
            if (source/f).exists():shutil.copy2(source/f,target/f)
        lineage.append(dict(run=name,config=json.loads((source/'config.json').read_text())))
    teacher=ROOT/'training/runs/docking_ppo_10k_20261014'
    shutil.copy2(teacher/'demonstrations.npz',MODEL/'demonstrations.npz')
    shutil.copy2(teacher/'demonstration_progress.json',MODEL/'demonstration_progress.json')
    (MODEL/'LINEAGE.json').write_text(json.dumps(dict(total_selected_terminal_ppo_steps=10000,
        parent_navigation_checkpoint='models/goal_pose_pilot_20261014/policy_3000.zip',
        previous_navigation_actor_frozen=True,localizer_router_neural_weights_frozen=True,
        localization_v4_is_engineering_not_ppo=True,stages=lineage,
        excluded_trial='docking_ppo_10k_20261014: independent aborted 2500-step trial'),indent=2))
    print('Archived separate 10k PPO candidate; starting fixed paired physics evaluation',flush=True)
    with (RUN/'full_evaluation.log').open('w') as log:
        subprocess.run([sys.executable,'-m','experiments.evaluate_docking_navigation',
            '--candidate',str(MODEL/'portable'),'--localized',str(bundle),
            '--cases',str(ROOT/'reports/real_adaptation_followup_20261014/goal_pose_cases.json'),
            '--additional','--cpu-cores','.5','--output',str(EVAL)],cwd=ROOT,
            stdout=log,stderr=subprocess.STDOUT,check=True)
    print('56 candidate/combined physical evaluations complete',flush=True)

if __name__=='__main__':main()
