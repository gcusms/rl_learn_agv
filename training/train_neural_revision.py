"""Run the second revision after component training, with fresh held-out maps."""
import subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
components=root/'runs/neural_components_v2'
while not (components/'config.json').exists():time.sleep(5)
cmd=[sys.executable,str(root/'train_neural_navigation.py'),'--components',str(components),'--output',str(root/'runs/neural_navigation_v2'),'--episodes','24','--epochs','100','--steps','4096']
with (root/'neural-navigation-v2-training.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
jobs=[]
for name,policy in [('warm',root/'runs/neural_navigation_v2/warm_start.zip'),('4000',root/'runs/neural_navigation_v2/checkpoints/ppo_4000_steps.zip'),('final',root/'runs/neural_navigation_v2/policy.zip')]:
    out=root/f'runs/neural_v2_{name}_validation'
    cmd=[sys.executable,str(root/'evaluate_navigation.py'),'--neural','--components',str(components),'--multi-map','--split','validation','--policy',str(policy),'--episodes','6','--seed','17000','--max-episode-steps','700','--output',str(out),'--record-success']
    log=(root/f'neural-v2-{name}-validation.log').open('w');jobs.append((subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT),log))
for p,log in jobs:
    code=p.wait();log.close()
    if code:raise RuntimeError('Validation failed, inspect logs')
subprocess.run([sys.executable,str(root/'finalize_neural_navigation_v2.py')],check=True)
