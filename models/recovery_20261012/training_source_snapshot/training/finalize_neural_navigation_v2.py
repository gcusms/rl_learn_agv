"""Freeze a validation-selected learned bundle, then evaluate untouched tasks."""
import hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs'
candidates=[('warm_start',RUN/'neural_navigation_v2/warm_start.zip',RUN/'neural_v2_warm_validation/metrics.json'),('ppo_4000',RUN/'neural_navigation_v2/checkpoints/ppo_4000_steps.zip',RUN/'neural_v2_4000_validation/metrics.json'),('ppo_4096',RUN/'neural_navigation_v2/policy.zip',RUN/'neural_v2_final_validation/metrics.json')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def score(m):
    e=m['episodes']
    return (sum(x['verified_success'] for x in e),-sum(x['collision_events'] for x in e),-sum(x['steps'] for x in e))
def evaluate(out,seed,episodes,map_index=None,stress=False,prior=None):
    cmd=[sys.executable,str(ROOT/'evaluate_navigation.py'),'--neural','--components',str(bundle),'--policy',str(bundle/'policy.zip'),'--multi-map','--dataset',str(ROOT/'assets/navigation_dataset_v2'),'--split','test','--episodes',str(episodes),'--seed',str(seed),'--max-episode-steps','700','--record-success','--record-most-stops','--output',str(out)]
    if map_index is not None:cmd+=['--map-index',str(map_index)]
    if stress:cmd+=['--obstacle-on-route']
    if prior is not None:cmd+=['--initial-prior-offset',*[str(x) for x in prior]]
    log=(ROOT/(out.name+'.log')).open('w')
    return subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT),log
print('Waiting for all validation results before freezing; no test-based selection.',flush=True)
while not all(m.exists() for _,_,m in candidates):time.sleep(10)
results=[(name,policy,json.loads(m.read_text())) for name,policy,m in candidates]
name,policy,metrics=max(results,key=lambda t:score(t[2]))
bundle=RUN/'full_model_navigation_v2';bundle.mkdir(exist_ok=True)
shutil.copy2(policy,bundle/'policy.zip')
for f in ['localizer.pt','router.pt']:shutil.copy2(RUN/'neural_components_v2'/f,bundle/f)
config=json.loads((RUN/'neural_navigation_v2/config.json').read_text());config.update(revision='v2: grid center sampling, matched 0.4 m waypoint teaching, near-wall localization training',teacher='offline controller following the same neural route; no teacher at inference',selected_actor=name,selection_basis='validation successes, then contact events, then total steps; no test selection',validation_scores={n:list(score(m)) for n,_,m in results},components=str(bundle),weight_sha256={f:sha(bundle/f) for f in ['policy.zip','localizer.pt','router.pt']},frozen_before_test=True)
(bundle/'config.json').write_text(json.dumps(config,indent=2));(bundle/'selection_validation.json').write_text(json.dumps({n:m for n,_,m in results},indent=2))
print('Frozen model:',name,'validation:',metrics['success_rate'],flush=True)
jobs=[]
for i in range(3):
    out=RUN/f'neural_v2_test_map{i}';p,l=evaluate(out,21000+i*3,3,i);jobs.append((p,l,out,False))
p,l=evaluate(RUN/'neural_v2_stress',22000,3,stress=True);jobs.append((p,l,RUN/'neural_v2_stress',True))
p,l=evaluate(RUN/'neural_v2_rough_prior',23000,3,prior=[.10,-.08,.05]);jobs.append((p,l,RUN/'neural_v2_rough_prior','rough'))
for p,l,out,stress in jobs:
    code=p.wait();l.close();print('Evaluation completed',out.name,'exit',code,flush=True)
    if code:raise RuntimeError(f'Failed evaluation {out}, inspect log')
normal=[json.loads((out/'metrics.json').read_text()) for _,_,out,s in jobs if not s]
stress=json.loads((RUN/'neural_v2_stress/metrics.json').read_text())
episodes=[e for m in normal for e in m['episodes']]
combined={**normal[0],'episodes':episodes,'sampled_tasks':[t for m in normal for t in m['sampled_tasks']],'success_rate':sum(e['verified_success'] for e in episodes)/len(episodes),'collision_rate':sum(e['collision_events']>0 for e in episodes)/len(episodes),'timeout_rate':sum(e['status']=='timeout' for e in episodes)/len(episodes),'evaluation_seeds':[21000,21003,21006]}
test=RUN/'neural_v2_test';test.mkdir(exist_ok=True);(test/'metrics.json').write_text(json.dumps(combined,indent=2))
record_candidates=[out/'successful_replay.json' for _,_,out,s in jobs if not s and (out/'successful_replay.json').exists()]
if record_candidates:
    selected=max(record_candidates,key=lambda p:(-json.loads(p.read_text())['outcome']['collision_events'],json.loads(p.read_text())['outcome']['policy_stop_near_obstacle_steps']))
    shutil.copy2(selected,test/'successful_replay.json')
    replay=json.loads(selected.read_text());replay['selection_note']='Successful test replay chosen for visible model stopping; this does not change frozen weights or aggregate results.'
    (test/'successful_replay.json').write_text(json.dumps(replay))
report=dict(rough_prior_test=json.loads((RUN/'neural_v2_rough_prior/metrics.json').read_text()),status='evaluated prototype; not production ready',architecture=config['architecture'],selected_actor=name,validation_success_rate=metrics['success_rate'],test=combined,stress=stress,runtime_amcl=False,runtime_robot_astar=False,runtime_navigation_supervisor=False,three_separately_trained_networks=True,initial_pose='User-provided coarse pose; navigation evaluation spawn jitter 0.025 m / 0.02 rad',localization_validation=json.loads((RUN/'neural_components_v2/config.json').read_text()),limitations=['Local pose tracking, no global relocalization','Limited 96-step neural grid propagation','Reachability scores are not calibrated arrival probabilities','Three separately trained networks, not joint end-to-end training','Simulation sensor and dynamics assumptions need real-world validation'],old_archive=str(ROOT/'archives/multimap_hybrid_20261010_145229.tar.gz'))
(bundle/'report.json').write_text(json.dumps(report,indent=2))
print('UNSEEN TEST',combined['success_rate'],'STRESS',stress['success_rate'],flush=True)
