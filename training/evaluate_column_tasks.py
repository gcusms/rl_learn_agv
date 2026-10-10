"""Evaluate six long-range column tasks with unchanged frozen robot weights."""
import argparse
import json,subprocess,sys,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--revision',default='v2');args=parser.parse_args()
scene_root=ROOT/('assets/column_navigation_tasks_'+args.revision);out=ROOT/('runs/column_navigation_'+args.revision);out.mkdir(exist_ok=True)
weights=ROOT/'runs/full_model_navigation_v2'
hashes={f:hashlib.sha256((weights/f).read_bytes()).hexdigest() for f in ['policy.zip','localizer.pt','router.pt']}
jobs=[]
for i in range(3):
    folder=out/f'map_{i}';scene=scene_root/f'columns_{i:02d}'
    cmd=[sys.executable,str(ROOT/'evaluate_navigation.py'),'--neural','--components',str(weights),'--policy',str(weights/'policy.zip'),'--scene-dir',str(scene),'--episodes','2','--seed',str(26000+2*i),'--max-episode-steps','1000','--scheduled-columns','--record-success','--record-most-stops','--output',str(folder)]
    log=(out/f'map_{i}.log').open('w');jobs.append((subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT),log,folder))
for p,l,folder in jobs:
    code=p.wait();l.close()
    if code:raise RuntimeError('Evaluation failed: '+str(folder))
episodes=[e for _,_,folder in jobs for e in json.loads((folder/'metrics.json').read_text())['episodes']]
for i,e in enumerate(episodes):
    scene=scene_root/f'columns_{i//2:02d}/physical_scene.json';task=json.loads(scene.read_text())['tasks'][i%2]
    e.update(task_name=task['name'],straight_distance_m=task['straight_distance_m'],offline_reference_length_m=task['offline_reference_length_m'])
report=dict(episodes=episodes,success_rate=sum(e['verified_success'] for e in episodes)/len(episodes),collision_rate=sum(e['collision_events']>0 for e in episodes)/len(episodes),model_weight_sha256=hashes,weight_changes=False,scene_type='Moving solid cylinders; 6.5 m or greater start-goal separation',task_distribution='3 open-space bypass, 3 paired-column chokepoint tasks; 12/18 s holds after approach or 60 s trigger timeout',runtime_robot_amcl=False,runtime_robot_astar=False,success_criterion='Estimated goal distance <0.12 m plus diagnostic true distance <=0.30 m; contacts do not invalidate arrival')
for f,h in hashes.items():assert hashlib.sha256((weights/f).read_bytes()).hexdigest()==h
(out/'metrics.json').write_text(json.dumps(report,indent=2))
traces=[folder/'successful_replay.json' for _,_,folder in jobs if (folder/'successful_replay.json').exists()]
if traces:
    def score(p):
        d=json.loads(p.read_text());events=d['outcome']['column_block_events']
        return (sum(e['trigger_time_s'] is not None for e in events),sum(e['phase']=='cleared' for e in events),d['outcome']['policy_stop_near_obstacle_steps'],-d['outcome']['collision_events'])
    selected=max(traces,key=score);shutil.copy2(selected,out/'successful_replay.json')
rows=''.join(f"<tr><td>{e['map_id']}</td><td>{e['task_name']}</td><td>{e['straight_distance_m']:.2f}</td><td>{'成功' if e['verified_success'] else e['status']}</td><td>{e['true_goal_distance_m']:.3f}</td><td>{e['collision_events']}</td><td>{e['policy_stop_near_obstacle_steps']}</td><td>{', '.join(x['phase'] for x in e['column_block_events'])}</td></tr>" for e in episodes)
(out/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>移动柱子长距离导航测试</title><style>body{{font:16px sans-serif;margin:30px;background:#eef2f6}}table{{border-collapse:collapse;background:white}}td,th{{border:1px solid #bbb;padding:9px}}video{{max-width:100%;width:1280px}}</style><h1>移动实体柱子：长距离导航与临时堵路</h1><p>本轮抵达 {sum(e['verified_success'] for e in episodes)}/6，接触回合比例 {report['collision_rate']:.1%}。起终点直线距离 至少 6.5 米。沿用冻结的 v2 模型，本轮没有重训或根据结果更换权重。</p><p>3 个开阔绕行任务与 3 个狭窄堵路任务。柱子接近触发后保持 12/18 秒，双柱同时计时；60 秒未接近则触发释放计时。柱子移开使用加速度限制和原生实体接触，运行中不传送。机器人等待与绕行由现有策略输出，场景调度只控制柱子。</p><p>任务几何设计使用离线参考路线，运行时不将该路线传入机器人。测试地图沿用已知 v2 场景，本轮测试的是长距离任务和柱子调度，不能当作新的未见地图泛化成绩。</p><a href="../column_navigation_v2_demo/replay.html">完整交互回放</a> · <a href="metrics.json">所有任务与触发释放记录</a><table><tr><th>场景</th><th>任务</th><th>直线距离 m</th><th>结果</th><th>终点距离 m</th><th>接触事件</th><th>近障策略停车步</th><th>柱子阶段</th></tr>{rows}</table><p>地图画面、状态栏和机器人特写使用独立区域，避免遮挡。成功演示不能代替上述全部任务结果。</p><video controls src="../column_navigation_v2_demo/successful_navigation.mp4"></video>''')
print('Column tests completed:',sum(e['verified_success'] for e in episodes),'/6',flush=True)
