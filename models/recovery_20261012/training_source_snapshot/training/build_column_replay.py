"""Choose a recorded task that visibly exercises the paired blockage, without tuning weights."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;out=ROOT/'runs/column_navigation_v2';report=json.loads((out/'metrics.json').read_text())
candidates=[]
for i in range(3):
    folder=out/f'map_{i}';scene=ROOT/f'assets/column_navigation_tasks_v2/columns_{i:02d}'
    metrics=json.loads((folder/'metrics.json').read_text());traces=json.loads((folder/'trajectories.json').read_text())
    config=json.loads((scene/'physical_scene.json').read_text())
    for e,t in zip(metrics['episodes'],traces):
        if not e['verified_success']:continue
        samples=t['samples'];holds=sum(any(b['phase']=='holding' for b in s.get('column_block_events',[])) and s['commanded_velocity'][0]<.02 for s in samples)
        record=dict(scene_directory=str(scene),scene_config=config,map_id=e['map_id'],scene_file='physical_scene.xml',simulation_backend=metrics['simulation_backend'],seed=e['seed'],task=e['episode'],controller='PPO',localizer=metrics['localizer'],lidar=metrics['lidar'],outcome=e,goal=t['goal'],path=t['path'],samples=samples,replay_selection='Successful paired-column blockage with visible policy low-speed time; weights unchanged',policy_low_speed_during_hold_steps=holds)
        (folder/f"successful_replay_{e['seed']}.json").write_text(json.dumps(record))
        candidates.append((len(e['column_block_events']),holds,-e['collision_events'],record))
if not candidates:raise RuntimeError('No verified successful column task to render')
record=max(candidates,key=lambda x:x[:3])[-1]
(out/'successful_replay.json').write_text(json.dumps(record))
report.update(replay_seed=record['seed'],replay_low_speed_during_hold_steps=record['policy_low_speed_during_hold_steps'],map_note='Known v2 layouts with a new explicit passage barrier in the third scene; these are task tests, not an independent unseen-map generalization benchmark')
(out/'metrics.json').write_text(json.dumps(report,indent=2))
p=out/'index.html';s=p.read_text().replace('测试地图沿用已知 v2 场景','场景基于已知 v2 地图，第三张增加明确的狭窄入口').replace('3 个狭窄堵路任务。','3 个狭窄堵路任务；双柱放在真正通道口，机器人距离任一柱子 1.5 米内同步启动 18 秒计时。');p.write_text(s)
print('Replay chosen:',record['seed'],'policy low-speed steps while columns holding:',record['policy_low_speed_during_hold_steps'])
