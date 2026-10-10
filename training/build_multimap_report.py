"""Combine independent navigation evaluations without inferring success from a replay."""
import argparse
import json
from pathlib import Path
from html import escape
from navrl.geometry import ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'runs/multimap_demo');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    folders={'particle_unseen_maps':'multimap_visible_test','amcl_unseen_maps':'multimap_visible_amcl_test','route_obstacle_stress':'multimap_visible_stress'}
    reports={k:json.loads((ROOT/'runs'/v/'metrics.json').read_text()) for k,v in folders.items()}
    selection=json.loads((ROOT/'runs/multimap_navigation/config.json').read_text())['model_selection']
    manifest=json.loads((ROOT/'assets/navigation_dataset/manifest.json').read_text())
    train={x['sha256'] for x in manifest['splits']['train']};test={x['sha256'] for x in manifest['splits']['test']}
    assert train.isdisjoint(test),'Train/test map contamination'
    result={'model_selection':selection,'train_test_maps_disjoint':True,'training_maps':6,'validation_maps':2,'unseen_test_maps':3,'evaluations':reports,'limits':'Small procedural-map benchmark, known initial-pose prior, 0.2 m/s forward limit; no arbitrary-map guarantee. Classical localization and planning, learned PPO local velocity decisions. Collision allowed for arrival; separately reported.'}
    (a.output/'validation.json').write_text(json.dumps(result,indent=2))
    labels={'particle_unseen_maps':'未见地图 · 粒子激光定位','amcl_unseen_maps':'未见地图 · 真实 Nav2 AMCL','route_obstacle_stress':'实体行人堵路压力测试'}
    rows=[]
    for name,report in reports.items():
        e=report['episodes'];n=len(e);s=sum(x['verified_success'] for x in e)
        rows.append(f'<tr><td>{escape(labels[name])}</td><td>{s}/{n} ({s/n:.1%})</td><td>{sum(x["collision_events"]>0 for x in e)}/{n}</td><td>{sum(x["status"]=="timeout" for x in e)}</td><td>{sum(x.get("plan_revisions",0) for x in e)}</td></tr>')
    maps=[]
    for map_id in sorted(set(x['map_id'] for x in reports['particle_unseen_maps']['episodes'])):
        e=[x for x in reports['particle_unseen_maps']['episodes'] if x['map_id']==map_id];maps.append(f'<li>{escape(map_id)}：{sum(x["verified_success"] for x in e)}/{len(e)} 到达</li>')
    html='''<!doctype html><meta charset="utf-8"><title>多地图导航评估</title><style>body{font:17px sans-serif;max-width:1100px;margin:30px auto;color:#233044;background:#f3f6fa}table{border-collapse:collapse;background:white}td,th{padding:13px;border:1px solid #ccd4df}img{width:32%}a{color:#0965b3}</style><h1>多地图导航：独立评估与实际回放</h1><p>6 张训练地图，2 张验证地图，3 张未参与训练的测试地图。模型先按验证集固定，再评估测试集。起终点与初始朝向随机；目标实际距离不超过 0.30 米算抵达，允许碰撞但单独记录。</p><p>激光地图定位和 A* 动态重规划属于算法模块；PPO 学习局部前进速度与转向。监督器等待、模型主动停车与物理接触阻挡分开记录，不能把监督器行为说成模型学会的行为。</p><table><tr><th>测试</th><th>抵达</th><th>有接触的回合</th><th>超时</th><th>路线变化次数</th></tr>ROWS</table><ul>MAPS</ul><p>这是小规模仿真验证，不证明任意地图都成功；使用粗略初始定位，最高前进速度 0.2 m/s，实体行人停车让行。不运行实车。</p><p><a href="replay.html">查看成功回合：激光、实时路线和决策依据</a> · <a href="successful_navigation.mp4">MuJoCo 视频</a> · <a href="validation.json">完整评估数据（包含失败回合）</a></p><p>下面各图是三个不同测试地图的第一回合，不是把不同地图的轨迹叠在同一张地图上。</p><img src="../multimap_visible_test/map_episode_00.png"><img src="../multimap_visible_test/map_episode_01.png"><img src="../multimap_visible_test/map_episode_02.png">'''
    (a.output/'index.html').write_text(html.replace('ROWS',''.join(rows)).replace('MAPS',''.join(maps)))
    print(json.dumps({name:dict(successes=sum(x['verified_success'] for x in r['episodes']),episodes=len(r['episodes']),collision_rate=r['collision_rate']) for name,r in reports.items()},indent=2))


if __name__=='__main__':main()
