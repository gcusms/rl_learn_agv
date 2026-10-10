"""Present frozen-weight test evidence without conflating demos and aggregate results."""
import html,json,hashlib,shutil,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
bundle=ROOT/'runs/full_model_navigation'
r=json.loads((bundle/'report.json').read_text());normal=r['test'];stress=r['stress'];episodes=normal['episodes']
def rows(es):
    return ''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [e['map_id'],e['seed'],'抵达' if e['verified_success'] else e['status'],e['steps'],f"{e['true_goal_distance_m']:.3f}",f"{e['mean_localization_error_m']:.3f}",e['collision_events'],e['supervisor_wait_steps']])+'</tr>' for e in es)
def table(es):return '<table><thead><tr><th>地图</th><th>种子</th><th>结果</th><th>步数</th><th>终点距离 m</th><th>定位平均误差 m</th><th>接触事件</th><th>外部等待覆盖</th></tr></thead><tbody>'+rows(es)+'</tbody></table>'
n=sum(e['verified_success'] for e in episodes);s=sum(e['verified_success'] for e in stress['episodes'])
p=f'''<!doctype html><meta charset="utf-8"><title>学习定位与神经路线导航验证</title><style>body{{font:16px sans-serif;background:#edf1f5;color:#1e293b;margin:30px}}main{{max-width:1100px;margin:auto;background:white;padding:28px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #cbd5e1;padding:8px;text-align:left}}.notice{{background:#fff2cc;padding:16px}}a{{color:#0369a1}}</style><main><h1>学习定位、神经路线与速度决策导航原型</h1><p class="notice">未见地图任务成功 {n}/{len(episodes)}（{normal['success_rate']:.1%}）；临时堵路成功 {s}/{len(stress['episodes'])}（{stress['success_rate']:.1%}）。原型尚未达到可靠导航要求，成功演示不能代表整体表现。</p><p>运行时机器人不调用 AMCL、粒子滤波、A* 或导航监督器；接收粗略初始位姿，学习定位持续修正，神经网络输出路线，速度策略输出前进和转向。几何预处理、里程计积分与实体物理仍为确定性计算。三个模块分别训练，尚非联合训练的端到端网络。</p><p>激光视场为实车 SCLIDAR 配置约 219.44°。机器人使用 URDF 外观，行人和墙体为 MuJoCo 实体。训练阶段允许离线教师标签，测试阶段教师关闭。行人的独立行为控制器使用路径搜索，不给机器人提供路线。</p><p>控制权重选型仅使用两个验证地图上的六个任务。选中：{html.escape(r['selected_actor'])}；验证成功率 {r['validation_success_rate']:.1%}。本轮 PPO 后续权重未改善成绩，因此保留初步学习权重。测试种子 18000–18014 和堵路种子 19000–19002 在权重冻结后才运行。</p><p>成功条件：学习定位进入 0.12 米目标容差并结束，随后真实位置距离目标不超过 0.30 米。碰撞不取消已确认的抵达，但另行报告。普通测试发生接触的回合比例：{normal['collision_rate']:.1%}。所有外部监督等待覆盖次数合计：{sum(e['supervisor_wait_steps'] for e in episodes)}。</p><p>定位仅支持粗略位姿附近的跟踪，不支持完全未知位置的全地图重定位；本轮导航初始实际位姿扰动约 0.025 米、0.02 弧度。单帧定位验证与长时导航分开评估，失败回合的定位漂移不能用单帧平均精度掩盖。有限神经传播范围和未校准的通行预测限制了泛化能力。</p><p><a href="../neural_demo/replay.html">成功任务交互回放与 MuJoCo 视频</a> · <a href="report.json">完整报告 JSON</a> · <a href="selection_validation.json">验证集选型记录</a> · <a href="config.json">冻结权重及 SHA256</a> · <a href="../../NEURAL_NAVIGATION_DESIGN.md">实现说明</a></p><h2>15 个未见地图任务</h2>{table(episodes)}<h2>3 个临时堵路任务</h2>{table(stress['episodes'])}<p>失败记录分别保存在 neural_test_map0、neural_test_map1、neural_test_map2 和 neural_stress 目录的 failed_replay_*.json 中。旧版完整存档位于 training/archives/multimap_hybrid_20261010_145229.tar.gz。</p></main>'''
(bundle/'index.html').write_text(p)
source=bundle/'source';source.mkdir(exist_ok=True)
for f in ['train_neural_components.py','train_neural_navigation.py','evaluate_navigation.py','test_neural_navigation.py','finalize_neural_navigation.py','build_neural_report.py','NEURAL_NAVIGATION_DESIGN.md']:
    shutil.copy2(ROOT/f,source/f)
for d in ['neural_navigation','navrl']:
    shutil.copytree(ROOT/d,source/d,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copy2(ROOT/'requirements.txt',source/'requirements.txt')
manifest={str(p.relative_to(bundle)):hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.rglob('*') if p.is_file() and p.name!='SHA256.json'}
(bundle/'SHA256.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(ROOT/'archives/full_model_navigation_20261010.tar.gz','w:gz') as t:t.add(bundle,arcname='full_model_navigation')
print('Report and learned prototype archive saved:',bundle)
