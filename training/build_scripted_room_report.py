"""Report every fixed doorway-intervention mission with actual measured outcomes."""
import json
from pathlib import Path
from navrl.geometry import ROOT
out=ROOT/'runs/room_navigation_scripted';results=[json.loads((out/f'mission_{i}/metrics.json').read_text()) for i in range(3)]
report=dict(mission_count=3,mission_successes=sum(r['verified_success'] for r in results),mission_success_rate=sum(r['verified_success'] for r in results)/3,
            planned_targets=12,reached_targets=sum(r['targets_reached'] for r in results),attempted_targets=sum(len(r['legs']) for r in results),episodes=results,
            obstacle_behavior='Approach-cross-hold-return once, then remain parked; never roam randomly',trigger_distance_m=2.2,hold_seconds_after_arrival=12,
            robot_weights='Unchanged full_model_navigation_v2',runtime_robot_amcl=False,runtime_robot_astar=False,
            limitation='Three fixed waypoint orders on one room map; all failures and contact/guard counters are retained')
(out/'metrics.json').write_text(json.dumps(report,indent=2))
rows='';options=''
for i,r in enumerate(results):
    link='../room_navigation_scripted_demo/replay.html' if i==0 else f'mission_{i}/replay.html'
    status='全部完成' if r['verified_success'] else f"第 {len(r['legs'])} 个目标未完成"
    blocks=r['column_block_events'];event='；'.join(f"C{b['column_index']+1}："+(f"{b['trigger_time_s']:.1f}s 走入，{b['entered_time_s']:.1f}s 开始停留，{b['release_time_s']:.1f}s 退回" if b.get('entered_time_s') is not None else '未完成进入堵路点') for b in blocks)
    rows+=f'<tr><td><a href="{link}">任务 {i+1}</a></td><td>{r["targets_reached"]}/4</td><td>{status}</td><td>{r["collision_events"]}</td><td>{r["penetration_guard_activations"]}</td><td>{event}</td></tr>'
    options+=f'<option value="{link}">任务 {i+1} · {status}</option>'
html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>四房间导航 · 定点临时堵路</title>
<style>body{{font:16px system-ui;background:#eef2f6;color:#233347;margin:20px}}main{{max-width:1280px;margin:auto}}section{{background:white;border-radius:12px;padding:20px;margin:16px 0}}video{{width:100%;background:#14191e}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;border-bottom:1px solid #dce3eb;text-align:left}}a{{color:#1765a9}}select{{padding:10px}}iframe{{width:100%;height:1150px;border:0}}.stat{{font-size:24px;font-weight:600}}</style>
<main><h1>四房间导航 · 定点临时堵路</h1><section><h2>障碍物现在怎样移动</h2><p>两根柱子先停在左侧门口旁的固定位置，小车距门口约 2.2 米时才连续走入堵路点。两根都到位后停留 12 秒，再沿原路退回门旁并停住。整个任务只执行一次，不再随机游走；小车不接近，就一直在旁边等待。</p><p>移动受实体碰撞和加速度约束，靠近小车时停下让行，不会瞬移或穿模。这是障碍物的场景脚本；机器人的定位、选路和动作仍来自存档模型。</p></section>
<section><p class="stat">完整任务成功：{report['mission_successes']} / 3（{report['mission_success_rate']*100:.1f}%） · 抵达导航点：{report['reached_targets']} / 12 个计划点</p><p>三组任务各连续访问 4 个点，切换目标不重置位置和定位；模型权重未改变。共尝试 {report['attempted_targets']} 个目标。表中保留所有接触和防穿模保护计数，小样本结果不代表任意场景的成功率。</p><table><thead><tr><th>回放</th><th>抵达点数</th><th>结果</th><th>碰撞事件</th><th>防穿模触发</th><th>实际堵路时间</th></tr></thead><tbody>{rows}</tbody></table></section>
<section><h2>MuJoCo 实际录像 · 任务 1 前 65 秒</h2><p>录像展示等待、走入、堵路、退回，以及第一个导航点的抵达；不是完整任务成功录像。完整任务后续超时，请在下方查看完整交互轨迹。青色：神经规划路线；橙色：实际轨迹；绿色射线：219.44° 激光扫描。右侧显示归位后的 URDF 机械臂和柱子状态。</p><video controls preload="metadata" poster="../room_navigation_scripted_demo/first_frame.png" src="../room_navigation_scripted_demo/navigation.mp4"></video></section>
<section><h2>交互轨迹回放</h2><p>W1/W2 是门旁等待位置，B1/B2 是堵路位置。拖动进度条可查看柱子等待、走入、堵路、退回的过程。</p><select id="mission">{options}</select><iframe id="replay" src="../room_navigation_scripted_demo/replay.html"></iframe></section><p><a href="metrics.json">完整测试数据</a> · <a href="../../assets/room_navigation_scripted/physical_scene.json">场景及导航点配置</a></p></main><script>document.getElementById('mission').onchange=e=>document.getElementById('replay').src=e.target.value;</script></html>'''
(out/'index.html').write_text(html)
print('Scripted-room missions:',report['mission_successes'],'/ 3; targets:',report['reached_targets'],'/ 12',flush=True)
