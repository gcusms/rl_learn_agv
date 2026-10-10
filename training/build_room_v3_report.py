"""Complete-task comparison for the selected room revision, preserving failures."""
import json,argparse
from pathlib import Path
from navrl.geometry import ROOT

def aggregate(root):
    cases=[json.loads((root/f'mission_{i}/metrics.json').read_text()) for i in range(3)]
    return dict(missions=3,successful_missions=sum(x['verified_success'] for x in cases),planned_targets=12,reached_targets=sum(x['targets_reached'] for x in cases),collisions=sum(x['collision_events'] for x in cases),guard_activations=sum(x['penetration_guard_activations'] for x in cases),cases=cases)

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,default=ROOT/'runs/full_model_navigation_v3');a=p.parse_args();out=a.bundle
    baseline=aggregate(ROOT/'runs/room_navigation_scripted');new=aggregate(out);validation=aggregate(ROOT/'runs/room_v3_selected_validation')
    report=dict(baseline=baseline,regression=new,changed_waypoint_validation=validation,training=json.loads((out/'config.json').read_text()),limitation='Three missions each on one known room map; regression tasks appeared in training, changed coordinates are not unseen-map generalization.')
    (out/'report.json').write_text(json.dumps(report,indent=2))
    rows=''
    for label,r in [('旧模型 · 同任务',baseline),('新模型 · 同任务',new),('新模型 · 新起点/导航点',validation)]:rows+=f"<tr><td>{label}</td><td>{r['successful_missions']}/3</td><td>{r['reached_targets']}/12</td><td>{r['collisions']}</td><td>{r['guard_activations']}</td></tr>"
    options=''.join(f'<option value="mission_{i}/replay.html">原任务 {i+1} · {new["cases"][i]["targets_reached"]}/4 点</option>' for i in range(3))
    options+=''.join(f'<option value="../room_v3_selected_validation/mission_{i}/replay.html">新导航点任务 {i+1} · {validation["cases"][i]["targets_reached"]}/4 点</option>' for i in range(3))
    html=f'''<!doctype html><meta charset="utf-8"><title>房间导航续训对比</title><style>body{{font:16px system-ui;background:#eef2f6;color:#233347;margin:20px}}main{{max-width:1280px;margin:auto}}section{{background:white;padding:20px;border-radius:12px;margin:16px 0}}table{{width:100%;border-collapse:collapse}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}iframe{{width:100%;height:1180px;border:0}}select{{padding:10px}}video{{width:100%}}</style><main><h1>四房间导航 · 策略续训与实际对比</h1><section><p>原模型完整保存。新版本增加路线朝向、近期移动情况和前方走廊距离观测；已有神经路线仍有效时保持路线，避免局部目标反复跳变。加入离线纠错样本、转向覆盖样本及 8,192 步 PPO 续训。另对齐地图网格中心，并按有限视野维护障碍物记忆。这些普通程序处理不替机器人输出速度。</p><p>激光定位和神经规划权重沿用存档版本；本次重点训练动作策略。运行时没有 AMCL、机器人 A* 或动作接管，速度与转向均由训练模型输出，包括最多 0.08 m/s 的低速倒退。后方没有激光覆盖，倒退表现仍须持续验证。</p><p>柱子实体在小车接近时走入堵路点，停留 12 秒后退回原位。小车和柱子受 MuJoCo 接触约束；防穿模保护计数单独列出。</p></section><section><h2>实际测试结果</h2><table><thead><tr><th>版本 / 场景</th><th>完整任务成功</th><th>抵达计划点</th><th>碰撞事件</th><th>防穿模触发</th></tr></thead><tbody>{rows}</tbody></table><p>每组只有 3 个连续四点任务。原任务已用于训练纠错，属于回归测试；另一组换了起点、朝向、导航点和随机种子，但地图布局相同。这些结果不能证明任意新场景的成功率。失败和超时均保留。旧任务 1 本版首个目标就超时，旧任务 3 仍在第三点超时；虽整体完成数提高，仍有退化，不能作为可靠导航模型。</p></section><section><h2>完整 MuJoCo 成功任务录像 · 五倍速</h2><p>展示新导航点任务 3 连续访问全部 4 个目标的实际过程，模拟时间约 224 秒；不是所有任务都成功。右侧显示复刻 URDF 机器人与机械臂，激光保持实际约 219.44° 扫描范围。</p><video controls preload="metadata" poster="../room_v3_complete_demo/first_frame.png" src="../room_v3_complete_demo/navigation.mp4"></video></section><section><h2>全部实际轨迹</h2><p>青色虚线显示规划路线，扫描来自实际约 219.44° SCLIDAR。拖动进度条可以查看完整任务和每次换目标。</p><select id="mission">{options}</select><iframe id="replay" src="mission_0/replay.html"></iframe></section><p><a href="report.json">完整测试及训练记录</a> · <a href="config.json">模型配置</a> · <a href="../room_navigation_scripted/index.html">旧模型全部结果</a></p></main><script>document.getElementById('mission').onchange=e=>document.getElementById('replay').src=e.target.value;</script>'''
    (out/'index.html').write_text(html)
    print(json.dumps({k:{f:v for f,v in report[k].items() if f!='cases'} for k in ['baseline','regression','changed_waypoint_validation']},indent=2))

if __name__=='__main__':main()
