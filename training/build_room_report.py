"""Publish all room missions, including failures, with no selection in aggregate metrics."""
import json
from pathlib import Path
from navrl.geometry import ROOT

out=ROOT/'runs/room_navigation'
results=[json.loads((out/f'mission_{i}/metrics.json').read_text()) for i in range(3)]
report=dict(scene='Four connected rooms, 1.8 m doors',mission_count=3,mission_successes=sum(r['verified_success'] for r in results),mission_success_rate=sum(r['verified_success'] for r in results)/3,
            planned_targets=sum(r['targets_total'] for r in results),reached_targets=sum(r['targets_reached'] for r in results),attempted_targets=sum(len(r['legs']) for r in results),
            frozen_model='full_model_navigation_v2',runtime_robot_amcl=False,runtime_robot_astar=False,localizer_reset_between_targets=False,
            success_criterion='Every one of four targets must be reached; each model arrival is checked against actual distance <= 0.30 m; collisions allowed',
            episodes=results,limitation='Three fixed missions on one new room map; not a broad generalization benchmark. Mission 2 hit a door jamb and timed out; all results retained.')
(out/'metrics.json').write_text(json.dumps(report,indent=2))
rows=''
for i,r in enumerate(results):
    distances=' / '.join(f"{l['true_goal_distance_m']:.3f}" for l in r['legs'])
    link='../room_navigation_demo/replay.html' if i==0 else f'mission_{i}/replay.html'
    rows+=f'<tr><td><a href="{link}">任务 {i+1}：交互回放</a></td><td>{r["targets_reached"]}/4</td><td>{"完成" if r["verified_success"] else "第二点超时"}</td><td>{r["collision_events"]}</td><td>{r["mean_localization_error_m"]*100:.2f} cm</td><td>{distances}</td></tr>'
scene=json.loads((ROOT/'assets/room_navigation/physical_scene.json').read_text())
angles='，'.join(f'{k}={v:.4f}' for k,v in scene['arm_home']['joint_positions_rad'].items())
html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>四房间连续导航测试</title>
<style>body{{font:16px system-ui;background:#eef2f6;color:#233347;margin:20px}}main{{max-width:1280px;margin:auto}}section{{background:white;border-radius:12px;padding:20px;margin:16px 0}}video{{width:100%;background:#14191e}}table{{border-collapse:collapse;width:100%}}th,td{{padding:12px;border-bottom:1px solid #dce3eb;text-align:left}}a{{color:#1765a9}}.stat{{font-size:24px;font-weight:600}}button,select{{padding:10px}}iframe{{width:100%;height:1150px;border:0}}details{{margin:14px 0}}code{{font-size:13px}}</style>
<main><h1>四房间 · 连续多点导航</h1><p>10 × 8 米，四个相连房间，1.8 米门口；移动实体柱子接近小车会让行。门口双柱临时堵路 15 秒后连续移动离开。</p>
<section><p class="stat">完整任务成功：2 / 3（66.7%） · 抵达导航点：9 / 12 个计划点</p><p>每组连续访问 4 个导航点；位置、里程计和学习定位状态持续保留。共尝试 10 个点，其中 9 个抵达；失败任务提前终止，后续 2 个未尝试。</p><p>使用存档模型的学习定位、神经路线模块和 PPO 决策；运行时机器人没有 AMCL 或 A*。测试未重新训练权重。</p><table><thead><tr><th>任务</th><th>抵达点数</th><th>结果</th><th>碰撞事件</th><th>平均定位误差</th><th>各点实际距离（米）</th></tr></thead><tbody>{rows}</tbody></table><p>任务 2 在第二个导航点途中接触门框，之后动作没有解困并超时。仿真的防穿模保护触发 {results[1]['penetration_guard_activations']} 次，该失败计入统计。小样本结果不能证明任意房间布局均可靠。</p></section>
<section><h2>MuJoCo 实际录像 · 任务 1</h2><p>依次访问左上 → 右上 → 右下 → 左下。编号标记导航点；青色为模型规划路线，橙色为实际轨迹；显示真实 219.44° 激光扫描和机器人机械臂细节。模拟总时长 287.4 秒。</p><video controls preload="metadata" poster="../room_navigation_demo/first_frame.png" src="../room_navigation_demo/navigation.mp4"></video></section>
<section><h2>交互回放</h2><p>可拖动进度条查看任何时刻的扫描、定位、当前目标、规划路径和模型动作，包括失败任务。</p><select id="mission"><option value="../room_navigation_demo/replay.html">任务 1 · 全部成功</option><option value="mission_1/replay.html">任务 2 · 失败记录</option><option value="mission_2/replay.html">任务 3 · 全部成功</option></select><iframe id="replay" src="../room_navigation_demo/replay.html"></iframe></section>
<section><h2>机械臂初始化修正</h2><p>读取源码 <code>agv/driver/ros_arm_controller/config/init_pose.yaml</code>：机械臂基座坐标中末端 (95, 0, 216) mm，pitch −50°，roll 0°，夹爪打开。通过 CAD 逆运动学求关节角，按 URDF 的关节轴和旋转顺序应用；MuJoCo 中末端位置、俯仰角测试通过。旧版本的全零关节姿态已修正。</p><details><summary>关节角与启动链说明</summary><p><code>{angles}</code></p><p>启动链：controller.launch.py → odom_publisher.launch.py → ros_robot_controller.launch.py → arm_controller。当前源码的 auto_home_on_startup 默认 false；归位逻辑和姿态配置在 arm_controller 中。此次仅离线复刻归位姿态，没有启动实车控制节点。</p></details><p><a href="../../assets/room_navigation/physical_scene.json">场景与导航点配置</a> · <a href="metrics.json">完整测试数据</a></p></section></main><script>document.getElementById('mission').onchange=e=>document.getElementById('replay').src=e.target.value;</script></html>'''
(out/'index.html').write_text(html)
print(json.dumps({k:v for k,v in report.items() if k!='episodes'},ensure_ascii=False,indent=2))
