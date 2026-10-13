"""Validate remote checkpoints, render real traces, and persist all outcomes."""
import json,time,subprocess,sys,os,shutil,hashlib,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent
RUN=ROOT/'runs/remote_room_100k';OUT=ROOT/'runs/remote_room_100k_report'
PERSIST=Path('/home/featurize/work/rl_navigation_100k_20261010');PY=sys.executable
OUT.mkdir(exist_ok=True)
def status(phase,**more):
    d=dict(phase=phase,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**more)
    (OUT/'status.json').write_text(json.dumps(d,indent=2))
    shutil.copy2(OUT/'status.json',PERSIST/'evaluation-status.json');print(d,flush=True)
def evaluate(args):
    name,policy,components,scene,task=args;dest=OUT/name;dest.mkdir(exist_ok=True)
    cmd=[PY,str(ROOT/'evaluate_room_navigation.py'),'--revision','v3','--policy',str(policy),'--components',str(components),'--scene-dir',str(scene),'--output-root',str(dest),'--task',str(task),'--seed-base','31000' if name.endswith('validation') else '27000']
    with (dest/f'task_{task}.log').open('w') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=PROJECT)
    print('EVALUATED',name,task,json.loads((dest/f'mission_{task}/metrics.json').read_text())['targets_reached'],flush=True)
def aggregate(name):
    cases=[json.loads((OUT/name/f'mission_{i}/metrics.json').read_text()) for i in range(3)]
    return dict(missions=3,successful_missions=sum(r['verified_success'] for r in cases),reached_targets=sum(r['targets_reached'] for r in cases),planned_targets=12,collision_events=sum(r['collision_events'] for r in cases),penetration_guard_activations=sum(r['penetration_guard_activations'] for r in cases),cases=cases)
def main():
    status('waiting_for_training');deadline=time.monotonic()+10800
    while True:
        if (RUN/'config.json').exists():
            c=json.loads((RUN/'config.json').read_text())
            if c.get('status')=='complete':break
        if time.monotonic()>deadline:raise TimeoutError('Training not complete after three hours')
        time.sleep(10)
    assert c['additional_ppo_steps_completed']==100000
    status('evaluating',completed_training_steps=100000)
    parent=ROOT/'runs/full_model_navigation_v3';room=ROOT/'assets/room_navigation_scripted';validation=ROOT/'assets/room_navigation_v3_validation'
    jobs=[]
    for name,policy,components,scene in [('parent_validation',parent/'policy.zip',parent,validation),('checkpoint50_validation',RUN/'policy_50000.zip',RUN,validation),('final_validation',RUN/'policy.zip',RUN,validation),('final_regression',RUN/'policy.zip',RUN,room)]:
        jobs.extend((name,policy,components,scene,i) for i in range(3))
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(evaluate,jobs))
    stats={n:aggregate(n) for n in ['parent_validation','checkpoint50_validation','final_validation','final_regression']}
    def rank(s):return (s['successful_missions'],s['reached_targets'],-s['collision_events'],-s['penetration_guard_activations'],-sum(r['seconds'] for r in s['cases']))
    selected=max(['parent_validation','checkpoint50_validation','final_validation'],key=lambda n:rank(stats[n]))
    report=dict(training=c,results=stats,validation_selection=selected,limitation='One known room layout. Validation goals/seeds differ from online training but were used in earlier development. Validation is used for checkpoint selection; original regression tasks appeared in previous correction samples. No unseen-map success claim.',benchmark={d:json.loads((ROOT/f'runs/remote_smoke_{d}/config.json').read_text()) for d in ['cpu','cuda']})
    (OUT/'report.json').write_text(json.dumps(report,indent=2));status('rendering',selected_by_validation=selected)
    for name in ['final_validation','final_regression']:
        for i in range(3):
            trace=OUT/name/f'mission_{i}/replay.json'
            subprocess.run([PY,str(ROOT/'render_success.py'),str(trace),'--output',str(trace.parent),'--html-only','--allow-failure'],check=True,cwd=PROJECT,env={**os.environ,'MUJOCO_GL':'egl'})
    cases=stats['final_validation']['cases'];success=[i for i,r in enumerate(cases) if r['verified_success']]
    i=min(success,key=lambda i:cases[i]['seconds']) if success else max(range(3),key=lambda i:cases[i]['targets_reached'])
    cmd=[PY,str(ROOT/'render_success.py'),str(OUT/f'final_validation/mission_{i}/replay.json'),'--output',str(OUT/'demo'),'--stride','4','--render-scale','.7','--playback-speed','5']
    if not success:cmd.extend(['--allow-failure','--max-time','90'])
    subprocess.run(cmd,check=True,cwd=PROJECT,env={**os.environ,'MUJOCO_GL':'egl'})
    build_page(c,stats,selected,bool(success),i)
    shutil.copytree(OUT,PERSIST/'report',dirs_exist_ok=True)
    for n in ['policy.zip','localizer.pt','router.pt','config.json']:shutil.copy2(RUN/n,PERSIST/n)
    shutil.copytree(ROOT/'remote',PERSIST/'source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('*.tar.gz','__pycache__'))
    status('complete',completed_training_steps=100000,final_validation_successes=stats['final_validation']['successful_missions'],final_regression_successes=stats['final_regression']['successful_missions'],recommended_version=selected)
    shutil.copy2(OUT/'status.json',PERSIST/'report/status.json')
    manifest={str(p.relative_to(PERSIST)):hashlib.sha256(p.read_bytes()).hexdigest() for p in PERSIST.rglob('*') if p.is_file() and p.name!='SHA256.json'}
    (PERSIST/'SHA256.json').write_text(json.dumps(manifest,indent=2));print('ALL COMPLETE',OUT,flush=True)
def build_page(c,stats,selected,success,i):
    labels={'parent_validation':'续训前 · 新导航点','checkpoint50_validation':'5 万步 · 新导航点','final_validation':'10 万步 · 新导航点','final_regression':'10 万步 · 原任务'}
    rows=''
    for name,s in stats.items():
        rows+=f"<tr><td>{labels[name]}</td><td>{s['successful_missions']}/3</td><td>{s['reached_targets']}/12</td><td>{s['collision_events']}</td><td>{s['penetration_guard_activations']}</td></tr>"
    options=''.join(f'<option value="{name}/mission_{k}/replay.html">{labels[name]} · 任务 {k+1} · {stats[name]["cases"][k]["targets_reached"]}/4 点</option>' for name in ['final_validation','final_regression'] for k in range(3))
    description=f'最终模型的新导航点任务 {i+1}，连续完成全部四点；录像五倍速。' if success else '最终模型尚未完整成功，录像为部分任务片段；全部超时结果保留在表中。'
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>远程 10 万步导航训练</title><style>body{{font:16px system-ui;background:#eef2f6;color:#233347;margin:20px}}main{{max-width:1280px;margin:auto}}section{{background:white;padding:20px;margin:16px 0;border-radius:12px}}table{{width:100%;border-collapse:collapse}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}video{{width:100%}}iframe{{width:100%;height:1180px;border:0}}select{{padding:10px}}</style><main><h1>远程平台 · 10 万步导航策略续训</h1><section><p>RTX 3080 / Gold 6142 平台：CUDA 短测通过。CPU 并行短测较快，因此正式训练使用 CPU、四个仿真进程；MuJoCo 录像使用 GPU EGL 渲染。在 v3 存档基础上追加 100,000 步 PPO，定位和神经规划权重保持存档版本。</p><p>训练实际耗时 {c['wall_seconds']/60:.1f} 分钟。目标切换不重置机器人或定位。速度、倒退和转向由策略输出，运行时无 AMCL、机器人 A* 或控制接管；实体柱子接近时走入堵路点，停留后退回。</p></section><section><h2>实际完整任务测试</h2><table><thead><tr><th>版本 / 场景</th><th>完整成功</th><th>抵达目标</th><th>碰撞</th><th>防穿模保护</th></tr></thead><tbody>{rows}</tbody></table><p>三组验证任务推荐保留版本：{labels[selected]}。最终权重和所有检查点均单独保存。每组只有三个四点任务，地图布局相同，不能据此推断任意新地图成功率。验证任务用于选择版本，原任务已在先前纠错样本中出现。</p></section><section><h2>最终模型实际 MuJoCo 录像</h2><p>{description}</p><video controls preload="metadata" poster="demo/first_frame.png" src="demo/navigation.mp4"></video></section><section><h2>全部最终模型轨迹与实际激光</h2><p>扫描范围约 219.44°，保持实车 SCLIDAR 配置。拖动进度查看路线、激光、柱子运动和目标切换；失败任务也可完整回放。</p><select id="select">{options}</select><iframe id="replay" src="final_validation/mission_0/replay.html"></iframe></section><p><a href="report.json">训练与测试记录</a></p></main><script>document.getElementById('select').onchange=e=>document.getElementById('replay').src=e.target.value;</script></html>'''
    (OUT/'index.html').write_text(html)

if __name__=='__main__':
    try:main()
    except Exception as e:status('failed',error=str(e));traceback.print_exc();raise
