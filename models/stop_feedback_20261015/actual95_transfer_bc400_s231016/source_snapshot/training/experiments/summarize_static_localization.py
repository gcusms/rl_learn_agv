"""Report every frozen case, including weak geometry and failed regressions."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    reference=json.loads((root/'fixed_28/RESULTS.json').read_text())
    accelerated=json.loads((root/'native_final_28/RESULTS.json').read_text())
    if not reference['completed'] or not accelerated['completed'] or not accelerated['all_parity']:raise ValueError('Incomplete experiment/parity')
    native={r['id']:r for r in accelerated['rows']};rows=[]
    labels=['legacy','v5','fast_static','fast_native','amcl_standard','amcl_skip','amcl_budget_gated']
    for row in reference['cases']:
        results={k:row['algorithms'][k] for k in ['legacy','v5','fast_static']}
        results['fast_native']=native[row['id']]['algorithms']['native']
        for label in labels[4:]:
            runs=[v for k,v in row['algorithms'].items() if k.startswith(label+'_')]
            results[label]={key:float(np.mean([v[key] for v in runs])) for key in ['xy_p95_m','xy_rmse_m','yaw_p95_rad','cpu_mean_ms','cpu_p95_ms','update_p95_ms']}
        rows.append(dict(id=row['id'],split=row['split'],mode=row['mode'],frames=row['frames'],algorithms=results,physical_events=row['physical_events']))
    summary={}
    for subset,selected in [('all',rows),('development',[r for r in rows if r['split']=='development']),('validation',[r for r in rows if r['split']=='validation'])]:
        summary[subset]={label:dict(cases=len(selected),median_case_xy_p95_m=float(np.median([r['algorithms'][label]['xy_p95_m'] for r in selected])),
            worst_case_xy_p95_m=float(max(r['algorithms'][label]['xy_p95_m'] for r in selected)),
            frame_weighted_cpu_mean_ms=float(np.average([r['algorithms'][label]['cpu_mean_ms'] for r in selected],weights=[r['frames'] for r in selected])),
            median_case_cpu_p95_ms=float(np.median([r['algorithms'][label]['cpu_p95_ms'] for r in selected]))) for label in labels}
        # Long trajectories contain many more frames. Report their contribution
        # explicitly instead of hiding it behind a median of short episodes.
        for label in labels:
            pooled=[]
            for row in selected:
                truth=np.load(root/'sensor_cache'/(row['id']+'.npz'))['truth']
                if label.startswith('amcl_'):
                    estimates=[]
                    for seed in [11,29,47]:
                        values=np.genfromtxt(root/'fixed_28'/(row['id']+f'_{label}_{seed}.csv'),delimiter=',',names=True)
                        estimates.append(np.c_[values['x'],values['y']])
                else:
                    location=root/('native_final_28' if label=='fast_native' else 'fixed_28')/(row['id']+'_'+('native' if label=='fast_native' else label)+'.npz')
                    estimates=[np.load(location)['estimate'][:,:2]]
                for estimate in estimates:pooled.extend(np.linalg.norm(estimate-truth[:,:2],axis=1))
            summary[subset][label]['pooled_frame_xy_p95_m']=float(np.quantile(pooled,.95))
    comparisons={}
    for baseline in ['v5','amcl_standard','amcl_skip','amcl_budget_gated']:
        selected=[r for r in rows if r['split']=='validation']
        comparisons[baseline]=dict(cases=len(selected),accuracy_wins=sum(r['algorithms']['fast_native']['xy_p95_m']<r['algorithms'][baseline]['xy_p95_m'] for r in selected),
            cpu_wins=sum(r['algorithms']['fast_native']['cpu_mean_ms']<r['algorithms'][baseline]['cpu_mean_ms'] for r in selected),
            both_wins=sum(r['algorithms']['fast_native']['xy_p95_m']<r['algorithms'][baseline]['xy_p95_m'] and r['algorithms']['fast_native']['cpu_mean_ms']<r['algorithms'][baseline]['cpu_mean_ms'] for r in selected))
    real=json.loads((root/'recorded_fast_static/replay.json').read_text());navigation_path=root/'closed_loop_12/RESULTS.json'
    navigation=json.loads(navigation_path.read_text()) if navigation_path.exists() else None
    continuation=root/'closed_loop_remaining/RESULTS.json'
    if navigation is not None and continuation.exists():
        next_part=json.loads(continuation.read_text());combined=navigation['rows']+next_part['rows']
        expected=json.loads((root/'closed_loop_cases.json').read_text());ids=[r['case_id'] for r in combined]
        if len(ids)!=len(set(ids)) or not set(ids)<=set(c['id'] for c in expected):raise ValueError('Invalid resumed case set')
        navigation=dict(completed=next_part['completed'] and set(ids)==set(c['id'] for c in expected),rows=combined,
            scope=navigation['scope'],deployment_approved=False,sources=[str(navigation_path),str(continuation)])
        (root/'closed_loop_combined.json').write_text(json.dumps(navigation,indent=2))
    old_navigation=[];old_manifests=[]
    for directory in ['preserved_turn_historical_12','preserved_turn_navigation_48']:
        previous=root.parent/directory
        old_cases={c['id']:c for c in json.loads((previous/'cases.json').read_text())}
        wanted={c['id']:c for c in json.loads((root/'closed_loop_cases.json').read_text())}
        old_manifests.append(json.loads((previous/'PROVENANCE.json').read_text())['bundle_manifest_sha256'])
        for row in json.loads((previous/'RESULTS.json').read_text())['rows']:
            if row['case_id'] in wanted:
                if old_cases[row['case_id']]!=wanted[row['case_id']]:raise ValueError('Historical task mismatch')
                old_navigation.append(row)
    if len(set(old_manifests))!=1 or len(old_navigation)!=12:raise ValueError('Historical baseline identity mismatch')
    result=dict(completed=True,summary=summary,comparisons=comparisons,cases=rows,native_all_parity=accelerated['all_parity'],
        parity_frames=sum(r['frames'] for r in accelerated['rows']),
        maximum_native_pose_difference=max(r['maximum_pose_difference'] for r in accelerated['rows']),
        real_record_frames=real['frames'],closed_loop_completed=bool(navigation and navigation['completed']),deployment_approved=False)
    result['historical_navigation_baseline']=dict(cases=len(old_navigation),success=sum(r['safe_success'] for r in old_navigation),
        bundle_manifest_sha256=old_manifests[0],source='same archived v6 candidate, historical runs, not newly rerun')
    (root/'SUMMARY.json').write_text(json.dumps(result,indent=2))
    names={'legacy':'原定位器','v5':'上一版 v5','fast_static':'轻量 Python','fast_native':'轻量 C++ 内核','amcl_standard':'AMCL 300–1000','amcl_skip':'AMCL 300–1000＋跳束','amcl_budget_gated':'AMCL 100–300＋跳束＋运动触发'}
    text=['# 定位优化与 AMCL 对照（2026-10-15）','',
        '**范围：已知二维地图、给定初始位姿后的局部跟踪。不是全局重定位、SLAM 建图或真机绝对精度验收。**','',
        '**结论：定位 CPU 开销下降，但长距离压力测试和导航回归仍有退化，暂不替换部署模型。不能宣称全面优于 AMCL。**','',
        '## 已实现','',
        '- 保留完整雷达给避障；只将与静态地图相容的回波用于定位，减少新增障碍拉动位姿。',
        '- 用有里程计先验的鲁棒 SE(2) 距离场匹配替代每帧神经候选位姿搜索；用几何信息矩阵识别弱约束方向。',
        '- 传播完整 3×3 协方差；缺帧/弱方向保留里程计预测和不确定性，重复时间戳不重复融合。协方差尚未实车标定。',
        '- 可选 Eigen C++ 内核；不在模型启动时编译。底盘轮廓、动作集合、router 和 PPO actor 文件保持原始字节，方向停车仍由原策略执行。',
        '- 两个候选独立保存。原部署包未替换；这轮没有进行 PPO 训练。定位前端是几何融合，不能宣称是新训练的神经定位能力。','',
        '## 固定验证结果','',
        '28 条轨迹：6 条开发、22 条预先冻结的验证轨迹；全部算法使用相同扫描、里程计和初始先验。真值来自独立 MuJoCo 位姿，仅评价使用。短轨迹包含初值偏差、未入图障碍、缺束/缺帧、指令积分失配。长轨迹含长墙、走廊、稀疏和进入开放区。','',
        '长程压力测试人为叠加 8% 平移比例误差、2.5% 航向增量误差及横向 1 mm/s 偏差，持续约 93 秒；这不是对实车误差的标定或预测。','',
        '表中精度为各轨迹位置误差 P95 的中位数；CPU 为按帧加权的单次更新平均 CPU 时间。AMCL 每种配置运行 3 个运动/重采样种子，先按任务平均。','',
        '| 方法 | 验证轨迹 P95 中位数 | 全部验证帧 P95 | 最差轨迹 P95 | 平均 CPU/帧 |','|---|---:|---:|---:|---:|']
    for label in labels:
        m=summary['validation'][label];text.append(f"| {names[label]} | {m['median_case_xy_p95_m']*100:.2f} cm | {m['pooled_frame_xy_p95_m']*100:.1f} cm | {m['worst_case_xy_p95_m']*100:.1f} cm | {m['frame_weighted_cpu_mean_ms']:.3f} ms |")
    text+=['','**不能只看中位数。长距离弱几何情况如下：**','','| 场景 | v5 P95 | 新候选 P95 | AMCL 跳束 P95 |','|---|---:|---:|---:|']
    for row in rows:
        if row['mode']=='long_drift':text.append(f"| {row['id']} | {row['algorithms']['v5']['xy_p95_m']*100:.1f} cm | {row['algorithms']['fast_native']['xy_p95_m']*100:.1f} cm | {row['algorithms']['amcl_skip']['xy_p95_m']*100:.1f} cm |")
    text+=['','## 公平性与边界','',
        '- AMCL 直接链接本机 Nav2 1.1.19 的真实粒子滤波、全向运动模型和 likelihood_field_prob 库；没有将 ROS 进程启动、传输或旧适配器 50 ms 轮询算作算法耗时。',
        '- 相同 60 束索引、10 m 量程、初始标准差 0.12/0.12 m、0.08 rad。AMCL 地图坐标转换严格遵循该版本 convertMap，包括其栅格取整约定；新候选使用连续距离场插值，亚栅格差异是本对照的一部分。',
        '- 300–1000 粒子配置每帧更新；另测 100–300 粒子、0.02 m/rad 运动触发的节省算力配置。不是穷尽 AMCL 调参，也不覆盖其全局多假设优势。',
        '- C++ 内核与 Python 参考逐帧核对位姿和标准差；不会以降低迭代次数或更换动作换取速度。',
        '- CPU 时间不包含文件 IO、限额等待和预处理地图；这些启动成本单独记录。当前电脑成绩不能当作 Jetson Orin Nano 实测。',
        '- Python/SciPy 定位进程仍比 AMCL 裸 C++ 核心占更多内存；不将 CPU 减少表述为所有资源都更省。',
        '- 所有长轨迹保留。手工指令如发生物理接触会截断，并在原始记录标明；这不构成机器人自主导航成绩。',
        '- 未见真实地图泛化、真实绝对位姿误差、绑架重定位、无地图持续建图和回环均未因此通过验收。','',
        '## 实车数据回放','',
        f"共 {real['frames']} 帧。没有独立参考位姿，只能检查纠偏行为及与里程计的一致性；不能以与里程计接近证明绝对准确。",'',
        '| 片段 | 原记录位移 | 新候选位移 | 输入里程计位移 |','|---|---:|---:|---:|']
    for row in real['missions']:
        if row['mission'] in (13,19):text.append(f"| mission {row['mission']} | {row['recorded_displacement_m']:.3f} m | {row['candidate_displacement_m']:.3f} m | {row['input_odometry_displacement_m']:.3f} m |")
    text+=['','## 导航回归','']
    if navigation:
        checked=navigation['rows'];text.append(f"已完成 {len(checked)}/12 个冻结任务，{sum(r['safe_success'] for r in checked)} 成功，{sum(r['collisions'] for r in checked)} 碰撞，{sum(r['timeout'] for r in checked)} 超时，{sum(r['false_arrival'] for r in checked)} 假到达。")
        text.append(f"同一组任务的归档 v6 候选历史参考为 {sum(r['safe_success'] for r in old_navigation)}/12 成功；两个来源的 bundle manifest 一致，任务内容逐项一致。历史参考未重新运行，因此不比较端到端耗时。")
        text.append('这组为开发回归，包含过去的困难案例和方向停车；不能替代完整部署验收。若失败增加，候选保持实验状态。')
        text.append('新增超时 `train_failure_w11_e7`：定位 P95 约 0.68 cm，但 700 帧中 269 帧有可行动作而策略主动等待，332 帧只剩停止；99 次采样运动动作均实际执行。不能把所有停车归为“动作被安全层改写”。需补训新定位输入下的避障/等待退出行为，不放宽车体保护。')
        smoke=root/'native_runtime_2/RESULTS.json'
        if smoke.exists():
            checks=json.loads(smoke.read_text());text.append(f"C++ 入口另外完成 {len(checks['rows'])} 项闭环停车检查，{sum(r['safe_success'] for r in checks['rows'])} 项成功；同任务与 Python 参考的动作编号及动作掩码逐帧一致。两者定位数学结果在全部 3,221 帧中最大差异低于 2×10⁻¹⁴。")
    text+=['','## 使用与下一步','',
        '候选位于 `models/localization_mapping_20261015/fast_static_candidate/portable` 和 `fast_native_candidate/portable`。需要本分支的新运行代码；不能只拷贝 actor 文件。C++ 库当前为本机架构，Orin 上需用源码重新编译并验证 manifest。',
        '先确认长距离误差、错误高置信度、导航回归和时间戳语义，再做真机受控对照。若要比较真实精度，需要独立参考轨迹；当前录制无法提供这一指标。',
        '无地图建图仍需单独修复/验证 SLAM 前端退化与图优化错误约束；本轮没有通过限制输出幅度掩盖地图漂移。','',
        '## 复现与原始记录','',
        '- `sensor_cache/TASKS.json`：冻结任务；NPZ 保存真实扫描、传感器输入及独立评价真值。',
        '- `fixed_28/RESULTS.json`：逐任务全部对照；`source_snapshot` 固定对照时的源码，AMCL 参数与哈希在报告中。',
        '- `native_final_28/RESULTS.json`：C++/Python 一致性及耗时。',
        '- `recorded_fast_static/replay.json`：用户实录回放。',
        '- `closed_loop_combined.json`：导航/停车回归（修复限额等待后接续未完成任务）。','',
        '功能回归：24 项测试通过；同时修复了资源限额程序追逐微小等待余量、反复短睡眠的问题。该修复降低实验调度开销，不应计为定位算法精度提升。','',
        '参考：[Nav2 AMCL 配置](https://docs.nav2.org/rolling/configuration_and_development/configuration_guide/others/configuring_amcl/)、[Nav2 1.1.19 AMCL 源码](https://github.com/ros-navigation/navigation2/tree/1.1.19/nav2_amcl)、[几何退化识别论文](https://www.cs.cmu.edu/~kaess/pub/Zhang16icra.pdf)。论文提供退化处理思路，并非本候选的精度证明。','']
    (root/'REPORT.md').write_text('\n'.join(text))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));selected=[r for r in rows if r['split']=='validation']
    for label,color in [('v5','#777777'),('fast_native','#16826b'),('amcl_skip','#cf742b'),('amcl_budget_gated','#6a58aa')]:
        axes[0].plot(range(len(selected)),[r['algorithms'][label]['xy_p95_m']*100 for r in selected],'.-',label=label,color=color)
        axes[1].scatter([r['algorithms'][label]['cpu_mean_ms'] for r in selected],[r['algorithms'][label]['xy_p95_m']*100 for r in selected],label=label,color=color,s=22,alpha=.75)
    axes[0].set(xlabel='Frozen validation trajectory',ylabel='Position error P95 (cm)',yscale='log')
    axes[1].set(xlabel='Mean CPU time per update (ms)',ylabel='Position error P95 (cm)',yscale='log',xscale='log')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Known-map local tracking — simulated sensor replay, not real-robot accuracy')
    fig.tight_layout();fig.savefig(root/'comparison.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(summary=summary['validation'],comparisons=comparisons),indent=2))


if __name__=='__main__':main()
