"""Audit and summarize the frozen constraint experiments without promoting a model."""
import hashlib
import json
from pathlib import Path
import numpy as np
from navrl.geometry import ROOT


def main():
    workspace=ROOT.parent
    parent=workspace/'reports/localization_mapping_20261015'
    out=parent/'constraints_repair_20261015'
    baseline_dir=parent/'sparse_repair_20261015/temporal_closed_loop_12'
    candidate_dir=out/'combined_closed_loop_12'
    baseline=json.loads((baseline_dir/'RESULTS.json').read_text())
    candidate=json.loads((candidate_dir/'RESULTS.json').read_text())
    uncertainty=json.loads((out/'uncertainty_28/RESULTS.json').read_text())
    tests=json.loads((out/'REGRESSION_TESTS.json').read_text())
    if not all(r['completed'] for r in (baseline,candidate,uncertainty)) or not tests['available_checks_passed']:
        raise RuntimeError('Complete successful checks before writing the final report')
    old={r['case_id']:r for r in baseline['rows']}
    new={r['case_id']:r for r in candidate['rows']}
    assert set(old)==set(new) and len(new)==12
    def cases_by_id(folder):return {c['id']:c for c in json.loads((folder/'cases.json').read_text())}
    assert cases_by_id(baseline_dir)==cases_by_id(candidate_dir)
    checks={}
    for name,digest in json.loads((candidate_dir/'PROVENANCE.json').read_text())['source_sha256'].items():
        checks[name]=hashlib.sha256((workspace/name).read_bytes()).hexdigest()==digest
    if not all(checks.values()):raise RuntimeError('Evaluated runtime source changed; inspect provenance')
    sources=workspace/'models/localization_mapping_20261015'
    old_bundle=sources/'temporal_native_candidate/portable'
    new_bundle=sources/'contour_mask_candidate/portable'
    preserved={name:(old_bundle/name).read_bytes()==(new_bundle/name).read_bytes()
        for name in ('actor.pt','router.pt','localizer.pt','localization_kernel.so','localization_contract.json','docking_contract.json')}
    assert all(preserved.values())
    def aggregate(rows):
        return dict(tasks=len(rows),success=sum(r['safe_success'] for r in rows),
            collisions=sum(r['collisions'] for r in rows),timeouts=sum(r['timeout'] for r in rows),
            false_arrivals=sum(r['false_arrival'] for r in rows))
    paired=[k for k in new if old[k]['safe_success'] and new[k]['safe_success'] and not k.startswith('localization_nav_')]
    efficiency={metric:dict(baseline=sum(old[k][metric] for k in paired),candidate=sum(new[k][metric] for k in paired))
        for metric in ('seconds','travel_m')}
    for values in efficiency.values():values['change_percent']=100*(values['candidate']/values['baseline']-1)
    summary=dict(completed=True,deployment_approved=False,weights_retrained=False,
        compared_with='temporal_native_candidate, not the original deployed quality recovery 2k model',
        scope='12 frozen development/regression tasks; 28 previously used localization replays; no new real-world ground truth',
        baseline=aggregate(list(old.values())),candidate=aggregate(list(new.values())),
        dynamic_baseline=aggregate([r for k,r in old.items() if not k.startswith('localization_nav_')]),
        dynamic_candidate=aggregate([r for k,r in new.items() if not k.startswith('localization_nav_')]),
        docking_baseline=aggregate([r for k,r in old.items() if k.startswith('localization_nav_')]),
        docking_candidate=aggregate([r for k,r in new.items() if k.startswith('localization_nav_')]),
        common_success_dynamic_cases=paired,efficiency=efficiency,
        uncertainty_maximum_pose_difference=max(r['maximum_pose_difference'] for r in uncertainty['cases']),
        unchanged_files=preserved,runtime_sources_match=all(checks.values()),tests=tests,
        candidate_resource=candidate['resource'],uncertainty_resource=uncertainty['resource'],
        remaining_failures=[k for k in new if not new[k]['safe_success']])
    # These are independent simulator labels for review, never policy inputs.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from navrl.map import StaticMap
    cases=cases_by_id(candidate_dir)
    fig,axes=plt.subplots(1,2,figsize=(11,5),constrained_layout=True)
    for axis,key in zip(axes,('train_failure_w11_e7','motion_profile_015')):
        case=cases[key];m=StaticMap(ROOT/case['scene']/'decision_map.yaml')
        gray=np.ones(m.occupied.shape);gray[m.unknown]=.78;gray[m.occupied]=.24
        axis.imshow(gray,cmap='gray',vmin=0,vmax=1,origin='upper',
            extent=[m.origin[0],m.origin[0]+m.width*m.res,m.origin[1],m.origin[1]+m.height*m.res])
        traces=[]
        for folder,row,color,label in ((baseline_dir,old[key],'#c44837','Previous temporal'),
                (candidate_dir,new[key],'#2467af','Constraint candidate')):
            data=np.load(folder/f"{key}_{row['variant']}.npz");xy=data['labels'][:,1:3];traces.append(xy)
            axis.plot(*xy.T,color=color,lw=2,label=label)
            axis.scatter(*xy[-1],color=color,marker='o' if row['safe_success'] else 'x',s=40)
            if folder==candidate_dir:
                for i in range(data['human_positions'].shape[1]):
                    axis.plot(*data['human_positions'][:,i].T,color='#bd8d22',ls='--',alpha=.7,lw=1,
                        label='Moving obstacle (candidate run)' if i==0 else None)
        start=np.asarray(case['start'][:2]);goal=np.asarray(case['goal'][:2])
        axis.scatter(*start,c='black',marker='s',s=30,label='Start');axis.scatter(*goal,c='#237f44',marker='*',s=110,label='Goal')
        bounds=np.vstack([*traces,goal,start]);low=bounds.min(axis=0)-.8;high=bounds.max(axis=0)+.8
        axis.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),xlabel='x (m)',ylabel='y (m)',aspect='equal',
            title=f"{key}\nPrevious: collision; candidate: {'reached' if new[key]['safe_success'] else 'collision'}")
        axis.grid(alpha=.15);axis.legend(fontsize=7,loc='best')
    fig.savefig(out/'trajectory_comparison.png',dpi=160);plt.close(fig)
    (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    decision=dict(deployment_approved=False,replace_previous_bundle=False,
        reasons=['One blind-sector stop/go collision remains',
            'Weak-geometry uncertainty budget is uncalibrated and can substantially increase stopping',
            'Desktop control latency still has excursions above the 0.2s control period',
            'Small reused development set; unknown-map SLAM and real absolute positioning not certified'],
        next_priority='Avoid entering a blind-sector near-contact state; measure uncertainty against independent physical reference before training or relaxing guards')
    (out/'PROMOTION_DECISION.json').write_text(json.dumps(decision,indent=2)+'\n')
    text=['# 动态动作约束与弱几何不确定性优化（2026-10-15）','',
        '本轮已完成代码、独立候选包和验证。相对上一轮 temporal 定位候选，12 个固定开发/回归场景从 **10/12 安全到达、2 次碰撞**变为 **11/12 安全到达、1 次碰撞**，均无超时和假到达。只修复了其中一个原有碰撞；**未通过部署验收，原部署模型未替换**。',
        '', '## 修改内容','',
        '- 用当前雷达、连续里程计和实际反馈速度预测短时碰撞，在 actor 选择动作前屏蔽危险候选。跟踪使用 odom 坐标，地图定位修正不会直接制造障碍运动。',
        '- 缓存同一传感器时刻的预测；观测中的动作探测、训练掩码和执行检查共享该结果。保留原有静态/雷达制动检查、0.04m 安全余量、真实车体和 12 个动作。停止动作始终可选。',
        '- 累计比较可见轮廓，减轻逐渐遮挡造成的速度偏差；运动假设有明确的原始测量年龄和 1.2s 上限。它仍只描述可见点轮廓，不保证完整物体形状、突变运动或盲区安全。',
        '- 在缺少约束的方向累计里程计误差预算，避免用大量小步积分得到过于乐观的不确定性。该预算未标定；不改变位姿估计本身。',
        '- 更新动作合同校验、导出及加载流程，防止模型和硬件配置混用不同版本的掩码。actor、router、localizer 权重、定位核和方向停靠合同逐字节保持一致。本轮没有增加 PPO 步数。',
        '', '## 闭环对照','',
        '两组使用相同的 12 个任务、种子、车体、动作和物理到达/碰撞标准。仿真真值只用于评分和绘图，不进入新增约束。该集合已用于开发，不能称为新的泛化测试。',
        '', '![一个已修复和一个未解决场景的物理轨迹对照](trajectory_comparison.png)',
        '', '| 项目 | 上一轮 temporal 候选 | 本轮组合约束候选 |','|---|---:|---:|',
        '| 总安全到达 | 10/12 | 11/12 |','| 动态场景到达 | 4/6 | 5/6 |',
        '| 停靠/近目标回归 | 6/6 | 6/6 |','| 碰撞 | 2 | 1 |','| 超时 / 假到达 | 0 / 0 | 0 / 0 |',
        '', '| 场景 | 上一轮 | 本轮 | 本轮时间 / 路程 |','|---|---|---|---:|']
    for key,row in new.items():
        result=lambda r:'到达' if r['safe_success'] else '碰撞' if r['collisions'] else '超时' if r['timeout'] else '未通过'
        text.append(f"| {key} | {result(old[key])} | {result(row)} | {row['seconds']:.1f}s / {row['travel_m']:.3f}m |")
    text.extend(['',
        f"共同成功的 {len(paired)} 个动态场景：总路程变化 {efficiency['travel_m']['change_percent']:+.1f}%，总任务时间变化 {efficiency['seconds']['change_percent']:+.1f}%。约束改善了一个碰撞案例，但带来了更多等待；这不代表整体最短路或最快到达。",
        '', f"本轮各场景端到端 P95 耗时为 {min(r['runtime_ms']['p95'] for r in new.values()):.1f}～{max(r['runtime_ms']['p95'] for r in new.values()):.1f}ms，最大单步 {max(r['runtime_ms']['maximum'] for r in new.values()):.1f}ms。它包括定位、路由、策略和安全检查；桌面数据不能替代 Orin 实测，也不能用几毫秒的定位耗时代替全流程延迟。",
        '', '## 定位回放：改善的是风险提示，不是精度','',
        '28 组冻结传感器回放中，新增预算前后的位姿最大差异为 0。24 组短回放没有新增定位不确定停车；4 组长距离弱几何回放则明显更保守。下列停车比例是在相同记录轨迹上按现有阈值计算，不能直接视为闭环停车结果。',
        '', '| 长距离场景 | 原错误高置信度帧占比 | 新占比 | 新定位不确定帧占比 |','|---|---:|---:|---:|'])
    for row in uncertainty['cases']:
        if row['id'].startswith('long'):
            text.append(f"| {row['id']} | {100*row['original']['erroneous_trusted_fraction']:.1f}% | {100*row['candidate']['erroneous_trusted_fraction']:.1f}% | {100*row['candidate_guard_uncertain_fraction']:.1f}% |")
    text.extend(['',
        '稀疏场景中错误高置信度从 18.3% 降到 0，但不确定帧达到 87.5%。这是拒绝过度自信的保守处理，不是定位误差消失，也不足以证明校准正确。本轮没有新的 AMCL 精度或算力对照；回放 RESULTS 中 original 的耗时列复用了 candidate 测量，不应做速度比较。',
        '', '## 未解决的边界与下一步','',
        'motion_profile_015 仍碰撞：障碍物停走后进入侧后方盲区，当前短时预测与停车不足以规避接触。新候选已出现仅剩停止可选的时段；需要在接近盲区之前保留可行的绕行动作，并补足遮挡下轮廓/运动假设评估，不能通过取消停车保护换取通过。',
        '',
        '弱几何下应先用独立参考位姿测量漂移与误差预算，区分指令积分和轮速反馈，再决定方向融合或针对性训练。已有地图中给定初始位置的测试，不代表未知环境 SLAM、无特征区域长期不漂移或全局重定位已经完成。',
        '', '建议下一轮只针对盲区停走碰撞与误差预算标定做对照；通过后再做相同动作合同下的 PPO 微调。当前不宜直接用大量 PPO 步数掩盖这些边界。',
        '', '## 验证与产物','',
        f"原有相关回归 {tests['passed']} 项通过；另有 {tests['blocked']} 项依赖已缺失的历史模型/训练日志，未能完成，不计为通过。新增预测/不确定性测试 10 项通过，覆盖地图修正不影响 odom 跟踪、同帧缓存、丢帧过期、合同混用拒绝、掩码收紧且与执行一致、按物理时间累计误差及可观测方向恢复。旧 adapter 测试的 hardware.json 占位文本已改为合法合同后单独复测通过。",
        '',
        f"闭环评估平均占用 {candidate['resource']['average_cpu_cores']:.2f} 个 CPU 核、峰值进程内存 {candidate['resource']['peak_rss_mib']:.0f}MiB；28 组回放平均占用 {uncertainty['resource']['average_cpu_cores']:.2f} 个 CPU 核。没有后台长训练。",
        '',
        '- [逐场景结果](combined_closed_loop_12/RESULTS.json) · [定位回放](uncertainty_28/RESULTS.json) · [机器可读摘要](SUMMARY.json)',
        '- [验收决定](PROMOTION_DECISION.json) · [回归记录](REGRESSION_TESTS.txt) · [源码快照与哈希](combined_closed_loop_12/PROVENANCE.json)',
        '- 新组合候选：`models/localization_mapping_20261015/contour_mask_candidate/portable`；初步预测候选：`predictive_mask_candidate/portable`。两者均为实验包，不应直接替换实车发布包。',
        '',
        '相对最早实车部署的 2026-10-14 质量恢复 2k 模型，本轮没有做新的完整配对测试，不能把 11/12 当作对它的直接提升。归档参考版本在这 12 项曾为 11/12，本轮未在成功率上超过它。'])
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    for folder in ('predictive_mask_candidate','contour_mask_candidate'):
        card_path=sources/folder/'portable/MODEL_CARD.json'
        card=json.loads(card_path.read_text());card.update(deployment_approved=False,evaluation_report=str(out.relative_to(workspace)/'REPORT.md'))
        if folder=='contour_mask_candidate':card['fixed_development_result']=summary['candidate']
        else:card['fixed_two_failure_tasks']=dict(tasks=2,success=1,collisions=1)
        card_path.write_text(json.dumps(card,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
