"""Publish the failed ablations as well as any candidate regression results."""
import hashlib
import json
from pathlib import Path
from experiments.paired_navigation_gate import compare


def main():
    root=Path('reports/localization_mapping_20261015/blind_repair_20261015')
    baseline_path=root.parent/'anticipatory_ppo_20261015_132151/paired_12/RESULTS.json'
    baseline=[r for r in json.loads(baseline_path.read_text())['rows'] if r['variant']=='constraints_before_ppo']
    variants={}
    for name in ('registered_failure_2','residual_failure_2','soft_failure_2','soft_regression_12'):
        path=root/name/'RESULTS.json'
        if not path.exists():continue
        result=json.loads(path.read_text());rows=result['rows']
        keys={r['case_id'] for r in rows};before=[r for r in baseline if r['case_id'] in keys]
        variants[name]=dict(completed=result['completed'],rows=rows,resource=result.get('resource'),
            results_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            paired_gate=compare(before,rows) if len(rows)==len(before) else None)
    report=dict(deployment_approved=False,original_bundle_replaced=False,new_ppo_training=False,
        scope='Reused development failures and regression tasks, not independent hardware acceptance',variants=variants)
    (root/'SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# 动态等待与绕行专项修复','',
        '原模型保留。此次先诊断、做独立机制对照，尚未追加 PPO，也不自动发布候选。',
        '', '## 诊断与改动', '',
        '- 遮挡造成轮廓中心变化，会污染速度估计；新增连续轮廓的点到线平移配准实验，保留低几何约束时的未知状态。',
        '- 历史轨迹的存在不能表示本帧有新测量；新实验明确使用原测量时间，不延长可靠速度的保持时间。',
        '- 第二个配准实验同步用配准信息与跨时间估计分歧计算误差，避免把轮廓中心变化再次计入；这不是经过实车标定的置信区间。',
        '- 将未来多时刻占据并成硬墙会诱发全局折返；另做单项路线实验：真实/持久障碍仍为硬限制，未来单独使用软代价。局部预测掩码、原始扫描、车体和保护间距保持不变。',
        '- 新增配对模型选择检查：逐例安全不退化，再检查共同成功任务的路程、耗时；旧 2k PPO 未通过效率检查。',
        '', '## 已完成闭环结果', '',
        '| 实验 | 任务 | 安全到达 | 碰撞 | 超时 |','|---|---:|---:|---:|---:|',
        '| 原约束候选（历史已完成配对记录） | 2 | 1 | 1 | 0 |']
    for name,result in variants.items():
        rows=result['rows']
        lines.append(f"| {name}{'' if result['completed'] else '（进行中）'} | {len(rows)} | {sum(r['safe_success'] for r in rows)} | {sum(r['collisions'] for r in rows)} | {sum(r['timeout'] for r in rows)} |")
    lines+=['', '配准两项实验均需按闭环结果判定，不能用合成跟踪检查通过宣称导航已改善。失败产物与各次源代码快照保留；每个候选均有独立合同与 manifest。',
        '', '## 验证与限制', '',
        '本轮不修改定位器权重、方向停靠 actor、底盘尺寸、12 个动作或速度上限。当前约束不包含倒车，边界条件保持可追溯。',
        '原 V2 对 58 帧传感器回放的动态接触时间与此前记录完全一致，旧候选没有被静默改变。',
        '实验服务单线程、CPU 硬上限 50% 单核，软平均 0.4 核，内存硬上限 4 GiB。',
        '这些案例已经用于开发，不能称为未见场景泛化。此轮不构成未知环境 SLAM、实车定位精度或 Orin 实时性的验收。',
        '', '[详细数据](SUMMARY.json) · [旧 PPO 效率检查](previous_ppo_efficiency_gate.json) · [传感器逐帧诊断](motion_shadow_v2_v3.json)']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(root/'REPORT.md')


if __name__=='__main__':main()
