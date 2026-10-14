"""Choose a development checkpoint from complete, provenance-tagged results."""
import copy,csv,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import MaskablePPO

ROOT=Path(__file__).resolve().parents[2];RUNS=ROOT/'training/runs';REPORT=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(1)
    comparison=RUNS/'docking_matched_comparison_20261014'
    original=json.loads((comparison/'RESULTS.json').read_text());assert original['completed']
    adapted=RUNS/'docking_v4_adapted_eval_20261015'
    adapted_report=json.loads((adapted/'RESULTS.json').read_text());assert adapted_report['completed']
    old5=RUNS/'docking_staged_5000_validated_20261014';new5=RUNS/'docking_5000_new_tasks_20261015'
    reports=[json.loads((p/'RESULTS.json').read_text()) for p in [old5,new5]];assert all(r['completed'] for r in reports)
    cases=json.loads((comparison/'cases.json').read_text());assert json.loads((adapted/'cases.json').read_text())==cases
    assert json.loads((old5/'cases.json').read_text())==[c for c in cases if c['pilot_category'] in ['alignment','short_approach']]
    assert json.loads((new5/'cases.json').read_text())==cases[12:]
    output=RUNS/'docking_all_versions_20261015';output.mkdir(exist_ok=False)
    rows=[]
    def append(source,original_row,variant,cached=False):
        r=copy.deepcopy(original_row);p=source/f"{r['case_id']}_{r['variant']}.npz"
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r['trace_sha256']
        r['variant']=variant;r['provenance']=dict(source=str(source.relative_to(ROOT)),historical_cached=cached)
        shutil.copy2(p,output/f"{r['case_id']}_{variant}.npz");rows.append(r)
    for r in original['rows']:append(comparison,r,r['variant'],r['provenance']['historical_cached'])
    for source,report in zip([old5,new5],reports):
        for r in report['rows']:append(source,r,'trained_5k')
    policy5=MaskablePPO.load(ROOT/'models/docking_selected_5000_20261015/policy.zip',device='cpu')
    cached=0
    for r in original['rows']:
        if r['variant']!='trained' or r['category']!='position_regression':continue
        with np.load(comparison/f"{r['case_id']}_trained.npz") as a:
            assert np.all(a['observations'][:,302]==0)
            choices=policy5.predict(a['observations'],action_masks=a['masks'],deterministic=True)[0]
            np.testing.assert_array_equal(choices,a['choices']);cached+=len(choices)
        append(comparison,r,'trained_5k',True)
    for r in adapted_report['rows']:append(adapted,r,'trained_12500_v4')
    assert len(rows)==140 and len({(r['case_id'],r['variant']) for r in rows})==140
    variants=sorted({r['variant'] for r in rows});groups={}
    for category in sorted({r['category'] for r in rows}):
        groups[category]={}
        for v in variants:
            items=[r for r in rows if r['category']==category and r['variant']==v]
            groups[category][v]=dict(tasks=len(items),success=sum(r['safe_success'] for r in items),collisions=sum(r['collisions'] for r in items),
                timeouts=sum(r['timeout'] for r in items),false_arrivals=sum(r['false_arrival'] for r in items))
    total={v:dict(tasks=28,success=sum(r['safe_success'] for r in rows if r['variant']==v),
        directional_success=sum(r['safe_success'] for r in rows if r['variant']==v and r['category']!='position_regression'),
        collisions=sum(r['collisions'] for r in rows if r['variant']==v),false_arrivals=sum(r['false_arrival'] for r in rows if r['variant']==v)) for v in variants}
    models={'trained_5k':('models/docking_selected_5000_20261015',5000),'trained':('models/docking_candidate_20261014',10000),
        'trained_v4_localization':('models/docking_candidate_20261014',10000),'trained_12500_v4':('models/docking_v4_adapted_20261015',12500)}
    selected=max(models,key=lambda v:(-total[v]['collisions'],-total[v]['false_arrivals'],total[v]['success'],-models[v][1]))
    info=dict(deployment_approved=False,selected_variant=selected,selected_model=models[selected][0],
        selected_bundle='portable_v4_localization' if selected=='trained_v4_localization' else 'portable',selected_terminal_steps=models[selected][1],
        maximum_trained_terminal_steps=12500,selection_scope='same fixed development set; not independent final test',totals=total,
        all_comparison_rows=140,fresh_runs=124,cached_runs=16,position_only_5k_cached_action_parity_frames=cached,
        rejected_variants=[v for v in models if v!=selected],groups=groups)
    (output/'cases.json').write_text(json.dumps(cases,indent=2))
    (output/'RESULTS.json').write_text(json.dumps(dict(completed=True,rows=rows,groups=groups,scope=info['selection_scope'],deployment_approved=False),indent=2))
    (REPORT/'MODEL_SELECTION.json').write_text(json.dumps(info,indent=2))
    chosen=RUNS/'docking_selected_evaluation_20261015';chosen.mkdir()
    (chosen/'cases.json').write_text(json.dumps(cases,indent=2));chosen_rows=[]
    for row in rows:
        if row['variant']!=selected:continue
        r=copy.deepcopy(row);r['variant']='trained';chosen_rows.append(r)
        shutil.copy2(output/f"{row['case_id']}_{selected}.npz",chosen/f"{row['case_id']}_trained.npz")
    (chosen/'RESULTS.json').write_text(json.dumps(dict(completed=True,rows=chosen_rows,scope=info['selection_scope'],deployment_approved=False),indent=2))
    shutil.copytree(output,REPORT/'all_versions_evaluation')
    shutil.copy2(REPORT/'REPORT.md',REPORT/'REPORT_10000.md')
    shutil.copy2(REPORT/'SUMMARY.json',REPORT/'SUMMARY_10000.json')
    (REPORT/'SUMMARY.json').write_text(json.dumps(info,indent=2))
    title={'previous_3k':'上轮 3k','trained_5k':'停靠 5k','trained':'停靠 10k','trained_v4_localization':'10k＋定位 v4','trained_12500_v4':'组合适配 12.5k'}
    table='| 版本 | 方向停靠 | 旧位置导航 | 碰撞 | 假到达 |\n|---|---:|---:|---:|---:|\n'
    for v in ['previous_3k','trained_5k','trained','trained_v4_localization','trained_12500_v4']:
        table+=f"| {title[v]} | {total[v]['directional_success']}/24 | {groups['position_regression'][v]['success']}/4 | {total[v]['collisions']} | {total[v]['false_arrivals']} |\n"
    failed=[r for r in chosen_rows if not r['safe_success']]
    prefix=f'''# 本轮最终选择：{title[selected]}

**本轮最高训练到 12,500 步终点 PPO，最终选择 {models[selected][1]:,} 步检查点。** 按固定开发任务的碰撞、假到达和物理到达选择，没有按步数选择。原部署模型保留。候选均未通过部署验收。

{table}
5k 的 8 项旧方向测试为 8/8，新增 16 项为 15/16，因此总计 23/24。10k 为 20/24，组合 v4 的 10k 也为 20/24；更大的步数没有自动带来更好的闭环结果。又在真实底盘参数保持原样、新定位 v4 的输入下追加 2.5k，结果见表格，不因训练完成而默认发布。

最终候选失败任务：{', '.join(r['case_id'] for r in failed)}。原横穿碰撞仍未修复。10k 的额外超时均有运动动作可选，并已进入可训练停靠分支；其中若干实际方向误差停在 0.157–0.188 rad，未满足 0.15 rad 标准。这说明需要针对小角度停车边界的课程/示范保持，而不是放宽验收或强迫车继续前进。[等待诊断](waiting_diagnosis.json)。

本轮方向停靠明显改善；定位 v4 在未入图物体等专项物理对照中有改善，但初始偏差及组合闭环仍须独立检查。**不能据此声称整体实车导航已可靠、路线最短或超过 TEB。** 新定位的改善不是本轮 PPO 训练了定位网络。

产物：`{models[selected][0]}/policy.zip` 与配套 `portable`（若选择 10k 原候选的定位 v4 版本则对应 `portable_v4_localization`）。详情见 [MODEL_SELECTION.json](MODEL_SELECTION.json)，逐例数据在 [all_versions_evaluation/RESULTS.json](all_versions_evaluation/RESULTS.json)，五版本回放在 [index.html](index.html)。候选模型的功能检查 23 项通过。

五版本共 140 条配对结果：124 条在本轮实际运行，16 条明确复用。复用包括上轮 3k 的 12 条基线，以及 5k 的 4 条无方向位置任务；后者先验证全部对应观测下动作与冻结导航分支完全一致，未将复用算作重新成功。所有地图属于既有训练地图，测试用于开发与模型选择，未打开封存验收集。

以下保留 10k 主实验的完整方法、定位数值、资源和公开依据；它是试验记录，不覆盖上面的最终选择。

'''
    (REPORT/'REPORT.md').write_text(prefix+(REPORT/'REPORT_10000.md').read_text())
    for m in ['models/docking_selected_5000_20261015','models/docking_v4_adapted_20261015']:
        folder=ROOT/m;shutil.copy2(REPORT/'MODEL_SELECTION.json',folder/'MODEL_SELECTION.json')
        (folder/'README.md').write_text(f"# 方向停靠候选\n\n最高训练 12.5k，开发选择结果见 [本轮报告](../../reports/docking_training_20261014/REPORT.md)。本目录是否被选择：{m==models[selected][0]}。未通过实车部署验收，原发布模型未替换。使用配套 303 维、12 动作 `experiments.docking_navigation.DockingRuntime`。\n")
        with zipfile.ZipFile(folder/'final_source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
            for directory in ['training/experiments','training/neural_navigation','deployment/ros2_recording_adapter']:
                for p in (ROOT/directory).glob('*.py'):z.write(p,str(p.relative_to(ROOT)))
        bundle=folder/'portable'
        config=dict(bundle=str(bundle),model_source=str(ROOT),sensor_core='learned-terminal-goal-pose',
            required_runtime='experiments.docking_navigation.DockingRuntime',observation_dimensions=303,action_choices=12,
            route_mode='shortcut',manifest_sha256=hashlib.sha256((bundle/'manifest.json').read_bytes()).hexdigest(),deployment_approved=False)
        (folder/'model.json').write_text(json.dumps(config,indent=2))
        manifest=dict(status='development_candidate',deployment_approved=False,
            files={str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob('*') if p.is_file() and p.name!='CANDIDATE_MANIFEST.json'})
        (folder/'CANDIDATE_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(info,indent=2))

if __name__=='__main__':main()
