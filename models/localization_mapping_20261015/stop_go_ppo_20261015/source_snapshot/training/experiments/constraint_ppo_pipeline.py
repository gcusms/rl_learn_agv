"""Bounded PPO pilot followed by paired regression; never replaces a release."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback


def stamp():return datetime.now().astimezone().isoformat()


def aggregate(rows):
    return dict(tasks=len(rows),success=sum(r['safe_success'] for r in rows),
        collisions=sum(r['collisions'] for r in rows),timeouts=sum(r['timeout'] for r in rows),
        false_arrivals=sum(r['false_arrival'] for r in rows))


def main():
    p=argparse.ArgumentParser()
    for name in ('bundle','checkpoint','cases','model-output','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--seed',type=int,default=119615)
    p.add_argument('--cpu-cores',type=float,default=.4)
    p.add_argument('--training-cases',type=Path)
    p.add_argument('--recorded-episodes-per-cycle',type=int,choices=[1,3],default=1)
    a=p.parse_args()
    for key in ('bundle','checkpoint','cases','model_output','output'):setattr(a,key,getattr(a,key).resolve())
    if a.model_output.exists() or (a.output/'status.json').exists():p.error('Fresh model and pipeline status required')
    a.output.mkdir(parents=True,exist_ok=True)
    status=dict(status='preparing',started=stamp(),additional_ppo_target=a.steps,
        model_output=str(a.model_output),deployment_approved=False,auto_replace=False)
    def save(**values):
        status.update(values,updated=stamp())
        temporary=a.output/'status.tmp';temporary.write_text(json.dumps(status,indent=2)+'\n')
        temporary.replace(a.output/'status.json')
    save()
    try:
        # Keep the supervisor small; the child performs the full runtime
        # contract check. Importing that runtime here would retain Torch RAM.
        manifest=json.loads((a.bundle/'manifest.json').read_text())
        for name,digest in manifest['files'].items():
            if Path(name).name!=name or hashlib.sha256((a.bundle/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Parent bundle integrity failure')
        cases=json.loads(a.cases.read_text())
        if len(cases)!=12 or len({c['id'] for c in cases})!=12:raise ValueError('Expected 12 fixed regression tasks')
        (a.output/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
        training_cases=a.cases if a.training_cases is None else a.training_cases.resolve()
        training_content=json.loads(training_cases.read_text())
        if not training_content or not any(c['id'].startswith('train_failure_') for c in training_content):
            raise ValueError('Explicitly marked training cases required')
        (a.output/'training_cases.json').write_text(json.dumps(training_content,indent=2)+'\n')
        identity=dict(bundle=str(a.bundle),bundle_manifest_sha256=hashlib.sha256((a.bundle/'manifest.json').read_bytes()).hexdigest(),
            checkpoint=str(a.checkpoint),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
            cases_sha256=hashlib.sha256((a.output/'cases.json').read_bytes()).hexdigest(),
            training_cases_sha256=hashlib.sha256((a.output/'training_cases.json').read_bytes()).hexdigest(),
            recorded_episodes_per_cycle=a.recorded_episodes_per_cycle,
            resource_policy=dict(soft_cpu_cores=a.cpu_cores,systemd_cpu_quota_percent=50,
                systemd_memory_high_gib=3,systemd_memory_max_gib=4,threads=1),
            evaluation_scope='paired reused development/regression cases, not unseen-map generalization',
            trainable='early sensor-gated residual and critic; normal base, localizer, router and directional approach frozen')
        (a.output/'PLAN.json').write_text(json.dumps(identity,indent=2)+'\n')
        command=[sys.executable,'-m','experiments.train_localization_recovery','--bundle',str(a.bundle),
            '--checkpoint',str(a.checkpoint),'--cases',str(a.output/'training_cases.json'),'--output',str(a.model_output),
            '--steps',str(a.steps),'--seed',str(a.seed),'--cpu-cores',str(a.cpu_cores),
            '--conditional-loss','--anticipatory-support',
            '--recorded-episodes-per-cycle',str(a.recorded_episodes_per_cycle)]
        save(status='training',command=command)
        with (a.output/'training.log').open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
        training=json.loads((a.model_output/'status.json').read_text())
        if training['status']!='complete' or training['additional_ppo_steps']!=a.steps:
            raise RuntimeError('Training did not complete the declared budget')
        if training['counts']['changed_nonzero']:raise RuntimeError('Training/execution mismatch')
        save(status='paired_evaluation',training=training)
        evaluation=a.output/'paired_12'
        command=[sys.executable,'-m','experiments.evaluate_docking_navigation',
            '--candidate',str(a.model_output/'portable'),'--baseline',str(a.bundle),
            '--output',str(evaluation),'--cases',str(a.output/'cases.json'),
            '--cpu-cores',str(a.cpu_cores),'--candidate-name','anticipatory_ppo','--baseline-name','constraints_before_ppo']
        with (a.output/'evaluation.log').open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
        results=json.loads((evaluation/'RESULTS.json').read_text())
        if not results['completed'] or len(results['rows'])!=24:raise RuntimeError('Incomplete paired evaluation')
        before=[r for r in results['rows'] if r['variant']=='constraints_before_ppo']
        after=[r for r in results['rows'] if r['variant']=='anticipatory_ppo']
        old,new=aggregate(before),aggregate(after)
        parking=lambda rows:aggregate([r for r in rows if r['case_id'].startswith('localization_nav_')])
        old_parking,new_parking=parking(before),parking(after)
        improved=new['success']>old['success'] and new['collisions']<=old['collisions']
        no_regression=(new['success']>=old['success'] and new['collisions']<=old['collisions']
            and new['false_arrivals']<=old['false_arrivals'] and new_parking['success']>=old_parking['success'])
        from experiments.paired_navigation_gate import compare
        paired_gate=compare(before,after)
        summary=dict(completed=True,deployment_approved=False,before=old,after=new,
            before_parking=old_parking,after_parking=new_parking,additional_ppo_steps=a.steps,
            demonstrated_development_success_improvement=improved,
            no_regression_on_these_tasks=paired_gate['all_regression_checks_passed'],
            count_nonregression=no_regression,paired_gate=paired_gate,
            training_counts=training['counts'],further_training_gate_passed=paired_gate['larger_training_gate_passed'],
            localization_accuracy_training=False,unknown_map_slam_training=False,
            limitation='Small previously used regression set; no hardware acceptance or promotion')
        (a.output/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
        counts=training['counts']
        lines=['# 提前动态决策 PPO 试训','',
            f'新增 {a.steps:,} 步 PPO，已完成训练和 12 个相同场景的训练前后配对评估。原部署模型未替换。',
            '', '| 项目 | 训练前约束候选 | PPO 候选 |','|---|---:|---:|',
            f"| 安全到达 | {old['success']}/12 | {new['success']}/12 |",
            f"| 碰撞 | {old['collisions']} | {new['collisions']} |",
            f"| 超时 | {old['timeouts']} | {new['timeouts']} |",
            f"| 停靠回归 | {old_parking['success']}/6 | {new_parking['success']}/6 |",
            '', '本轮训练提前开放雷达已支持的运动目标与动作集合收紧状态，仍由 PPO 选择合法动作；未放宽车体、速度或安全约束。正常开阔状态的基础策略、定位、路由和近目标方向停靠保持冻结。',
            '', f"训练中可更新策略的样本 {counts['trainable_recovery']}/{a.steps}，其中新增提前决策样本 {counts['early_trainable_frames']}；仅剩停止可选 {counts['only_stop']} 步。非零策略动作与执行不一致为 {counts['changed_nonzero']}。",
            '', '已修复重放固定训练案例时回退课程随机数的采样问题；正常任务会继续变化，案例仍按来源与固定评估任务区分。',
            '', '本轮是否证明到达率改善：'+('是，仅限此开发集合。' if improved else '否。'),
            '是否保留原有回归表现：'+('此集合通过安全与效率检查。' if paired_gate['all_regression_checks_passed'] else '存在安全或效率退化，应保留原模型。'),
            '', '定位器没有参与此次 PPO 更新，因此不能据此声称实车定位更准确；未知环境 SLAM、实车绝对误差和 Orin 延迟仍需分别验证。测试集较小且已用于开发，本轮无论结果如何均不自动发布模型。',
            '', '[逐场景评估](paired_12/RESULTS.json) · [实验身份与资源计划](PLAN.json) · [训练日志](training.log) · [摘要](SUMMARY.json)',
            '', f'训练模型：`{a.model_output}`。完整优化器、候选导出、随机数与课程状态保存在该目录；随机数快照不等同于精确恢复回合中途的物理状态。']
        (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
        save(status='complete',finished=stamp(),summary=summary)
    except BaseException as error:
        save(status='failed',finished=stamp(),error=repr(error))
        (a.output/'ERROR.txt').write_text(traceback.format_exc())
        raise


if __name__=='__main__':main()
