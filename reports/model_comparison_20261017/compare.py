"""Summarize saved evaluations without running training or navigation workers."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports/stop_feedback_20261015"
sources = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    path = ROOT / path
    sources[str(path.relative_to(ROOT))] = sha(path)
    return json.loads(path.read_text())


def evaluation(name):
    path = REPORTS / name / "RESULTS.json"
    data = read(path)
    rows = []
    for group, contents in data["groups"].items():
        for row in contents["rows"]:
            rows.append(dict(row, comparison_group=group,
                             comparison_source=str(path.relative_to(ROOT))))
    return data, rows


def counts(rows):
    return dict(checks=len(rows), safe_arrivals=sum(bool(r["safe_success"]) for r in rows),
                collision_cases=sum(r.get("collisions", 0) > 0 for r in rows),
                timeouts=sum(bool(r.get("timeout")) for r in rows))


specs = [
    ("bc4500", "625维等待修复版（BC 4500）", "normal_suite_wait_pair_v2_s231016/wait_continuation", "normal_suite_wait_validation_v2_20261016"),
    ("ppo1993", "PPO 1993步版", "guard_aligned_mixed_ppo_resume_s231017", "guard_aligned_mixed_ppo_resume_validation_20261017"),
    ("aux119", "行为保留119版", "v2_behavior_retention_s231017", "v2_behavior_retention_validation_20261017"),
    ("margin2500", "间隔约束2500版", "full_navigation_margin_resume_s231017", "full_navigation_margin_resume_validation_20261017"),
    ("latest2501", "最新联合约束版", "full_navigation_joint_linear_s231017", "full_navigation_joint_linear_validation_20261017"),
]
models = {}
for key, label, model, report in specs:
    raw, rows = evaluation(report)
    models[key] = dict(label=label, model_path="models/stop_feedback_20261015/" + model,
                       observations=625, evaluation_rows=rows, source_evaluation_counts=counts(rows))

parent_path = REPORTS / "native625_local_speedup_retention_20261017/RESULTS.json"
parent_data = read(parent_path)
parent_rows = [dict(r, comparison_group="observed6", comparison_source=str(parent_path.relative_to(ROOT)))
               for r in parent_data["frozen_rows"] if r["variant"] == "parent"]
assert len(parent_rows) == 6
assert all(r["bundle"] == models["bc4500"]["model_path"] + "/portable" for r in parent_rows)
models["bc4500"]["evaluation_rows"].extend(parent_rows)

supp_path = REPORTS / "deployment_acceptance_supplement_20261017/FINAL_REVIEW.json"
supp = read(supp_path)
assert supp["completed"] and not supp["deployment_approved"]
assert supp["model"] == models["latest2501"]["model_path"]
models["latest2501"]["evaluation_rows"].extend(
    dict(r, comparison_group="hard3", comparison_source=str(supp_path.relative_to(ROOT))) for r in supp["rows"])

fixed_ids = {r["case_id"] for r in models["bc4500"]["evaluation_rows"]
             if r["comparison_group"] not in ("observed6", "hard3")}
normal_ids = {r["case_id"] for r in models["bc4500"]["evaluation_rows"]
              if r["comparison_group"] in ("normal_other16", "normal_repaired2")}
assert len(fixed_ids) == 28 and len(normal_ids) == 18
for model in models.values():
    rows = model["evaluation_rows"]
    assert len({r["case_id"] for r in rows}) == len(rows)
    model["saved_check_counts"] = counts(rows)
    model["fixed28"] = counts([r for r in rows if r["case_id"] in fixed_ids])
    model["known6"] = counts([r for r in rows if r["comparison_group"] == "observed6"])
    model["hard3"] = counts([r for r in rows if r["comparison_group"] == "hard3"])
    assert model["fixed28"]["safe_arrivals"] == 28
    normal = [r for r in rows if r["case_id"] in normal_ids]
    model["normal18"] = dict(seconds=sum(r["seconds"] for r in normal),
                              travel_m=sum(r["travel_m"] for r in normal))

old_path = "models/quality_recovery_20261014"
old_manifest = read(old_path + "/RELEASE_MANIFEST.json")
old_verified = read(old_path + "/reports/VERIFIED_FULL48.json")
old = dict(label="原实车使用的质量恢复2k版", model_path=old_path, observations=300,
           historical_evaluation={k: old_verified[k] for k in ("tasks", "safe_success", "collisions", "timeouts")},
           historical_evaluator_limitation=old_verified["limitation"],
           selected_additional_ppo_steps=old_manifest["selected_additional_ppo_steps"],
           directly_comparable_to_current_37=False, deployment_approved=old_manifest["deployment_approved"])

for model in [old, *models.values()]:
    path = ROOT / model["model_path"]
    portable = path / "portable"
    names = ["actor.pt", "localizer.pt", "router.pt", "manifest.json"]
    model["artifact_sha256"] = {name: sha(portable / name) for name in names}
    sources.update({str((portable / name).relative_to(ROOT)): sha(portable / name) for name in names})
    model["portable_bytes"] = sum(p.stat().st_size for p in portable.rglob("*") if p.is_file())
    model["actor_bytes"] = (portable / "actor.pt").stat().st_size
    if (path / "checkpoint.pt").exists():
        model["checkpoint_sha256"] = sha(path / "checkpoint.pt")
        sources[str((path / "checkpoint.pt").relative_to(ROOT))] = model["checkpoint_sha256"]

plan_paths = [REPORTS / "native625_local_speedup_retention_20261017/PLAN.json",
              REPORTS / "full_navigation_joint_linear_validation_20261017/PLAN.json"]
plans = [read(p) for p in plan_paths]
case_key = "reports/stop_feedback_20261015/native625_frozen_holdout_20261016/CASES.json"
actor_key = models["bc4500"]["model_path"] + "/portable/actor.pt"
assert plans[0]["inputs"][case_key] == plans[1]["inputs"][case_key] == sha(ROOT / case_key)
assert plans[0]["inputs"][actor_key] == plans[1]["inputs"][actor_key] == sha(ROOT / actor_key)
read(REPORTS / "deployment_acceptance_supplement_20261017/VERSIONED_TRACE_AUDIT.json")
localization_contract = read(models["latest2501"]["model_path"] + "/portable/localization_contract.json")

before = {r["case_id"]: r for r in models["bc4500"]["evaluation_rows"]}
after = {r["case_id"]: r for r in models["latest2501"]["evaluation_rows"]}
assert before.keys() == after.keys() and len(before) == 37
assert counts(list(before.values())) == dict(checks=37, safe_arrivals=33, collision_cases=1, timeouts=3)
assert counts(list(after.values())) == dict(checks=37, safe_arrivals=35, collision_cases=0, timeouts=2)
assert models["latest2501"]["checkpoint_sha256"] == supp["checkpoint_sha256"]
assert all(before[c][k] == after[c][k] for c in fixed_ids
           for k in ("seconds", "travel_m", "localization_xy_p95_m", "localization_yaw_p95_rad"))

paired = []
for case_id in sorted(before):
    b, a = before[case_id], after[case_id]
    paired.append(dict(case_id=case_id, before_safe=b["safe_success"], after_safe=a["safe_success"],
                       before_seconds=b["seconds"], after_seconds=a["seconds"],
                       before_travel_m=b["travel_m"], after_travel_m=a["travel_m"],
                       before_source=b["comparison_source"], after_source=a["comparison_source"]))

result = dict(at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              scope="Saved-record comparison of six representative versions. Mixed correlated development/recovery checks; no new training or physics.",
              original_real_car_version=old, recent_versions=models,
              paired_development37=paired,
              became_safe=[r["case_id"] for r in paired if not r["before_safe"] and r["after_safe"]],
              lost_safe=[r["case_id"] for r in paired if r["before_safe"] and not r["after_safe"]],
              all_six_localizer_weights_identical=len({m["artifact_sha256"]["localizer.pt"] for m in [old, *models.values()]}) == 1,
              all_six_router_weights_identical=len({m["artifact_sha256"]["router.pt"] for m in [old, *models.values()]}) == 1,
              current_localization_contract=localization_contract,
              deployment_approved=False, new_training_steps=0, new_physical_episodes=0,
              important_limits=["300-dimension original real-car model did not run this matched 37-check protocol.",
                                "625-v1 and 625-v2 change 42 risk dimensions despite equal total dimension.",
                                "All known6 tasks have influenced candidate development/training.",
                                "Late recovery prefixes are initialization, not full autonomous missions.",
                                "Fresh eight pairs and Orin Nano/Raspberry Pi validation remain unexecuted."])
(OUT / "COMPARISON.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
(OUT / "SOURCES.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def cell(model, case):
    row = next((r for r in model["evaluation_rows"] if r["case_id"] == case), None)
    if row is None:
        return "未测"
    state = "到达" if row["safe_success"] else "碰撞" if row["collisions"] else "超时" if row["timeout"] else "未通过"
    return f"{row['seconds']:.1f}s {state}"


matrix = "\n".join("| " + " | ".join([m["label"], *[cell(m, "pre_ppo_holdout_" + c) for c in
                     ("dynamic_00", "dynamic_01", "dynamic_02", "sensor_01")]]) + " |" for m in models.values())
coverage = "\n".join(f"| {m['label']} | {m['fixed28']['safe_arrivals']}/28 | {m['known6']['safe_arrivals']}/{m['known6']['checks']} | {m['hard3']['safe_arrivals']}/{m['hard3']['checks']} | {m['normal18']['seconds']:.1f} / {m['normal18']['travel_m']:.3f} |" for m in models.values())
paths = "\n".join(f"- {m['label']}：[模型目录](../../{m['model_path']}/)。" for m in [old, *models.values()])
sizes = "\n".join(f"| {m['label']} | {m['actor_bytes']/1048576:.3f} MiB | {m['portable_bytes']/1048576:.3f} MiB |" for m in [old, *models.values()])

text = f"""# 现有导航模型对比（2026-10-17）

在已完成的共同开发检查上，**最新联合约束版最值得保留为下一轮实车对照候选**：它同时保住了传感器退化任务的避碰修复和动态任务的通行能力。现有记录仍不足以选出全面通过部署验收的版本，也不能认定其定位更准或路径全局最短。

本报告读取已有原始结果并核对模型 SHA256，没有追加训练或仿真。训练与优化继续暂停，原实车模型未替换。工作目录下还有多个研究变体；以下选择有关键闭环评估记录的六个代表版本，并未给未经同条件验证的全部目录排总榜。

## 1. 原实车版与新候选的比较边界

用户确认的原实车模型为 `quality_recovery_20261014`，300维观测、12个动作、追加2000步PPO。历史48项记录为 **46次安全到达、1次碰撞、1次超时**。这套48项与当前检查集不同；不能用46/48直接判断它比最新候选更好或更差。原发布清单本身也标记 `deployment_approved=false`。

原48项评估的到达/碰撞标签由当时评估器给出，并非独立重建的物理结论；发布时没有重新跑完整48项。[发布清单](../../models/quality_recovery_20261014/RELEASE_MANIFEST.json)和[历史核验](../../models/quality_recovery_20261014/reports/VERIFIED_FULL48.json)保留这些限制。

近期五个版本均为625维观测、相同12个速度动作。最高前进速度0.2m/s、角速度绝对值0.5rad/s，车体和4cm安全余量沿用原验收设置。本报告中的2k、1993为相应阶段PPO交互步数；119、2500、2501为辅助训练/约束更新次数，不能当作PPO步数相加。

## 2. 同任务逐项对比

表内时间为仿真任务经过的时间，非电脑计算耗时。“未测”既不是成功也不是失败。四个任务的完整标识分别是 `pre_ppo_holdout_dynamic_00/01/02` 与 `pre_ppo_holdout_sensor_01`，现已属于开发任务。

| 模型 | 动态00 | 动态01 | 动态02 | 传感器退化01 |
|---|---|---|---|---|
{matrix}

- 相比625维等待修复版，最新候选将动态00从140s超时改善为61.6s到达，将传感器退化01从13s碰撞改善为37s安全到达。
- PPO1993步候选通过了动态00，但动态01从34.6s变成53.4s，慢54.3%，超过原20%的单项效率门限。该对比也涉及v2风险特征适配，不能把差异全归因于步数，或只按训练步数选择模型。
- 行为保留119版恢复了动态01效率，但传感器退化01仍有碰撞。间隔约束2500版修复该碰撞后，动态02又因主动等待而140s超时。
- 最新版让动态02回到49.2s到达，并保留37s的传感器退化01安全到达。它对直接上一候选的改善是修复一项回归，不应称为所有路线都明显提速。

## 3. 覆盖范围与共同37项记录

固定28项由普通18项、传感器5项、源恢复3项及带方向停车2项组成。已知整程任务有6项；困难恢复有3项。下面的分母是实际已测数量，0/0表示没有运行，部分版本因先前门限失败提前停止后续测试。

| 模型 | 固定28项 | 已知整程6项（已测） | 困难恢复3项（已测） | 普通18项总用时s / 总路程m |
|---|---|---|---|---|
{coverage}

从625维等待修复版的两份原始评估与最新版的正式评估、补验中，可以重建**相同37个任务标识的配对记录**：

| 同一37项开发/恢复检查 | 安全到达 | 碰撞 | 超时 |
|---|---:|---:|---:|
| 625维等待修复版 | 33 | 1 | 3 |
| 最新联合约束版 | 35 | 0 | 2 |

改善的是动态00和传感器退化01，没有丢失旧版在这37项中的安全到达。以上包含相关前缀、训练参考和局部恢复，**不构成37次独立任务的泛化成功率**，也不能与原实车版48项相混比较。

任务文件与基础actor的冻结哈希在两套评估计划中一致。v1和v2虽然总维度同为625，42维风险语义发生改变；其余输入与版本化风险重建由[补验审计](../stop_feedback_20261015/deployment_acceptance_supplement_20261017/VERSIONED_TRACE_AUDIT.json)核对。这里比较的是模型及配套特征合同的组合，不是“完全相同的625维张量只替换一个网络”。

## 4. 路程、定位与停车

普通18项从625维等待修复版到最新候选均为459.4s、67.136m，固定28项的路程、用时和定位P95记录也一致。最近一步没有继续改善这些指标。更早固定18项记录的489.4s、67.474m所对应的约6.13%用时和0.50%路程改善，早在等待修复版已获得，不归因于最新更新。

已知动态03从52.0s减到46.6s，但路程从7.142m增到7.474m（约增加4.64%）。动态02从等待修复版的49.0s、7.588m到当前49.2s、7.597m，接近持平。因此目前证据支持部分场景通行改善，不支持“每条路线更短”或“最短路径保证”。

六个代表版本的 `localizer.pt` SHA256全部相同，`router.pt`也全部相同。配套定位内核和配置历史上可能有变化，不能由权重相同推断所有时期运行行为相同；**最近这几版的主要收益不来自定位网络重新学习，也没有新增同条件实车定位精度证据**。当前导出清单明确定位方法为 `fast_static_tracking_temporal_native_v1`、`localization_is_neural=false`；定位合同仍要求外部给定初始地图位姿，`global_relocalization=false`。

带方向停车的两项固定检查各版均通过，最新候选没有额外证明停车方向精度或陌生场景泛化进一步提升。到达位置误差是车到目标的误差，不应称为定位误差。

三项晚期恢复对比：

| 释放模型控制的时刻 | 等待修复版 | 最新候选 |
|---|---|---|
| 96.0s | 132.4s安全到达 | 132.4s安全到达，保留 |
| 97.6s | 140s超时 | 140s超时，未解决 |
| 106.0s | 140s超时 | 140s超时，未解决 |

这些恢复检查通过已记录的前缀初始化状态，前缀不是全程模型自主导航；统计已按原协议区分。97.6s恢复后，最新候选仍出现187帧“有运动动作可选却主动停下”，连续最长约36.6s，因此不能宣告等待问题全部修复。

## 5. 模型体积与板端资源

以下为实际文件大小，portable目录包含运行所需的组合模型和合同；不等于RAM占用或模型加载耗时。

| 模型 | 导出actor文件 | portable目录总大小 |
|---|---:|---:|
{sizes}

最新候选与直接上一版actor文件字节数相同；与原300维实车版相比，特征与组合模型增加，导出包也更大。目前没有同条件加载时间对照，也没有Orin Nano或树莓派5上的端到端P95/P99实测，不能声称最近的训练降低了板端算力需求。训练/评估进程的RSS也不能直接作为部署节点的内存需求。

## 6. 选择建议与原始证据

保留最新联合约束版作为下一轮受控实车对照候选，保留原实车2k版用于回退。已知安全和恢复任务的组合结果更完整，但当前没有哪一版据这些记录通过完整部署验收。两个晚期恢复超时、未执行的新8组配对、实车定位以及板端运行预算仍是明确缺口。

最新候选继承1993步PPO与行为保留分支119次辅助更新；后续辅助头累计2500次拟合，再进行1次带保留约束的输出层更新。最后一步新增PPO为0，不能称为新一轮大规模PPO训练。

{paths}

原始评估：[等待修复版31项](../stop_feedback_20261015/normal_suite_wait_validation_v2_20261016/RESULTS.json)、[该版6个整程任务的parent行](../stop_feedback_20261015/native625_local_speedup_retention_20261017/RESULTS.json)、[PPO1993](../stop_feedback_20261015/guard_aligned_mixed_ppo_resume_validation_20261017/RESULTS.json)、[行为保留119](../stop_feedback_20261015/v2_behavior_retention_validation_20261017/RESULTS.json)、[间隔约束2500](../stop_feedback_20261015/full_navigation_margin_resume_validation_20261017/RESULTS.json)、[最新正式35项](../stop_feedback_20261015/full_navigation_joint_linear_validation_20261017/RESULTS.json)、[最新补验2项最终审查](../stop_feedback_20261015/deployment_acceptance_supplement_20261017/FINAL_REVIEW.json)。

机器可读结果见[COMPARISON.json](COMPARISON.json)，来源哈希见[SOURCES.json](SOURCES.json)。在本工程根目录执行 `python3 reports/model_comparison_20261017/compare.py` 可从原始记录重建本报告，不会启动训练或物理回放。
"""
(OUT / "README.md").write_text(text)
assert all(sha(ROOT / p) == digest for p, digest in sources.items())
print(json.dumps(dict(output=str(OUT), compared_versions=6, matched_checks=37,
                      before=counts(list(before.values())), after=counts(list(after.values())),
                      became_safe=result["became_safe"], lost_safe=result["lost_safe"],
                      source_hashes_unchanged=len(sources), new_training_steps=0,
                      new_physical_episodes=0), ensure_ascii=False, indent=2))
