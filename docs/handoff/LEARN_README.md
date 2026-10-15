# AGV 导航项目交接

本次交接分支：**`learn`**，日期：**2026-10-15**。包含当前源码、模型与优化器、必要地图与实车回放数据、完整的 V2 失败对照，以及新的停车反馈探索实验。当前进度以 [SNAPSHOT.json](reports/handoff_learn_20261015/SNAPSHOT.json) 为准；冻结时间 **2026-10-15 23:12:27（北京时间）**，新探索版配对评估已归档 **1/28 条结果**。

**先看结论：部署基线保留，最新训练候选尚未通过部署验收。** V2 完整配对评估中，原基线 12/14 安全到达、2 次碰撞；V2 为 11/14、3 次碰撞，因此未采用。新的探索实验已完成 4,000 步、用时 1,677.84 s（约 28 分钟），峰值 RSS 2,459.81 MiB，正在做配对评估；用于解决“明明有合法动作却长期选择停车”，不能把训练结束或功能检查通过当作导航改善。

## 1. 在另一台电脑恢复项目

```bash
git clone --depth 1 --branch learn https://github.com/gcusms/rl_learn_agv.git
cd rl_learn_agv

# 推荐先用 Ubuntu 22.04、Python 3.10 复现；不要复制旧机器的 .venv。
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r training/requirements.txt

export PYTHONPATH=training:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# 输出目录须不存在；检查文件哈希、模型合同、数据、运行时和功能。
# 不连接电机，不训练；不编译时会跳过一项原生数值对照。
python tools/check_handoff.py --output training/runs/handoff_check
```

CPU 离线验证不需要 CUDA、ROS 2 或显示器。建议预留至少 8 GB 安装空间；当前跟踪文件约 0.6 GB，Python 环境和下载缓存会额外占空间。`training/requirements.lock.txt` 是旧 ROS 主机的环境记录，不是跨电脑安装清单。Jetson 应先安装与本机 JetPack 匹配的 PyTorch，不能照搬桌面 CUDA wheel。

停车反馈包使用历史原生定位内核，换电脑需在新目录重建；最新射线定位候选默认使用 Python。完整检查命令：

```bash
sudo apt-get install g++ libeigen3-dev
python tools/check_handoff.py --build-native --output training/runs/handoff_check_native
```

本分支检查入口包含 90 项功能检查，并加载 V1、V2、探索版三个 PPO 检查点及 439 维 TorchScript，保留目标方向。实际执行结果见 [VERIFICATION.json](docs/handoff/VERIFICATION.json)。这些是迁移与实现检查，不是实车精度或安全验收。

## 2. 哪个模型该怎么用

| 用途 | 路径 | 当前判断 |
|---|---|---|
| 原实车部署基线 | [quality_recovery_20261014](models/quality_recovery_20261014) | 300 维；恢复训练追加 2k，历史开发集 46/48 安全到达、1 碰撞、1 超时；本次不替换 |
| 最新定位算法候选 | [ray_consistent_candidate/portable](models/localization_mapping_20261015/ray_consistent_candidate/portable) | 303 维，含目标方向；改善来自定位内核与配置，尚非实车精度验收 |
| 停车反馈 V1 | [ppo_4000_s221015](models/stop_feedback_20261015/ppo_4000_s221015) | 独立追加 4k；额外回归发生碰撞退化，不采用 |
| 停车反馈 V2 | [ppo_v2_4000_s221015](models/stop_feedback_20261015/ppo_v2_4000_s221015) | 独立追加 4k；14 例完整回归发生退化，不采用 |
| 新探索实验 | [ppo_explore_4000_s221015](models/stop_feedback_20261015/ppo_explore_4000_s221015) | V2 反馈＋有界温度探索；4k 训练完成，最终 policy.zip 已保存，配对评估尚未验收 |
| 历史方向停靠模型 | [docking_selected_5000_20261015](models/docking_selected_5000_20261015) | 接受目标位置与朝向；不能单凭停靠成绩替换整机模型 |

原始 2k 基线 `policy_2000.zip` SHA256：`7a3dad60f76060daf6b311b4bc6db921d3925ef0cdcf11683fe99e282173de11`。模型目录内保留配置、来源、优化器、随机数状态与源码快照。最终更新后的权重是 **`policy.zip`**；`policy_step_4000.zip` 是最后一批 rollout 更新前的回调产物，两者不能混用。

**模型与运行时必须配套。** 原基线 300 维、方向候选 303 维；停车反馈 PPO 训练为 436 维，完整方向运行时为 439 维。根据 manifest 的 `required_runtime` 调用 `neural_navigation.goal_pose_runtime.resolve_runtime(...)`，加载前 `check_files(bundle, manifest)`。不能仅替换 `actor.pt`，也不能把 V1 JSON 改成 V2 就复用其权重。完整加载例子见 [check_handoff.py](tools/check_handoff.py)。

车体约 39.39×32.95 cm，基本保护余量 4 cm；12 个动作，前进上限 0.20 m/s、转向上限 0.50 rad/s，不含横移或倒车。约 219.44° 视野的 220 束雷达，后方盲区保留；最新定位按 10 m 有效范围处理。没有通过缩小车体或放宽安全条件提升成绩。

## 3. 当前进度与结果

### 定位与方向停车

| 22 条冻结轨迹、2855 帧，同输入对照 | 时序定位父版本 | 射线一致性候选 |
|---|---:|---:|
| 位置误差 P95 | 18.51 cm | 18.04 cm |
| 航向误差 P95 | 1.526° | 1.368° |
| 轨迹终点位置误差均值 | 3.42 cm | 3.27 cm |
| 本机 Python 定位 CPU P95 | 7.88 ms | 14.59 ms |

定位采用里程计传播、静态支持筛选、弱方向约束和时序融合；最新增加自由射线与地图矛盾检查。以上是重复使用的仿真回归，改善小且耗时增加，**不能说明已经超过 AMCL**。实车 1516 帧录制没有独立绝对位姿真值，更多拒绝更新也可能增加等待。未训练新的定位网络；停车反馈探索实验使用原时序定位父包，不代表已经融合最新射线候选。

目标 `(x, y, yaw)` 已接入方向运行时，同位置改方向可以产生原地转向；仍需闭环停靠误差与恢复后的回归。已知地图需要可靠初值；稀疏环境、边导航边建图、搬车后自动重定位、Orin Nano 端到端耗时仍未验收。详情见[定位报告](reports/localization_mapping_20261015/real_input_repair_20261015/REPORT.md)、[算法说明](docs/localization/README.md)。

### 停车反馈与 PPO

V2 修正了“执行零速、有运动选项，但原因标记为保护停车”时等待计时漏记。训练/保存/导出的合同一致，全部 4k 实际采样已做原因与时间因果审计。完整结果：

| 相同的 14 例开发任务 | 原基线 | V2 4k |
|---|---:|---:|
| 安全到达 | 12 | 11 |
| 碰撞 | 2 | 3 |
| 超时 / 假到达 | 0 / 0 | 0 / 0 |
| 共同成功任务路程 | 29.0608 m | 29.1394 m（+0.27%） |
| 共同成功任务用时 | 220.8 s | 224.2 s（+1.54%） |

`train_failure_stop_go_variation_1` 从原基线成功变成碰撞。见 [V2 完整结果](reports/stop_feedback_20261015/pilot_v2_4000_final/SUMMARY.json)。这些是开发任务，部分与训练同来源，不能与历史 46/48 混成一个成功率。

V2 真实训练中发现 326 帧“等待至少 2 s、仍有合法运动、继续选停止”，部分原采样停车概率约 99.6%。14 例常规评估的最长途中主动等待仅 0.2 s，覆盖不足。因此新增状态相关的有界温度探索：训练时使用已有感知/等待特征将温度从 1 提高至最多 3；采样与 PPO 更新用同一概率分布，确定性排序保持，停车仍合法。**本轮没有改变奖励、车体、动作、速度或保护。**

物理诊断中，提前等待、早期小右弧以及停滞后短暂前进存在安全到达轨迹；另一些合法转向仍超时或碰撞。它们是固定时间干预，**不是训练模型的成绩，也不构成强迫前进的依据**。新训练结果和待验证项见[探索实验说明](reports/stop_feedback_20261015/exploration_20261015/README.md)。

## 4. 下一台电脑优先做什么

1. 运行第 1 节完整性检查，查看 [本次快照](reports/handoff_learn_20261015/SNAPSHOT.json)，区分已完成训练和未完成评估。原电脑后台进程不会随 Git 迁移，快照之后的结果也不会自动出现。
2. 先复评现成探索版，不急着增加步数。固定 14 例安全回归，必须保留 V2 已退化的变体；记录真实到达、碰撞、超时、假到达和方向停车。
3. 做停滞恢复专项：使用 [frozen_recovery_cases.json](reports/stop_feedback_20261015/frozen_recovery_cases.json)，先按记录恢复前 20 s 的真实扫描、里程计、动作与内部历史，再完全交给模型。前缀不算模型成绩；比较接管后是否安全到达与长期等待。两例来自同一任务，不是泛化证明。
4. 若探索增加但仍不恢复，检查动作序列是否持续可行、旧策略是否又回到停滞姿态，以及进展奖励/价值的信用分配。不要仅凭更多非零动作或更少等待宣布成功。
5. 定位另补实车独立参考、开放区域与新增未入图障碍对照；先测轮速/IMU、时间戳与 TF，再决定补训 localizer/router。恢复 actor 训练不能直接修复它们。

### 复评模型，不重新训练

```bash
# 下列输出目录均须不存在；先按第 1 节安装编译依赖。
python tools/prepare_native_bundle.py \
  --source models/stop_feedback_20261015/baseline/portable \
  --output training/runs/learn_baseline
python tools/prepare_native_bundle.py \
  --source models/stop_feedback_20261015/ppo_explore_4000_s221015/portable \
  --output training/runs/learn_candidate

# 14 例 × 原/新，共 28 条闭环。旧结果目录不支持断点追加，换新目录完整重跑。
nice -n 19 python -m experiments.evaluate_docking_navigation \
  --candidate training/runs/learn_candidate --baseline training/runs/learn_baseline \
  --cases reports/stop_feedback_20261015/feedback_development_cases.json \
  --output training/runs/learn_paired --cpu-cores 0.35 \
  --candidate-name explored --baseline-name prior_constraints

# 单独验证停滞后接管；只在整条前缀物理/传感器一致性通过后评分。
nice -n 19 python -m experiments.evaluate_feedback_recovery \
  --bundle training/runs/learn_candidate \
  --specifications reports/stop_feedback_20261015/frozen_recovery_cases.json \
  --output training/runs/learn_recovery --label explored --cpu-cores 0.15
```

原策略的配套 439 维参考包是 `models/stop_feedback_20261015/ppo_4000_s221015/initial_portable`；恢复评估器不能直接使用 303 维基线。先用 `prepare_native_bundle.py` 重建该参考包，再以相同前缀评估，保持保护、车体和数据一致。

`frozen_recovery_smoke_cases.json` 仅重放 100 帧再让模型控制 1 帧，验证状态恢复；它在 20.2 s 时间上限结束，**不能将它的超时或 1 帧动作当作完整恢复能力**。

### 审计训练数据与重新开一轮

```bash
python -m experiments.audit_feedback_rollouts \
  --model-run models/stop_feedback_20261015/ppo_explore_4000_s221015 \
  --output training/runs/learn_causal_audit
python -m experiments.audit_exploration_rollouts \
  --model-run models/stop_feedback_20261015/ppo_explore_4000_s221015 \
  --output training/runs/learn_sampling_audit

# 独立重现实验：从原 300 维父模型初始化，不是从已训练 436 维模型续跑。
nice -n 19 python -m experiments.stop_feedback_pipeline \
  --baseline training/runs/learn_baseline \
  --checkpoint models/quality_recovery_20261014/policy_2000.zip \
  --training-cases reports/localization_mapping_20261015/blind_repair_20261015/stop_go_training_cases.json \
  --evaluation-cases reports/stop_feedback_20261015/feedback_development_cases.json \
  --model-output training/runs/learn_repeat_model --output training/runs/learn_repeat_report \
  --steps 4000 --seed 221015 --cpu-cores 0.35 \
  --feedback-schema stop_decision_feedback_v2 --record-rollouts \
  --exploration sensor_temperature_v1
```

**续训限制：** `StopFeedbackPPO.load(path, device='cpu')` 可恢复网络/优化器，但当前训练 CLI 仅提供 300→436 维初始化；不能直接将 `--checkpoint` 替换为新 436 维模型。若要在新权重上接着训练，应先实现语义一致的续训入口，记录环境重置、步数与课程。未保存正在运行的完整物理环境，不承诺逐步精确续跑。采样似然审计使用对应的更新前检查点，不能拿最终 `policy.zip` 重算前一批概率。

### 保守使用电脑资源

默认单线程 CPU，`--cpu-cores 0.35` 为软平均限制。本机独立 4k 大致 28～30 分钟、峰值约 2.5 GiB，后续配对评估另计；新机器需实测。硬上限可使用以下前缀替换 `nice -n 19`，后接完整 `python -m ...` 命令：

```bash
systemd-run --user --unit=agv-feedback \
  --property=CPUQuota=50% --property=MemoryHigh=3G --property=MemoryMax=4G \
  --property=Nice=19 --working-directory="$PWD" \
  --setenv=PATH="$PATH" --setenv=PYTHONPATH=training:. \
  --setenv=OPENBLAS_NUM_THREADS=1 --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1
```

`CPUQuota=50%` 是半个 CPU 核。用 `systemctl --user status agv-feedback` 与 `journalctl --user -u agv-feedback` 检查。原电脑服务名与归档时状态写在快照中；不要在两台电脑向同一目录并发写结果。

## 5. 地图、实车数据与项目结构

| 位置 | 内容 |
|---|---|
| `training/neural_navigation/` | 定位、地图、路由、方向目标、actor、动作可行性与保护 |
| `training/experiments/` | PPO、数据审计、定位/建图/闭环评估与停滞复现 |
| `deployment/ros2_recording_adapter/` | 实车记录与模型接口；需接入实际 ROS 工程 |
| [measured_motion](deployment/ros2_recording_adapter/examples/measured_motion/README.md) | 编码器＋IMU 输入与唯一 EKF 的配置说明 |
| `models/`、`reports/` | 基线、独立候选、训练来源和逐例成功/失败证据 |
| `training/assets/localization_replay_20261015/` | 28 条冻结仿真输入及仅供评估的真值 |
| `training/assets/real_recording_20261014/` | 1516 条实车决策输入、1911 条反馈、轮速/IMU 时序与 map_01 PGM/YAML |
| [FILES.json](docs/handoff/FILES.json) | 当前交接文件 SHA256 清单 |

**完整 `record.zip`、rosbag、相机图片、虚拟环境、缓存及平台 `.so` 没有上传。** 当前已提取输入的定位回放无需原包；若要重新提取 ROS 话题、分析 TF 或图像，请另行复制原始录包。精简数据来源与哈希见 [MANIFEST.json](training/assets/real_recording_20261014/MANIFEST.json)。

```bash
# 固定输入回放，不更新模型，不改变车辆轨迹。
nice -n 19 python -m experiments.evaluate_real_input_repair \
  --cache training/assets/localization_replay_20261015 \
  --analysis training/assets/real_recording_20261014/analysis \
  --record training/assets/real_recording_20261014/record \
  --hardware models/quality_recovery_20261014/portable/hardware.json \
  --output training/runs/learn_localization_replay --cpu-cores 0.35
```

更早阶段的交接说明保留在 [NEXT_README.md](docs/handoff/NEXT_README.md)，其中 V2“3/28”是旧快照，不是当前最终结果。另两个旧工作目录的未提交补丁见 [legacy_wip](docs/handoff/legacy_wip/README.md)，不要覆盖当前主线。历史绝对路径仅用作来源记录；上述迁移命令均以仓库根目录为起点。

当前主线仍未完成：必须用闭环证明模型在可行动作存在时能合理恢复，并同时保持碰撞、正常导航和方向停车能力。已知地图定位、在线建图和目标硬件验收要分别报告。
