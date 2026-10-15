# AGV 导航项目交接

更新：2026-10-15。交接分支：**`sync`**。在 `progress`（`00be45c`）基础上归档最新源码、已完成的 4k PPO 检查点、基线、精简实车录制、评估快照及历史未提交工作，可在另一台电脑继续。

**最新停车反馈 PPO 已完成追加 4,000 步，耗时约 28.3 分钟；归档时原电脑仍在执行 12 例原/新模型配对评估。** 新模型尚未通过验收：已完成的对照中，原横穿案例 `motion_profile_015` 两个版本仍然碰撞。不能将训练回合成绩或部分评估写成最终成功率。原部署模型没有替换；已知地图定位候选与本轮 PPO 候选分开保留。

| 交接时的进度 | 已完成 | 下一步 |
|---|---|---|
| 已知地图定位 | 运动反馈适配、静态支持筛选、时序融合和射线一致性检查；22 条冻结轨迹有对照 | 独立真实位姿验证，以及同协议闭环回归 |
| 停车原因与动态预测 | 新增 136 维反馈；58 帧初始动作一致，风险与保护层对齐；V2 等待计时修复已单独版本化 | 当前训练模型仍用 V1；V2 尚未训练，不可只改 manifest 启用 |
| PPO 与模型导出 | 436 维训练、439 维方向运行时；4,000 步完成，网络/优化器与便携包均保存 | 先完成配对评估，分析等待与横穿碰撞，再决定下一轮 |
| 搬车重定位、在线建图 | 研究原型和历史失败证据已归档 | 分开验证；不能用已知地图回放成绩代替 |

先按第 1 节验证交接，再按第 7 节评估已经训练完的 4k 模型。冻结的进度和逐例证据见 [SNAPSHOT.json](reports/handoff_20261015/SNAPSHOT.json)；它不会随着原电脑后台任务自动更新。详细设计与证据见[停车反馈说明](reports/stop_feedback_20261015/README.md)。

## 1. 换电脑后从这里开始

```bash
git clone --depth 1 --branch sync https://github.com/gcusms/rl_learn_agv.git
cd rl_learn_agv
```

推荐先在 Ubuntu 22.04、Python 3.10 上复现。CPU 离线验证不需要 CUDA、ROS 2、显示器或实体机器人。建议预留至少 8 GB 安装空间；环境与 pip 缓存通常比模型本身大。不要复制旧机器的 `.venv`。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r training/requirements.txt

export PYTHONPATH=training:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# 文件与模型合同、303 维方向目标、28 条仿真输入、1516 条实车输入及功能检查。
# 输出目录必须不存在；不训练，不连接电机。
python tools/check_handoff.py --output training/runs/handoff_check
```

`training/requirements.lock.txt` 是旧 ROS 主机环境记录，包含系统提供的包，不作为跨电脑的 pip 安装清单。Jetson 应按本机 JetPack 安装匹配的 PyTorch，再处理其他依赖，不能照搬桌面 CUDA wheel。现有电脑可先用现成环境，无需重装。

默认运行 69 项功能检查，其中 1 项原生内核数值对照在没有内核时跳过；另外检查 4k 模型的 436 维检查点和 439 维 TorchScript。要执行全部 69 项及 439 维原生运行时试跑：

```bash
# 可选：只为重建历史候选和检查 Python/C++ 一致性。
sudo apt-get install g++ libeigen3-dev
python tools/check_handoff.py --build-native --output training/runs/handoff_check_native
```

内核在新输出目录编译并重新记录哈希，不修改已归档模型。最新射线候选默认使用 Python，本身无需编译。迁移验证见 [本次交接检查](docs/handoff/VERIFICATION.json)；实现检查通过不代表导航或实车精度验收。

## 2. 当前该看哪个模型

| 用途 | 仓库路径 | 说明 |
|---|---|---|
| 最新 PPO 试训候选 | [ppo_4000_s221015](models/stop_feedback_20261015/ppo_4000_s221015) | **追加 4,000 步、436/439 维、V1 反馈**；评估尚未完成，不能直接部署；最终检查点为 `policy.zip` |
| 最新定位实验候选 | [ray_consistent_candidate/portable](models/localization_mapping_20261015/ray_consistent_candidate/portable) | **303 维、12 动作、含目标朝向**；Python 射线一致性定位，带完整 manifest |
| 最新候选的父版本 | [temporal_native_candidate/portable](models/localization_mapping_20261015/temporal_native_candidate/portable) | 相同 actor/router 与方向停靠合同；历史原生定位，`.so` 需在目标机重建 |
| 原实车部署基线 | [quality_recovery_20261014](models/quality_recovery_20261014) | **300 维**、恢复阶段追加 2,000 步；包含可续训 ZIP、便携包及历史证据 |
| 方向停靠训练的历史检查点 | [docking_selected_5000_20261015](models/docking_selected_5000_20261015) | 从专项训练中选择 5,000 步检查点；不能仅凭停车成绩替换整机基线 |
| 未通过的后续 PPO / 算法对照 | [localization_mapping_20261015](models/localization_mapping_20261015) | 保留配置、权重、训练状态与源码快照供分析，不作为实车部署推荐 |

原基线 `policy_2000.zip` 的 SHA256 为 `7a3dad60f76060daf6b311b4bc6db921d3925ef0cdcf11683fe99e282173de11`。最新候选与父版本的 actor、router、硬件和方向目标文件逐字节相同；变化在定位合同和配套源码。

**运行时必须匹配模型。** 最新 303 维包通过 `neural_navigation.goal_pose_runtime.resolve_runtime(manifest['required_runtime'])` 解析到 `DockingRuntime`，不要直接套用原 300 维入口。加载前执行 `check_files(bundle, manifest)`；完整示例见 [check_handoff.py](tools/check_handoff.py)。单独复制 `actor.pt` 或 `localizer.pt` 不能获得这轮定位改进。

车体保持约 39.39×32.95 cm、4 cm 基本保护余量；前进上限 0.20 m/s、转向上限 0.50 rad/s，当前 12 动作不包含横移和倒车。输入仍为约 219.44° 视野的 220 束雷达，后方盲区保留。最新定位按 10 m 有效范围处理，旧硬件合同的 12 m 上限不能当作实车已观察到的空闲。

## 3. 目前改善了什么，还卡在哪里

最新工作见 [完整定位结果](reports/localization_mapping_20261015/real_input_repair_20261015/REPORT.md)、[汇总数据](reports/localization_mapping_20261015/real_input_repair_20261015/SUMMARY.json)。

| 同一固定回放：22 条原验证轨迹、2855 帧 | 上一轮时序定位 | 最新射线检查 |
|---|---:|---:|
| 位置误差 P95 | 18.51 cm | 18.04 cm |
| 航向误差 P95 | 1.526° | 1.368° |
| 各轨迹终点位置误差均值 | 3.42 cm | 3.27 cm |
| 本机 Python 定位 CPU P95 | 7.88 ms | 14.59 ms |

这是已有开发回归上的小幅改善，代价是耗时增加；不是实车精度、未见地图泛化或“超过 AMCL”的证明。本次没有重新比较 AMCL。长墙、稀疏结构和空旷区域仍有明显误差，不能保证仅比车宽 10–15 cm 的窄道通行。

实车记录做了 1516 帧、22 个任务代次的固定输入对照：原 EKF 输入可接受的扫描更新从 832 帧降为 583 帧，轮速/IMU 输入从 830 降为 606 帧。射线检查增加了一道拒绝矛盾配准的条件，但**更多拒绝也可能造成更多等待**。录制没有独立绝对位姿真值，不能把估计位移变小称作实车误差达标。

当前状态要分开看：

- **已知地图定位：** 里程计传播＋静态支持筛选＋弱方向约束＋0.2/0.4 s 历史融合；最新增加自由射线与地图矛盾检查。需要可靠初始地图位姿。
- **测量运动输入：** 新增实测轮速 `(vx, vy, wz)`＋IMU Z 角速度适配；保留麦轮横向测量，但不增加横移指令。积分断续后需恢复参考。必须显式接入，加载模型不会自动把底盘从指令积分切为编码器反馈。
- **搬车重定位：** 已有全图候选搜索与停车多帧确认原型，尚未自动接入运行时。历史实车 6 段均未确认，不能宣称搬车后自动恢复已完成。
- **边导航边建图：** 已有在线建图/SLAM 实验入口，但长距离漂移、假到达和动态任务失败尚未解决；最新已知地图回放不能替代它的验收。
- **方向停车：** 模型及运行时接受 `(x, y, yaw)`，同位置改朝向可产生原地转向；仍需闭环停车精度、恢复后停靠与实车回归。
- **动态避障：** 原风险输入与保护层预测来源不完全一致，停走横穿失败仍在。后两次各 2,000 步试训均为 11/12 安全到达、1 次碰撞，共同成功任务路程分别增加 2.10% 和 8.78%，未采用。

原 2k 基线历史固定开发集是 **46/48 安全到达、1 碰撞、1 超时**。另一轮“9/9、专项 2/5”和最新 12 例开发对照使用不同协议，不能合并或直接排名。最新候选尚未完成与原基线同协议的完整闭环对照。

## 4. 不依赖原电脑路径的实车与仿真回放

已随仓库保存：

- [28 条冻结仿真输入](training/assets/localization_replay_20261015)：扫描、里程计、时间和仅供评估的真值标签；地图与物理场景也在仓库中。
- [紧凑实车输入](training/assets/real_recording_20261014)：原录制中的 1516 条决策输入、1911 条反馈、轮速/IMU 时序和 map_01 配套 PGM/YAML。
- [实车输入来源及哈希](training/assets/real_recording_20261014/MANIFEST.json)：扫描、里程计及地图数值不变，仅去掉当前回放未使用的 JSON 字段；不是新真值数据。

**完整 `record.zip`、rosbag 和相机图像未放入 Git。** 当前定位回放无需它们；若要重新提取 ROS 话题、检查 TF 或分析图像，请另行复制原始录包。手动 2D Pose 初值不能作为真值。历史报告中的原机器绝对路径保留作来源记录，下面的当前入口均使用仓库相对路径。

```bash
source .venv/bin/activate
export PYTHONPATH=training:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# 本机此前约 5 分钟，软平均约 0.35 个 CPU 核；其他机器耗时需实测。
# 不启动 ROS/MuJoCo，不修改机器人轨迹，不训练。
nice -n 19 python -m experiments.evaluate_real_input_repair \
  --cache training/assets/localization_replay_20261015 \
  --analysis training/assets/real_recording_20261014/analysis \
  --record training/assets/real_recording_20261014/record \
  --hardware models/quality_recovery_20261014/portable/hardware.json \
  --output training/runs/handoff_replay --cpu-cores 0.35
```

输出 `RESULTS.json`、逐轨迹 NPZ 及资源统计。与 [历史最终回放](reports/localization_mapping_20261015/real_input_repair_20261015/paired_packed/RESULTS.json) 对照位置、航向、接受/拒绝原因；不同机器的耗时、RSS 和紧凑输入文件哈希会不同。相同轨迹再次回放不算新的独立成功样本。

## 5. 项目结构与继续工作的入口

| 位置 | 内容 |
|---|---|
| `training/neural_navigation/` | 定位、地图融合、路由、策略运行时、动作可行性和保护 |
| `training/experiments/` | PPO、定位/建图实验、回放、配对评估和候选生成 |
| `deployment/ros2_recording_adapter/` | 实车记录、模型适配与发布接口；需与实际 ROS 工程集成 |
| [measured_motion](deployment/ros2_recording_adapter/examples/measured_motion/README.md) | 编码器输入和单一轮速＋IMU EKF 的接入说明及 YAML 示例 |
| `models/` | 已归档基线、方向停车及后续独立实验；保留优化器等训练产物 |
| `reports/` | 各轮任务、逐例结果、失败证据与回放 |
| `tools/check_handoff.py` | 换电脑后的完整性与运行时检查 |
| `tools/prepare_native_bundle.py` | 从原生候选源码重建目标平台内核，输出新包 |
| [FILES.json](docs/handoff/FILES.json) | 本次交接代码、模型、文档和回放数据的 SHA256 清单 |
| `training/experiments/stop_feedback_pipeline.py` | 最新停车反馈 PPO 试训与原/新模型配对评估入口 |

关键源码：[静态时序定位](training/neural_navigation/fast_static_localizer.py)、[射线一致性检查](training/neural_navigation/verified_static_localizer.py)、[测量里程计适配](training/neural_navigation/measured_odometry.py)、[全局恢复原型](training/neural_navigation/global_relocalizer.py)、[方向目标运行时](training/neural_navigation/goal_pose_runtime.py)。

已有分析：[硬件边界与重定位](reports/localization_mapping_20261015/hardware_boundary_20261015/REPORT.md)、[动态失败与后续入口](reports/localization_mapping_20261015/blind_repair_20261015/REPORT.md)、[弱几何与时序定位](reports/localization_mapping_20261015/sparse_repair_20261015/REPORT.md)、[方向停靠训练](reports/docking_training_20261014/REPORT.md)、[在线定位与建图](reports/localization_mapping_20261015/REPORT.md)。[先前定位方法说明](docs/localization/README.md)保留历史分支口径；[原进度概览](docs/PROGRESS_BEFORE_HANDOFF.md)包含原模型与查看器入口。

## 6. 下一步按什么顺序做

1. **先复现本分支。** 跑交接检查及固定输入回放，确认输入/模型合同、地图坐标、时间和数据齐全。旧 300/303 维与新实验 436/439 维入口不同，必须按 manifest 选择运行时。
2. **测准真实运动来源。** 实车用已测直线距离、左右转、制动和受阻片段对照指令、编码器和 IMU；确认轮径/符号、时间戳、数据年龄和唯一 TF 发布者。接入示例不会自动修改实车；指令超时停车需在实际发布链与底盘确认。
3. **给定位补独立参考。** 已知位置放入/移走地图外障碍，做正常与缺失扫描、弱几何、闭环恢复的配对试验。报告真实位置/航向误差和拒绝原因，不能用模型输出或轮速积分当参考真值。
4. **做原/新候选的闭环回归。** 同任务、相同保护和底盘，比较到达、碰撞、等待、实际路程/耗时及方向停车。先确认拒绝增加没有把机器人变成长期停车，再决定采用定位候选。
5. **先评估已完成的 4k 模型，再定下一轮。** 新观测已对齐 actor 与保护层的风险来源，并提供实际停车原因。此前仅 377/2000 步进入可训练范围；本轮 3336/4000 步有可选运动。第 7 节直接评估现成权重，无需先重复训练。横穿碰撞仍在；V2 计时修复仅通过功能检查，不能宣称已学习改善。
6. **分别推进在线建图与板端性能。** 未知环境单独评估漂移、真回环和重定位；Jetson 测端到端 P95/P99、最大延迟与超时。电脑单组件耗时不能代替 Orin 结果。之后再按失败组件选择定位监督、路由监督或 PPO 微调。

恢复训练入口保留在 `training/experiments/train_localization_recovery.py`，原始检查点、各次配置、状态及源码快照也已归档。这一入口主要训练受条件启用的恢复分支，冻结 localizer/router/正常导航和方向停车；继续运行它不会直接修复定位或路由。已失败的配置仅作对照，不建议换电脑后直接接着大规模跑。

运行输出放 `training/runs/`，默认不跟踪。虚拟环境、缓存、平台 `.so` 和原始大录包不提交；新模型另存目录并附配置、种子、源码版本和评估，不覆盖历史基线。

## 7. 在新电脑继续最新停车反馈实验

新增输入保留原 300/303 维前缀，再追加 136 维实际停车原因、逐动作拒绝类别、保护层接触风险、数据年龄和等待状态。导航 PPO 为 **436 维**，带方向目标的完整运行时为 **439 维**，使用 manifest 指定的 `StopFeedbackRuntime`。本轮导航 actor 可以更新；localizer、router 和方向停靠分支冻结，进入目标前约 0.3 m 的方向分支仍使用原模型。这次没有改变奖励、动作、车体、保护距离或速度上限。

| 已完成的训练信息 | 数值 / 位置 |
|---|---|
| 原始父检查点 | `models/quality_recovery_20261014/policy_2000.zip` |
| 本轮追加步数 / 种子 | 4,000 / `221015` |
| 有运动选项 / 仅停止可选 | 3,336 / 664 步 |
| 非零采样动作被保护层改写 | 0 次 |
| 训练中已结束回合 | 18；14 次成功、3 次碰撞、1 次超时；**不是验证成功率** |
| 训练时间 / 峰值 RSS | 1,698.84 s（28.3 分钟）/ 2,459.86 MiB |
| 平均 CPU | 0.350 核；单线程 CPU 训练，未用 GPU |
| 最终检查点 | `models/stop_feedback_20261015/ppo_4000_s221015/policy.zip` |
| 完整运行包 | 同目录 `portable/`，含 actor、定位/路由权重、观测与动作合同 |
| 配置、训练状态与来源 | 同目录 `config.json`、`status.json`、`PROVENANCE.json`、`source_snapshot/` |
| 当前评估快照 | [逐例结果](reports/handoff_20261015/paired/RESULTS.json)、[运行进度](reports/handoff_20261015/SNAPSHOT.json) |

训练检查点包含网络和优化器，`random_state.pt` 保存随机状态；没有保存进行中 MuJoCo 环境和 rollout 缓冲，不能承诺逐步完全一致地断点续跑。`policy_step_4000.zip` 是收集最后一段 rollout 后、该段 PPO 更新前的回调产物，**最终更新后的模型用 `policy.zip`**。归档配置中的原电脑绝对路径只作来源记录，下面使用仓库相对路径。

**当前限制：** 固定 58 帧检查中新增风险与保护层一致，不代表学会了更好的避障。2k 中间检查点新增输入权重已变化，但该片段的确定性动作没有变化。4k 配对评估中，`motion_profile_015` 原、新均在 11.6 s 碰撞。不能宣称停车问题已解决或最新模型优于原部署版。

V1 等待计时只累计标记为 `policy_wait` 的停车，漏记“零速度带激光保护原因、但其他运动仍合法”的状态。源码新增独立 `stop_decision_feedback_v2` 修复此计时并保留实际原因；**归档 4k 模型按 V1 训练、默认继续使用 V1**。V2 尚未接通训练 CLI / 导出流程，也没有训练权重；下一轮需同步修改训练与导出合同并验证，不能仅修改已有模型的 JSON。

### 7.1 先评估现成模型，不重复训练

```bash
source .venv/bin/activate
export PYTHONPATH=training:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# 在本机重新编译两个新包。输出目录必须尚不存在。
sudo apt-get install g++ libeigen3-dev
python tools/prepare_native_bundle.py \
  --source models/stop_feedback_20261015/baseline/portable \
  --output training/runs/sync_baseline
python tools/prepare_native_bundle.py \
  --source models/stop_feedback_20261015/ppo_4000_s221015/portable \
  --output training/runs/sync_candidate

# 完整重跑相同 12 例、两个版本，共 24 条闭环导航，不更新权重。
nice -n 19 python -m experiments.evaluate_docking_navigation \
  --baseline training/runs/sync_baseline \
  --candidate training/runs/sync_candidate \
  --cases reports/localization_mapping_20261015/constraints_repair_20261015/all_cases.json \
  --baseline-name prior_constraints --candidate-name stop_feedback_ppo \
  --output training/runs/sync_paired --cpu-cores 0.35

# 区分任务途中、方向对齐和仅能停止的等待；输出目录必须尚不存在。
nice -n 19 python -m experiments.audit_decision_waits \
  --evaluation training/runs/sync_paired/RESULTS.json \
  --output training/runs/sync_wait_audit
```

完整评估应有 `completed=true`、24 条结果。原任务是已经使用过的开发集，不是独立测试集；原机后台后续结果不会自动出现在本次 Git 快照中。新电脑应重跑完整 24 条，当前评估器不支持在已有目录中接着追加。比较时先看安全到达、碰撞、假到达、超时及方向停车，再看共同成功任务的路径/时间和长期无进展等待。不能用减少停车时间抵消碰撞或正常任务退化。

### 7.2 训练与诊断入口

- `experiments.stop_feedback_pipeline`：从原 **300 维 2k** 检查点初始化新的 436 维试验，训练后自动配对评估。`--checkpoint` **不能直接换成 4k 的 436 维 `policy.zip`**，该入口目前只支持 300→436 的迁移初始化。
- `experiments.train_stop_feedback`：单独训练和导出；原生基线必需、默认 V1、步数限 1,000～10,000 且为 250 的倍数。
- `experiments.stop_feedback_ppo.StopFeedbackPPO.load(path, device='cpu')`：加载本轮 436 维网络和优化器；若继续微调，需要接回一致的 `StopFeedbackWrapper`，同步实现续训/导出入口，并记录新回合、步数和随机状态。单独加载不等于恢复完整环境。
- `experiments.inspect_feedback_learning`：固定输入上的权重/概率诊断；已有 [2k 检查](reports/stop_feedback_20261015/learning_inspection_2000) 及 436 维冻结观测可复用。
- `experiments.audit_recorded_stops`：分析原 `record.zip` 的停车原因；[分析结果](reports/stop_feedback_20261015/RECORDED_STOP_AUDIT.json) 已归档，重新提取时需另行复制原 ZIP。

若需要从原基线复现实验，使用新输出目录：

```bash
nice -n 19 python -m experiments.stop_feedback_pipeline \
  --baseline training/runs/sync_baseline \
  --checkpoint models/quality_recovery_20261014/policy_2000.zip \
  --training-cases reports/localization_mapping_20261015/blind_repair_20261015/stop_go_training_cases.json \
  --evaluation-cases reports/localization_mapping_20261015/constraints_repair_20261015/all_cases.json \
  --model-output training/runs/feedback_repeat_model \
  --output training/runs/feedback_repeat_report \
  --steps 4000 --seed 221015 --cpu-cores 0.35
```

下一轮先完成 4k 配对结果分析，再对 V2 等待计时、可更新样本覆盖和关键碰撞前的决策单独立项。停止始终合法，不强迫前进；只剩停止可选的状态不能靠强迫 actor 探索解决。历史两例方向等待在当前基线已分别于 1.2 s、3.8 s 完成，见 [原基线复核](reports/stop_feedback_20261015/current_heading_probe/RESULTS.json)，不能将它们的改善归功于本轮 PPO。

### 7.3 保守的资源限制

`--cpu-cores 0.35` 是软平均限额；本机训练约 28 分钟，**不含随后完整配对评估**，换电脑需重新测量。默认单线程、CPU 训练，训练进程检查约 3 GiB RSS 预算。若需要硬上限，可将对应命令的 `nice -n 19` 换为下面前缀，后接 `python -m ...` 和完整参数：

```bash
systemd-run --user --unit=agv-feedback \
  --property=CPUQuota=50% --property=MemoryHigh=3G --property=MemoryMax=4G \
  --property=Nice=19 --working-directory="$PWD" \
  --setenv=PATH="$PATH" --setenv=PYTHONPATH=training:. \
  --setenv=OPENBLAS_NUM_THREADS=1 --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1
```

`CPUQuota=50%` 表示半个 CPU 核；不是占整台多核电脑的 50%。查看 `systemctl --user status agv-feedback` 和 `journalctl --user -u agv-feedback`。原机单位名为 `rl-stop-feedback-20261015.service`，服务配置未安装到新电脑。

## 8. 交接边界与历史工作

本分支包含当前训练代码、模型/优化器、训练来源快照、地图、必要回放输入、已完成的诊断和冻结的部分配对评估；不包含 `.venv`、pip 缓存、平台 `.so`、ROS 编译目录和完整原始录包。新电脑重建原生包后再运行，不能跨平台直接复制 `.so`。当前跟踪文件约 550～600 MB，浅克隆可避免拉取全部历史。

另外两个旧工作目录的未提交补丁和未归档源码保存在 [legacy_wip](docs/handoff/legacy_wip/README.md)，基准提交与文件哈希均记录。它们属于历史实现，不要覆盖当前 `training/`。原机器各工作副本及部署模型保留。

**交接验证与效果验收不同。** [VERIFICATION.json](docs/handoff/VERIFICATION.json) 记录本次文件、模型加载、原生重建及实现检查；定位准确性仍缺独立实车参考，在线 SLAM、搬车自动恢复、Orin Nano 耗时和整体部署验收仍待完成。
