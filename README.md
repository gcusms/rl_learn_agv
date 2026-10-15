# AGV 导航项目交接

更新：2026-10-15。交接分支：**`progress`**。本分支在 `handoff` 快照基础上归档最新源码、候选模型、可续训检查点、精简实车录制、评估报告和停车反馈训练入口，可在另一台电脑接着开发。

**当前没有后台训练。** 定位输入与静态匹配修复已有回放结果；最新停车反馈训练入口已经过功能检查和固定输入验证，**本轮计划的 4,000 步 PPO 尚未启动，也没有对应的已训练新模型**。原 2k 模型和方向停车模型均保留。最新候选尚未通过整机或实车部署验收。

| 交接时的进度 | 已完成 | 下一步 |
|---|---|---|
| 已知地图定位 | 运动反馈适配、静态支持筛选、时序融合和射线一致性检查；22 条冻结轨迹有对照 | 独立真实位姿验证，以及同协议闭环回归 |
| 停车原因与动态预测 | 新增 136 维显式反馈；58 帧旧输入前缀、初始动作完全一致，风险与保护层对齐 | 小规模闭环 PPO 和原/新模型配对评估 |
| PPO 与模型导出 | 436 维训练、439 维方向运行时；32 项相关功能检查通过 | 实际物理环境试训，验证等待改善且不增加碰撞 |
| 搬车重定位、在线建图 | 研究原型和历史失败证据已归档 | 分开验证；不能用已知地图回放成绩代替 |

先按第 1 节验证交接，再按第 7 节继续最新实验。详细设计与证据见[停车反馈说明](reports/stop_feedback_20261015/README.md)。

## 1. 换电脑后从这里开始

```bash
git clone --depth 1 --branch progress https://github.com/gcusms/rl_learn_agv.git
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

默认运行 61 项功能检查，其中 1 项原生内核数值对照在没有内核时跳过。要完整执行 61 项：

```bash
# 可选：只为重建历史候选和检查 Python/C++ 一致性。
sudo apt-get install g++ libeigen3-dev
python tools/check_handoff.py --build-native --output training/runs/handoff_check_native
```

内核在新输出目录编译并重新记录哈希，不修改已归档模型。最新射线候选默认使用 Python，本身无需编译。迁移验证见 [本次交接检查](docs/handoff/VERIFICATION.json)；实现检查通过不代表导航或实车精度验收。

## 2. 当前该看哪个模型

| 用途 | 仓库路径 | 说明 |
|---|---|---|
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
| [FILES.json](docs/handoff/FILES.json) | 本次交接代码、关键模型与回放数据的 SHA256 清单 |
| `training/experiments/stop_feedback_pipeline.py` | 最新停车反馈 PPO 试训与原/新模型配对评估入口 |

关键源码：[静态时序定位](training/neural_navigation/fast_static_localizer.py)、[射线一致性检查](training/neural_navigation/verified_static_localizer.py)、[测量里程计适配](training/neural_navigation/measured_odometry.py)、[全局恢复原型](training/neural_navigation/global_relocalizer.py)、[方向目标运行时](training/neural_navigation/goal_pose_runtime.py)。

已有分析：[硬件边界与重定位](reports/localization_mapping_20261015/hardware_boundary_20261015/REPORT.md)、[动态失败与后续入口](reports/localization_mapping_20261015/blind_repair_20261015/REPORT.md)、[弱几何与时序定位](reports/localization_mapping_20261015/sparse_repair_20261015/REPORT.md)、[方向停靠训练](reports/docking_training_20261014/REPORT.md)、[在线定位与建图](reports/localization_mapping_20261015/REPORT.md)。[先前定位方法说明](docs/localization/README.md)保留历史分支口径；[原进度概览](docs/PROGRESS_BEFORE_HANDOFF.md)包含原模型与查看器入口。

## 6. 下一步按什么顺序做

1. **先复现本分支。** 跑交接检查及固定输入回放，确认输入/模型合同、地图坐标、时间和数据齐全。旧 300/303 维与新实验 436/439 维入口不同，必须按 manifest 选择运行时。
2. **测准真实运动来源。** 实车用已测直线距离、左右转、制动和受阻片段对照指令、编码器和 IMU；确认轮径/符号、时间戳、数据年龄和唯一 TF 发布者。接入示例不会自动修改实车；指令超时停车需在实际发布链与底盘确认。
3. **给定位补独立参考。** 已知位置放入/移走地图外障碍，做正常与缺失扫描、弱几何、闭环恢复的配对试验。报告真实位置/航向误差和拒绝原因，不能用模型输出或轮速积分当参考真值。
4. **做原/新候选的闭环回归。** 同任务、相同保护和底盘，比较到达、碰撞、等待、实际路程/耗时及方向停车。先确认拒绝增加没有把机器人变成长期停车，再决定采用定位候选。
5. **验证停车反馈后再定训练预算。** 新观测已对齐 actor 与保护层的风险来源，并显式提供停车原因和数据年龄。按第 7 节完成物理环境试训与配对评估，再看横穿、反向、遮挡、静态误报和丢帧回归。此前仅 377/2000 步进入可训练范围；新入口允许所有有运动选项的状态更新 actor，但等待是否改善仍需实测。
6. **分别推进在线建图与板端性能。** 未知环境单独评估漂移、真回环和重定位；Jetson 测端到端 P95/P99、最大延迟与超时。电脑单组件耗时不能代替 Orin 结果。之后再按失败组件选择定位监督、路由监督或 PPO 微调。

恢复训练入口保留在 `training/experiments/train_localization_recovery.py`，原始检查点、各次配置、状态及源码快照也已归档。这一入口主要训练受条件启用的恢复分支，冻结 localizer/router/正常导航和方向停车；继续运行它不会直接修复定位或路由。已失败的配置仅作对照，不建议换电脑后直接接着大规模跑。

运行输出放 `training/runs/`，默认不跟踪。虚拟环境、缓存、平台 `.so` 和原始大录包不提交；新模型另存目录并附配置、种子、源码版本和评估，不覆盖历史基线。

## 7. 在新电脑继续最新停车反馈实验

新实验保留原 300 维导航输入，追加 136 维停车原因、逐动作拒绝类别、与保护层共用的接触风险和数据年龄。训练输入为 **436 维**；带目标朝向的部署候选为 **439 维**，必须使用 manifest 指定的 `StopFeedbackRuntime`。新增特征不包含仿真真值；方向停车分支、定位器、路由器冻结，完整导航 actor 可以更新。停止始终可选，不强迫车运动。

本轮没有改变奖励、12 动作、车体、4 cm 基本余量或速度上限。配对基线使用历史累计轮廓预测候选，暂不合并新的射线定位变体，以免同时改变两个因素。`reports/stop_feedback_20261015/frozen_replay/initial_portable` 是**未训练的接口迁移样例**，不是效果已提升的新模型；其中平台内核需重建。

```bash
source .venv/bin/activate
export PYTHONPATH=training:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# 先检查新增反馈、PPO 更新、重载/导出、重定位复位和原运行时，共 32 项。
nice -n 19 python -m unittest -v \
  training.test_decision_feedback experiments.test_stop_feedback_ppo \
  training.test_global_relocalizer training.test_navigation_runtime

# 基线使用原生定位内核；每台电脑在新的目录重建，不复制旧 .so。
sudo apt-get install g++ libeigen3-dev
python tools/prepare_native_bundle.py \
  --source models/localization_mapping_20261015/contour_mask_candidate/portable \
  --output training/runs/stop_feedback_baseline

# 在相同 58 帧已知失败输入上复核初始动作、观测前缀和保护风险的一致性。
nice -n 19 python -m experiments.replay_decision_feedback \
  --baseline training/runs/stop_feedback_baseline \
  --checkpoint models/quality_recovery_20261014/policy_2000.zip \
  --trace reports/localization_mapping_20261015/constraints_repair_20261015/combined_closed_loop_12/motion_profile_015_contour_mask_combined.npz \
  --cases reports/localization_mapping_20261015/constraints_repair_20261015/all_cases.json \
  --output training/runs/stop_feedback_replay

# 开始新一轮 4,000 步 PPO；完成后自动进行 12 例原/新模型配对导航评估。
# 所有输出目录必须尚不存在。新电脑从原 2k 检查点初始化本轮实验。
nice -n 19 python -m experiments.stop_feedback_pipeline \
  --baseline training/runs/stop_feedback_baseline \
  --checkpoint models/quality_recovery_20261014/policy_2000.zip \
  --training-cases reports/localization_mapping_20261015/blind_repair_20261015/stop_go_training_cases.json \
  --evaluation-cases reports/localization_mapping_20261015/constraints_repair_20261015/all_cases.json \
  --model-output training/runs/stop_feedback_4000_model \
  --output training/runs/stop_feedback_4000_report \
  --steps 4000 --seed 221015 --cpu-cores 0.35
```

`--cpu-cores 0.35` 控制软平均 CPU 用量；本轮默认单线程、CPU 训练并检查 3 GiB 进程内存预算。若还需要系统级硬限制，可将最后一个命令的 `nice -n 19` 改为下面的前缀，后面保留 `python -m ...` 及全部参数（需要可用的 systemd 用户服务）：

```bash
systemd-run --user --unit=agv-feedback \
  --property=CPUQuota=50% --property=MemoryHigh=3G --property=MemoryMax=4G \
  --property=Nice=19 --working-directory="$PWD" \
  --setenv=PATH="$PATH" --setenv=PYTHONPATH=training:. \
  --setenv=OPENBLAS_NUM_THREADS=1 --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1
```

查看 `training/runs/stop_feedback_4000_model/progress.json` 的实际步数、可更新决策帧、仅停止帧、主动等待和执行动作改写数；查看 `training/runs/stop_feedback_4000_report/status.json` 判断是在训练、配对评估还是失败。最终结果是该报告目录中的 `SUMMARY.json`。只有定位可信、运动有用时的长期等待才是优化目标，正常避让停车应保留。

**继续工作的验收条件：** 相同任务下到达、碰撞、假到达和超时不退化，正常路线和方向停靠保留，再判断无进展等待与行驶效率是否改善。4,000 步是排查信号的小规模试训，不是充分训练或实车可部署的保证；流水线不自动替换部署模型。当前没有这一轮完成结果，不能把功能检查、58 帧固定回放或旧 11/12 成绩写成本轮训练成功率。
