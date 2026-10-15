# AGV 导航：换电脑继续开发

交接分支：**`transfer`**；日期：**2026-10-16（北京时间）**；仓库：[gcusms/rl_learn_agv](https://github.com/gcusms/rl_learn_agv/tree/transfer)。包含当前源码、原实车模型、最新实验模型、优化器、必要地图、精简实车录制、失败对照和运行入口。精确归档时刻及后台进度见 [SNAPSHOT.json](reports/handoff_transfer_20261016/SNAPSHOT.json)。

**已有开发集改善，尚未通过部署验收。** 最新提前规避候选在固定 14 例中全部安全到达，6 例方向停靠保留；另 4 个不同条件仅 1 例成功、3 例碰撞。它是示范学习初始化模型，**追加 PPO 步数为 0**。原实车部署模型保留。

## 1. 当前进度与结果

当前重点是让策略理解停车原因，在存在有用、安全动作时尝试恢复，同时保留合理等待、避障和方向停靠。

| 工作 | 已完成结果 | 限制 |
|---|---|---|
| 停车反馈 | 增加停止原因、动作可行性、预测约束、因果等待历史；训练 436 维，方向运行时 439 维 | 输入增加本身不等于导航改善 |
| 首轮恢复 PPO | 示范初始化＋2,000 步；固定 14 例为 13 到达、1 碰撞 | 与自身 BC 初始化的确定性轨迹一致，未证明 PPO 额外收益 |
| 第二轮 steady-base PPO | 独立 2,000 步；14 例仍为 13 到达、1 碰撞；2 个前缀和 3 个有效留出接管均到达 | 与首轮结果相同，限制随机探索解决不了可训练范围不足 |
| 最新 proactive BC | 近障接近阶段也可学习；1,058 条示范状态、400 次拟合；14 例 **14/14** 到达 | 9 条正常任务用于保留样本，是开发回归，不是未见地图测试 |
| 不同条件完整任务 | 最新 BC **1/4** 到达、3 碰撞；同输入原参考也是 1/4，但成功任务不同 | 有原成功任务退化，不能替换部署模型 |
| w11 随机训练停滞 | 原参考及三种右转/右弧诊断均超时；新 BC 的 90/95/106 秒接管对照进行中 | 动作合法不等于能成功脱困 |
| 下一轮 | 8 个冻结训练条件正在采集完整诊断轨迹；8k PPO 计划准备好，**未启动** | 干预轨迹不是模型成绩；需审核后才可选入示范 |

首轮 PPO 用时 **1,257.38 秒**、峰值 RSS **2,393 MiB**；第二轮独立 PPO 用时 **1,217.74 秒**、峰值 RSS **2,436 MiB**，不是同一模型累计 4k。最新 BC 用时 **100.84 秒**、峰值 RSS **1,775 MiB**；1,058 条示范的最大概率动作全部拟合正确，但最低运动标签概率 **0.567**，未达到 0.9 目标。

| 同一固定 14 例 | 安全到达 | 碰撞 | 超时 / 假到达 | 方向停靠 |
|---|---:|---:|---:|---:|
| 冻结原参考 | 12/14 | 2 | 0 / 0 | 6/6 |
| 首轮 BC / 首轮 BC＋2k PPO | 13/14 | 1 | 0 / 0 | 6/6 |
| 第二轮 steady-base BC＋2k PPO | 13/14 | 1 | 0 / 0 | 6/6 |
| 最新 proactive BC（0 PPO） | **14/14** | **0** | 0 / 0 | 6/6 |

最新模型将 `motion_profile_015` 从 11.6 秒碰撞改善为 **49.4 秒安全到达**，保留原 13 项成功任务。两个 20 秒前缀接管分别在总计 **47.2 / 52.2 秒**到达，接管后最长有运动选项却等待 **0 / 0.2 秒**。时间包含初始化，不是从起点自主导航成绩；两例同源且与训练相关。

另 4 个条件均从真正起点执行，失败不排除：候选改善条件 01，但使原参考能完成的条件 02 碰撞。分析发现同一位置、同一扫描下候选在 5.2 秒提前前进；接触前一帧车速近零、只剩停车合法。需要同时学会合理等待和提前绕行；尚未证明这是单一因果机制。

报告：[首轮 42 条](reports/stop_feedback_20261015/prefix_recovery_threeway_20261016/RESULTS.json)、[第二轮 14 例](reports/stop_feedback_20261015/steady_base_navigation_2000_20261016/RESULTS.json)、[最新 14 例](reports/stop_feedback_20261015/proactive_bc_navigation_20261016/RESULTS.json)、[两个接管](reports/stop_feedback_20261015/proactive_bc_prefix_recovery_20261016/RESULTS.json)、[4 条件配对](reports/stop_feedback_20261015/proactive_bc_full_reset_holdout_20261016/RESULTS.json)、[失败分析](reports/stop_feedback_20261015/proactive_bc_failure_analysis_20261016/RESULTS.json)。这 14 例与原模型历史 46/48 不同，不能合并统计。

## 2. 新电脑安装与检查

推荐先用 Ubuntu 22.04、Python 3.10 复现。CPU 检查不需要 CUDA、ROS 2 或显示器；不要复制旧虚拟环境。

```bash
git clone --depth 1 --branch transfer https://github.com/gcusms/rl_learn_agv.git
cd rl_learn_agv
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r training/requirements.txt
export PYTHONPATH=training:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# 输出目录须不存在；不连接电机、不训练。
python tools/check_handoff.py --output training/runs/transfer_check

# 原生定位库与平台有关，在目标电脑重建。
sudo apt-get install g++ libeigen3-dev
python tools/check_handoff.py --build-native --output training/runs/transfer_check_native
```

预留至少 8 GB 安装空间，下载缓存另计。`training/requirements.lock.txt` 是历史主机记录，不是跨电脑安装清单；Jetson 使用匹配 JetPack 的 PyTorch。检查入口核验 6 份检查点、TorchScript、输入合同、冻结 actor 和 108 项功能测试；启用原生检查时重建 7 个包。实际记录见 [VERIFICATION.json](docs/handoff/VERIFICATION.json)。这是迁移检查，不是实车验收。

## 3. 模型选用

| 路径（仓库根目录下） | 用途 |
|---|---|
| `models/quality_recovery_20261014/policy_2000.zip` | **原实车部署基线**，300 维，历史开发集 46/48；保留 |
| `models/localization_mapping_20261015/ray_consistent_candidate/portable` | 射线一致性定位候选，303 维；算法改善，未实车验收 |
| `models/stop_feedback_20261015/prefix_ppo_2000_s231015` | 首轮 BC＋2k PPO，含优化器、随机状态、采样记录 |
| `models/stop_feedback_20261015/steady_base_ppo_2000_s231015` | 独立 2k PPO，收敛结果与首轮相同 |
| **`models/stop_feedback_20261015/proactive_bc_s231015_prepared`** | **最新候选**：`policy_bc.zip` / `bc_portable`；0 PPO，没有最终 PPO `policy.zip` |
| `models/stop_feedback_20261015/ppo_4000_s221015`、`ppo_v2_4000_s221015`、`ppo_explore_4000_s221015` | 历史失败对照，回归退化 |
| `models/docking_selected_5000_20261015` | 历史位置＋方向停靠模型，不能凭专项成绩替换整机 |

原实车基线 SHA256：`7a3dad60f76060daf6b311b4bc6db921d3925ef0cdcf11683fe99e282173de11`。

`policy_migrated.zip` 是零残差初始化；`policy_bc.zip` / `bc_portable` 是示范初始化；只有完成 PPO 的 `policy.zip` / `portable` 才是最终 PPO 产物。`policy_step_*.zip` 是更新前回调检查点，主要用于采样概率审计。

原基线 300 维，方向候选 303 维；反馈训练 436 维、方向运行时 439 维。用 `neural_navigation.goal_pose_runtime.check_files` 核验包，再根据 manifest 的 `required_runtime` 调用 `resolve_runtime`。恢复检查点用 `StopRecoveryPPO.load`，旧反馈模型用 `StopFeedbackPPO.load`。例子见 [check_handoff.py](tools/check_handoff.py)。**模型、特征、manifest、运行时需一起迁移。**

车体约 39.39×32.95 cm，保护余量 4 cm；12 动作，前进上限 0.20 m/s、转向上限 0.50 rad/s，不含横移或倒车。雷达 220 束、约 219.44°，保留后方盲区；最新定位按 10 m 处理，历史 actor 按 12 m 归一化。没有放宽安全条件，不要直接倍乘旧模型速度或修改归一化。

## 4. 新电脑从哪里继续

先复评，再改善不同条件下的提前决策和合理等待。旧电脑的后台进程不会随 Git 迁移；快照中 `completed=false` 仅代表部分进度。

### 4.1 重建包、复评

以下命令不依赖旧电脑服务；一次运行一个，输出目录均须不存在。

```bash
python tools/prepare_native_bundle.py \
  --source models/stop_feedback_20261015/proactive_bc_s231015_prepared/bc_portable \
  --output training/runs/transfer_proactive
python tools/prepare_native_bundle.py \
  --source models/stop_feedback_20261015/ppo_4000_s221015/initial_portable \
  --output training/runs/transfer_reference

# 固定 14 例开发回归。
nice -n 19 python -m experiments.evaluate_docking_navigation \
  --candidate training/runs/transfer_proactive \
  --cases reports/stop_feedback_20261015/proactive_protocol_20261016/evaluation_cases.json \
  --output training/runs/transfer_navigation --candidate-name proactive_bc --cpu-cores 0.35

# 四个条件，从真正起点完整执行并保留失败。
nice -n 19 python -m experiments.evaluate_docking_navigation \
  --candidate training/runs/transfer_proactive --baseline training/runs/transfer_reference \
  --cases reports/stop_feedback_20261015/recovery_holdout_reference_20261016/cases.json \
  --output training/runs/transfer_conditions \
  --candidate-name proactive_bc --baseline-name frozen_reference --cpu-cores 0.35

# 两个 20 秒物理前缀后接管；初始化不是模型成绩。
nice -n 19 python -m experiments.evaluate_feedback_recovery \
  --bundle training/runs/transfer_proactive \
  --specifications reports/stop_feedback_20261015/frozen_recovery_cases.json \
  --output training/runs/transfer_prefixes --label proactive_bc --cpu-cores 0.35

# 另一条随机训练停滞，分别在 90、95、106 秒接管。
nice -n 19 python -m experiments.evaluate_feedback_recovery \
  --bundle training/runs/transfer_proactive \
  --specifications reports/stop_feedback_20261015/proactive_protocol_20261016/w11_prefixes.json \
  --output training/runs/transfer_w11 --label proactive_bc --cpu-cores 0.15
```

接管参考需要 439 维 `initial_portable`，不能用 303 维父包。`reports/.../proactive_protocol_20261016/validate_bc.py`、`collect_after_bc.py`、`w11_after_holdout.py` 是旧电脑执行来源，等待原机服务；新电脑不要直接执行它们。

### 4.2 补数据，再决定 8k PPO

1. 看 [PPO_PLAN.json](reports/stop_feedback_20261015/proactive_protocol_20261016/PPO_PLAN.json) 和失败分析。训练条件 12 个，新增 8 个；早/晚前缀 6 / 20 秒。4 个对照条件不用于本轮训练示范。
2. 用下方命令生成 8 条**离线诊断**。右转 6～7.4 秒、前进 20～22 秒是预先声明的示范干预，只在原掩码允许时执行。碰撞、超时全部保留，不是模型自主成功。
3. 审核完整物理到达、假到达、碰撞、保护、掩码、执行命令与哈希，再明确选入安全完整轨迹。**当前自动合并与合格性门槛尚未完成，不会自动进入 PPO。** 保留正常导航和合理等待样本。
4. 新目录重新拟合 BC，检查每个来源、最低标签概率，再回归普通导航、方向停靠、4 条件、w11。退化时先修复，不直接扩步数。
5. 合格后才运行计划中的 8,000 步，并与该轮自身 BC 对照。当前入口从原 300 维基线重新做相同 BC，**不是直接接续新 BC 权重和优化器**；需核对初始化权重与已验证版本一致。

```bash
# 仅采集声明的合法干预轨迹，不训练、不修改模型。
nice -n 19 python -m experiments.generate_proactive_demonstrations \
  --bundle training/runs/transfer_reference \
  --cases reports/stop_feedback_20261015/proactive_protocol_20261016/ppo_training_cases.json \
  --output training/runs/transfer_training_probes --cpu-cores 0.35
```

复现**当前 1,058 条示范的 BC**使用下方入口，不会自动合并新数据。`--steps 2000` 是接口范围参数，`--initialization-only` 使 PPO 实际为 0 步；还会真实重放 100 帧并检查两帧模型动作，不等于完整导航验收。

```bash
python tools/prepare_native_bundle.py \
  --source models/stop_feedback_20261015/baseline/portable \
  --output training/runs/transfer_parent
nice -n 19 python -m experiments.train_prefix_recovery \
  --bundle training/runs/transfer_parent \
  --checkpoint models/quality_recovery_20261014/policy_2000.zip \
  --training-cases reports/stop_feedback_20261015/proactive_protocol_20261016/training_cases.json \
  --prefixes reports/stop_feedback_20261015/proactive_protocol_20261016/bc_initialization_prefixes.json \
  --demonstrations reports/stop_feedback_20261015/proactive_protocol_20261016/demonstrations.json \
  --output training/runs/transfer_bc_reproduction \
  --proactive-recovery --initialization-only --steps 2000 --seed 231015 --cpu-cores 0.35
```

审核合格后可去掉 `--initialization-only`、设置 `--steps 8000`，换成审核后的示范清单、`ppo_training_cases.json`、`ppo_training_prefixes.json` 和新输出目录。不要直接使用未审核轨迹。已完成 PPO 的检查点包含网络/优化器，`random_state.pt` 保存 RNG；CLI 尚不支持精确恢复半回合、课程计数和全部物理状态。换机独立训练需记录新起点，不能宣称无缝续跑。

## 5. 定位和方向停车

定位改善来自里程计传播、静态支持筛选、弱方向约束、时序融合和自由射线矛盾检查，**未训练新定位网络**。22 条冻结轨迹、2,855 帧同输入中，时序父版 / 射线候选位置 P95 **18.51 / 18.04 cm**，航向 P95 **1.526 / 1.368°**，终点平均误差 **3.42 / 3.27 cm**。本机 Python 定位 CPU P95 **7.88 / 14.59 ms**：改善小、耗时增加。

实车无独立绝对位姿真值，不能声称超过 AMCL。恢复实验仍用时序定位父包，未自动融合射线候选。开放区域、边导航边建图、搬车全局重定位、Orin Nano 端到端预算待验收。位置＋方向目标已接入，同位置不同方向可转向。

说明：[定位算法与验证](docs/localization/README.md)、[实车输入修复](reports/localization_mapping_20261015/real_input_repair_20261015/REPORT.md)、[编码器＋IMU](deployment/ros2_recording_adapter/examples/measured_motion/README.md)。

```bash
# 固定输入定位回放，不更新模型、不改变实际轨迹。
nice -n 19 python -m experiments.evaluate_real_input_repair \
  --cache training/assets/localization_replay_20261015 \
  --analysis training/assets/real_recording_20261014/analysis \
  --record training/assets/real_recording_20261014/record \
  --hardware models/quality_recovery_20261014/portable/hardware.json \
  --output training/runs/transfer_localization_replay --cpu-cores 0.35
```

## 6. 资源和交接范围

主训练/评估软限 **0.35 核**，辅助 **0.15 核**，合计约半核，单线程、低优先级；单任务建议内存上限 4 GiB。新电脑先串行复评，再按实测余量调整。后台硬限制前缀如下，后接完整 `python -m ...`，并保留应用的 `--cpu-cores 0.35`：

```bash
systemd-run --user --unit=agv-recovery \
  --property=CPUQuota=50% --property=MemoryHigh=3G --property=MemoryMax=4G \
  --property=Nice=19 --working-directory="$PWD" \
  --setenv=PATH="$PATH" --setenv=PYTHONPATH=training:. \
  --setenv=OPENBLAS_NUM_THREADS=1 --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1
# systemctl --user status agv-recovery
# journalctl --user -u agv-recovery
```

| 目录 | 内容 |
|---|---|
| `training/neural_navigation/` | 定位、地图、路由、方向目标、运行时和保护 |
| `training/experiments/` | 训练、诊断、因果审计、闭环评估、测试 |
| `models/`、`reports/` | 模型、配置、优化器、来源、成功及失败证据 |
| `training/assets/localization_replay_20261015/` | 28 条冻结仿真输入，真值仅供监督/评估 |
| `training/assets/real_recording_20261014/` | 1,516 条实车决策、1,911 条反馈、轮速/IMU、map_01 |
| `deployment/ros2_recording_adapter/` | 记录/模型接口及实车集成配置 |
| `reports/handoff_transfer_20261016/` | 本次只读快照及后台状态，明确完整/部分结果 |
| [FILES.json](docs/handoff/FILES.json) | 本次 SHA256 清单 |

未上传完整 `record.zip`、原始 rosbag、相机图像、虚拟环境、缓存和平台 `.so`。精简定位回放已包含，见 [实车清单](training/assets/real_recording_20261014/MANIFEST.json)。分析原始 TF、图像或重新提取其他话题需另外复制原始录包。编译只重建目标平台定位库，不会重新训练。

历史交接：[move README](docs/handoff/MOVE_README.md)、[move 检查](docs/handoff/MOVE_VERIFICATION.json)、[recovery README](docs/handoff/RECOVERY_README.md)、[早期工作目录补丁](docs/handoff/legacy_wip/README.md)。历史“进行中”和绝对路径仅代表当时来源，不应覆盖本次主线。
