# AGV 强化学习导航：换机交接

交接分支：**[handoff2](https://github.com/gcusms/rl_learn_agv/tree/handoff2)**。冻结时间：**2026-10-16 12:49:16，北京时间**。本分支接续 `continue`，保存源码、定位算法、原部署与研究模型、优化器、示范数据、必要地图、精简实车记录，以及成功和失败的评估轨迹。

**当前重点：让模型理解停车原因，并在确有安全运动机会时恢复导航。最新候选已改善一个绕行变慢的任务，但完整验收尚未完成；106 秒停滞仍未解决，原实车模型未替换。** 截止时没有 PPO 在训练，原电脑有一个低资源物理评估进程运行；这里只保存已完成结果，后台进程不会迁移。

精确文件与模型哈希见 [交接快照](reports/handoff2_20261016/SNAPSHOT.json) 和 [文件清单](docs/handoff/FILES.json)。历史报告中的“当前”“运行中”“未上传”均只描述各自生成时间；本页与本次快照优先。

## 1. 新电脑怎么接着做

建议先用 Ubuntu 22.04、Python 3.10、CPU 复现。仓库为私有仓库，新电脑需已有 GitHub 访问权限。

```bash
git clone --depth 1 --branch handoff2 https://github.com/gcusms/rl_learn_agv.git
cd rl_learn_agv
python3 tools/check_transfer.py

sudo apt-get install python3.10-venv g++ libeigen3-dev
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r training/requirements.txt
export PYTHONPATH=training:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
nice -n 19 python tools/check_handoff_models.py
```

第一项检查只需 Python 标准库，检查文件、模型和完成轨迹；第二项重新加载三份最近检查点、Adam、随机状态，检查观测、动作掩码、原始数据和导出动作一致性。它们都不训练、不运行物理导航。

已在交接目录完成三模型检查：示范复算分别为 2,421/2,421、对照 2,354/2,355、最新 2,505/2,505，所有必要停车保留，共检查 690 条导出动作。对照那一条不匹配是原有记录，不应改成通过。见 [MODEL_CHECK.json](reports/handoff2_20261016/MODEL_CHECK.json)。

平台 `.so` 不随 Git 迁移；完整运行前，从已上传 C++ 源码重建新电脑的定位包：

```bash
python tools/prepare_native_bundle.py \
  --source models/stop_feedback_20261015/recovery_followup_bc400_s231016/bc_portable \
  --output training/runs/handoff2_native
```

输出目录必须不存在；此命令生成独立包与 manifest，不改归档权重。编译是适配硬件架构，不是重新训练。该源码已在本机重新编译，并用录制的一帧数据验证了最新模型 439 维运行时接口，见 [NATIVE_SMOKE.json](reports/handoff2_20261016/NATIVE_SMOKE.json)。这不代表在 Jetson 上已验收。Jetson 需按 JetPack 安装对应 PyTorch。

仓库约 1.3 GB；建议另留至少 8 GB 安装环境及缓存。先单线程、单物理任务、Nice19；评估命令的软预算保持 **0.35 核**。可在命令前加 `systemd-run --user --scope -p CPUQuota=50% -p MemoryMax=4G` 提供硬限制。源电脑资源不足时，不要并行开评估或重新训练。

## 2. 模型选哪一个

| 模型 | 文件夹（位于 `models/stop_feedback_20261015/`） | 已完成结果 | 尚未通过 |
|---|---|---|---|
| 原实车版 | `models/quality_recovery_20261014/policy_2000.zip`（相对仓库根目录） | 保留实车部署原文件 | 不能拿不同任务集历史成绩直接排名 |
| 诊断父版 | `early_transfer_bc800_s231015/` | 常规 18/18；额外 8/12 | 额外 4 次碰撞，95/106 秒接管超时 |
| actual95 父候选 | `actual95_transfer_bc400_s231016/` | 常规 **18/18**；90/95 秒恢复到达；额外等待 2.4 秒后到达；原恢复 2/2、方向探针 2/2 | 额外仍为 8/12；该组总时间增加 6.05%，效率门槛失败；106 秒仍超时 |
| **最新研究候选** | **`recovery_followup_bc400_s231016/`** | 冻结时 **5/5 安全到达**：4 项正常任务＋1 项补训任务；该补训任务由 48.0 秒恢复至 38.6 秒 | 尚未完成预定 8 项，以及完整 18 项／额外 12 项回归；不是部署升级 |

最新模型：`models/stop_feedback_20261015/recovery_followup_bc400_s231016/policy_bc.zip`。

SHA256：`dd29ed24e25191af8265d832dd0edda072160d1204f4a18b280dbf265eb3e212`。

本轮从 actual95 候选派生，加入一条完整安全旧轨迹的 84 个决策，追加 **400 次监督更新，新增 PPO 0 步**。原 2,421 条输入、掩码及标签逐字节保留，57 条停车标签保留；共 2,505 条标签拟合正确。未更新基础 actor、localizer、router、方向停靠，也未修改车体、动作或保护余量。

同一任务 `localization_nav_33`：诊断父版 38.6 秒／5.602 米，actual95 父候选 48.0 秒／5.659 米，新候选 **38.6 秒／5.602 米**，比直接父版减少 9.4 秒。此任务参与了补训，不能称为陌生地图泛化；标签正确也不能替代闭环成功。

模型目录均包含检查点、`bc_portable/` 导出包、示范、优化器／随机状态、配置和来源。恢复检查点使用 `experiments.stop_recovery_policy.StopRecoveryPPO.load(..., device='cpu')`；配套包按 manifest 加载。已有两组独立 2k PPO 的 `policy.zip` 也保留；监督更新、评估和初始化帧均不算 PPO 步数。

## 3. 换机后先验证，接着训练的条件

冻结记录：新候选预定 8 项已完成 5 项，见 [已完成结果](reports/stop_feedback_20261015/recovery_followup_validation_20261016/RESULTS.json)。剩余为 `motion_profile_015`、95 秒接管和 90 秒接管；原机截止后的结果不会自动进入此分支。先核实原机是否已有完成结果，避免重复算成新实验。

在第 1 节环境中，先补一项正常导航：

```bash
nice -n 19 python -m experiments.evaluate_docking_navigation \
  --candidate training/runs/handoff2_native \
  --candidate-name recovery_followup_bc400 \
  --cases reports/handoff2_20261016/pending_gate_cases.json \
  --output training/runs/handoff2_gate --cpu-cores 0.35

python tools/check_handoff_gate.py \
  --normal-result training/runs/handoff2_gate/RESULTS.json
```

门槛通过后，再独立检查 95/90 秒接管；与上一条物理任务串行：

```bash
nice -n 19 python -m experiments.evaluate_feedback_recovery \
  --bundle training/runs/handoff2_native \
  --specifications reports/handoff2_20261016/pending_recovery_specifications.json \
  --label recovery_followup_bc400 \
  --output training/runs/handoff2_recovery --cpu-cores 0.35
```

之后补齐常规剩余 13 项、额外条件剩余 11 项、原恢复历史 2 项和方向探针 2 项，并保留 2.4 秒等待对照。清单和配对依据见 [NEXT_STEPS.json](reports/handoff2_20261016/NEXT_STEPS.json)。每次用新输出目录，按 `case_id` 合并，不重复计数；成功、碰撞、超时、假到达、路径、耗时分别统计。

106 秒问题不能靠“动作合法”就宣称解决：传感器／记忆回放、三组物理试验及一次阶段转向试验已归档，均未得到完整安全恢复，不作为正示范。当前雷达和历史记忆都可能否决动作，恢复分支实际已启用。需要验证更早避免进入余量过小状态，或找到相同约束下完整安全恢复的证据；不能清空盲区记忆、缩小车体、降低 4 cm 余量或强制前进换取成功率。

只有安全与效率回归通过，再设计小规模、单因素 PPO。入口是 `python -m experiments.train_prefix_recovery --help`。已保存网络／Adam／RNG，但当前 CLI 不支持任意最终检查点连同半回合物理状态、扫描历史无缝恢复；`--reference-bc` 也不是通用续训开关。不要在历史 `reports/**/run.py` 原目录重跑，它们固定输出路径和原机服务依赖。

## 4. 代码、定位与数据位置

| 路径 | 内容 |
|---|---|
| `training/neural_navigation/` | 地图、定位、路由、雷达与安全、停车反馈、方向目标运行时 |
| `training/experiments/` | BC／PPO、回放诊断、固定任务评估、配对门槛 |
| `docs/localization/README.md` | 定位算法优化、编译原因、数据及验证办法 |
| `models/localization_mapping_20261015/ray_consistent_candidate/portable/` | 独立定位候选，未混入本次恢复对照 |
| `models/docking_selected_5000_20261015/` | 位置＋目标朝向停靠模型 |
| `training/assets/real_recording_20261014/` | map_01、1,516 条实车决策、1,911 条反馈及精简轮速／IMU |
| `training/assets/localization_replay_20261015/` | 冻结定位回放；真值只供监督／评价 |
| `reports/stop_feedback_20261015/` | 正负实验、来源、训练与完整评估轨迹 |
| `deployment/ros2_recording_adapter/` | ROS 2 录制与接入；导航控制不依赖 Nav2 行为树 |

定位改善主要来自里程计传播、静态支持筛选、弱方向约束、时序融合和自由射线检查，不是本轮恢复 BC／PPO 更新的定位网络。独立 22 条、2,855 帧回放中，时序父版／射线候选位置 P95 为 18.51／18.04 cm，CPU P95 为 7.88／14.59 ms；精度小幅改善、耗时增加。实车没有独立绝对位姿真值，未证明超过 AMCL。已知地图、边建图边导航、开放稀疏区、搬车重定位仍需分开验收。

当前模型车体约 39.39×32.95 cm，4 cm 安全余量；220 束、约 219.44° 视野，保留后方盲区；12 个离散 `(vx,wz)` 动作，无横移、无倒车，最高 0.20 m/s、0.50 rad/s。恢复训练 436 维、带朝向运行时 439 维、原部署 300 维，必须配套权重和观测合同。支持位置＋朝向目标及原地转向停靠；新候选仍需停靠回归。

原始 `record.zip`、rosbag／相机、虚拟环境、缓存和平台 `.so` 未上传。本轮所需精简实车数据已包含；重新提取完整话题时需另行复制原机 `/home/ubuntu/rl_learn/record.zip`，哈希见对应数据 MANIFEST。

上一份交接保留在 [continue README](docs/handoff/CONTINUE_README.md)。当前已完成、待验证和原机进程状态以本次快照为准；不自动替换实车模型。
