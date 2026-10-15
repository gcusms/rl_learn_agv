# AGV 导航：当前进度与换机指南

本次交接分支为 **[`dev`](https://github.com/gcusms/rl_learn_agv/tree/dev)**，物理评估固定截至 **2026-10-16 07:29:56（北京时间）**，另补充 **07:30:12 完成的离线监督候选**。包含当前代码、定位算法、地图、原部署模型、研究候选、PPO 权重与优化器、实际采样、训练示范、精简实车回放，以及成功和失败记录。

**原实车部署模型保留；新研究模型尚未通过部署验收。** 本次新增两组各 2,000 步的 PPO，训练已完成，独立导航评估尚未完成。两次训练不能合称单个模型 4,000 步，训练到达比例不能当作独立成功率。

精确状态见 [SNAPSHOT.json](reports/handoff_dev_20261016/SNAPSHOT.json)，文件哈希见 [FILES.json](docs/handoff/FILES.json)，迁移验证见 [VERIFICATION.json](docs/handoff/VERIFICATION.json)。上一份 `work` 分支保持原快照，详细历史说明归档为 [WORK_README.md](docs/handoff/WORK_README.md)。

## 1. 当前进度与模型选择

目标是让模型区分停车原因，在有安全可行动作时合理恢复，同时保留必要停车、避障、路径决策和位置＋方向停靠。当前冻结基础 actor、localizer、router 和终端方向停靠部分，只训练有启用条件的恢复 residual；因此，追加恢复 PPO 不会直接训练定位器或路由器。

| 模型 / 检查 | 已完成结果 | 判断 |
|---|---|---|
| 原实车质量恢复 2k | 历史开发集 46/48 | 保留作部署基线；与下述集合不同 |
| 直接父 BC：2,030 标签、800 次监督拟合 | 完整 18 项：15 到达、3 碰撞；原 14 项成功保留 | 配对参考，追加 PPO 为 0 |
| 2,089 标签 BC、800 次监督拟合 | 完整 18 项：16 到达、1 碰撞、1 超时；原 14 项 14/14 | 当前参考候选；共同成功 15 项总路程 −0.28%、总时间 −0.05%，仍未全部成功 |
| 独立旧数据 8k PPO | 完整 18 项：16 到达、2 碰撞；原 14 项只保留 13 项 | `motion_profile_015` 退化为碰撞，不选作部署升级 |
| **新增对照 PPO：2,000 步** | 训练及全部实际采样审计完成；训练中 8 个完整回合：7 到达、1 超时 | 独立导航评估未完成，不替换模型 |
| **新增示范保留 PPO：2,000 步** | 训练及审计完成；训练中 11 个完整回合：9 到达、1 碰撞、1 超时 | 保留损失确实参与更新，仍出现碰撞；独立评估未完成 |
| 新增 2,355 标签 BC、800 次监督拟合 | 离线拟合完成：2,349/2,355，新增 266 条全对，原 2,089 条有 6 条不一致，55 条停车全保留 | 未满足预设全标签保留条件；不准入物理初测，不替换参考 BC；追加 PPO 为 0 |

两组新 PPO 使用相同 2,089 标签 BC、网络、优化器、种子、课程和原保护，唯一主要变量为示范保留系数 `0` / `0.05`，长等待探索损失均关闭。首 250 个实际 rollout 的观测、动作、掩码、旧概率和奖励逐字节一致，之后策略更新造成轨迹分化。

同一份 2,089 条合法示范的离线动作一致性：BC **2,089/2,089**，对照 PPO **1,830/2,089**，保留 PPO **1,851/2,089**；双方均保留 **37/37** 合理停车标签。不同动作不自动等于错误，示范拟合率也不等于导航成功。两组在各自实际训练状态上均未出现“原 BC 贪心停止 → 新 PPO 贪心运动”，尚不能宣称已解决长等待。

证据：[训练保留审计](reports/stop_feedback_20261015/retention_pair_20261016/COMPLETED_TRAINING_AUDIT.json)、[实验计划](reports/stop_feedback_20261015/retention_pair_20261016/PLAN.json)、[对照实际采样](reports/stop_feedback_20261015/w11_control2000_audit_20261016/RESULTS.json)、[约束组实际采样](reports/stop_feedback_20261015/w11_retained2000_audit_20261016/RESULTS.json)、[约束组碰撞片段](reports/stop_feedback_20261015/retention_pair_20261016/RETAINED_COLLISION_INTERIM_1250.json)。碰撞前已只允许停车，具体接触原因仍需物理复现，不能仅凭池化雷达归因。

参考 BC 原两个恢复前缀 2/2 到达；W11 的 90/95/106 秒历史恢复为 1 到达、2 超时。旧 8k PPO 在另 12 条同地图条件上为 10 到达、1 碰撞、1 超时，父 BC 为 8 到达、4 碰撞；这不能抵消原成功任务的退化。训练用过的历史与重复路线不算未见泛化。

## 2. 截止时尚未完成的工作

只冻结完整结束的回合和对应轨迹，见 [本次评估目录](reports/handoff_dev_20261016/validation/)。

| 任务 | 本次截止状态 |
|---|---|
| 两组新 PPO，各 18 个完整起点＋2 个原前缀＋3 个 W11 历史＋12 条额外条件 | 均已训练；对照完整起点评估完成 2/18，均到达；其余尚待执行或完成 |
| 参考 BC 与父 BC 的额外条件配对 | 完成 20/24 行；不是已完成的配对结论 |
| 2,355 标签离线监督候选 | 物理评估截止时仍在拟合；后于 07:30:12 完成，作为明确标记的补充产物上传，未通过全标签保留条件 |
| 近障原地转向选项 | 已做几何诊断，完整物理专项排队；改变动作许可，未合入标准模型 |
| 新数据候选三项物理初测 | 无导航成绩；排队入口会按预设的全标签条件拒绝本候选，不自动续训 |

新增 266 条示范来自旧 PPO 已完整安全到达的两条开发路线，保留原 2,089 条，总数 2,355，其中新运动 248 条、新停车 18 条。00/01 路线因此进入训练，不再称为独立 holdout。八条额外同地图条件已在新结果产生前固定，但尚未运行。详见 [数据与实验说明](reports/stop_feedback_20261015/early_decision_transfer_20261016/README.md)。

**Git 不迁移进程。** 原机 systemd 服务、PID、排队关系及截止后的结果不会自动进入本分支。`run.py`、`after_*.py` 中的原机服务等待只供审计；换机使用下面的相对路径命令，不要等待旧 PID。当前入口不支持精确恢复半回合的物理、雷达历史和全部内部状态。

## 3. 新电脑安装与检查

先按 Ubuntu 22.04、Python 3.10、CPU 环境复现，不复制旧虚拟环境。以下在同一终端执行。

```bash
git clone --depth 1 --branch dev https://github.com/gcusms/rl_learn_agv.git
cd rl_learn_agv
sudo apt-get install python3.10-venv g++ libeigen3-dev
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r training/requirements.txt
export PYTHONPATH=training:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# 输出目录必须不存在；不启动训练，不发送电机指令。
nice -n 19 python tools/check_handoff.py --output training/runs/dev_check
```

建议安装至少预留 8 GB，下载缓存另计。Jetson 使用与 JetPack 匹配的 PyTorch，不直接安装上述桌面 CPU 包。`training/requirements.lock.txt` 为原机版本记录；CUDA 不是当前检查的前提。

检查核对文件哈希、模型合同、13 份反馈模型、两组新 PPO 的最终权重与优化器、实际样本数、方向目标和实现检查。具体通过数及原生检查范围见 VERIFICATION；它不等于闭环或实车验收。

平台 `.so` 不上传。闭环前用 `tools/prepare_native_bundle.py` 从同包源码重新编译定位内核；编译不是重新训练。`check_handoff.py --build-native` 可额外重建全部原生包，耗时更长。

## 4. 模型位置及版本区别

路径相对仓库根目录。权重、manifest、特征、保护参数和定位内核须配套。

| 路径 | 用途 |
|---|---|
| `models/quality_recovery_20261014/policy_2000.zip` | 原实车部署模型，未替换 |
| `models/stop_feedback_20261015/w11_recovery_bc800_s231015/` | 2,089 标签参考 BC：`policy_bc.zip`、`bc_portable/`、优化器、RNG、示范、配置 |
| `models/stop_feedback_20261015/w11_control2000_s231015/` | 新 2k PPO 对照：最终 `policy.zip`、`portable/`、优化器、全部 rollout 与初始化 |
| `models/stop_feedback_20261015/w11_retained2000_s231015/` | 新 2k PPO，保留系数 0.05，完整产物同上 |
| `models/stop_feedback_20261015/early_transfer_bc800_s231015/` | 新 2,355 标签 BC 完整产物；6 条旧标签未保留，未准入物理初测，供分析和继续研究 |
| `models/stop_feedback_20261015/proactive_augmented_ppo_8000_s231015/` | 旧独立 8k PPO，含全部 8,000 条实际样本及源码快照 |
| `models/stop_feedback_20261015/proactive_augmented_bc_800_s231015/` | 2,030 标签直接父 BC |
| `models/stop_feedback_20261015/baseline/portable/` | 303 维训练父包 |
| `models/localization_mapping_20261015/ray_consistent_candidate/portable/` | 独立定位算法候选，未合入本轮恢复对照 |
| `models/docking_selected_5000_20261015/` | 历史位置＋方向停靠模型 |
| `reports/stop_feedback_20261015/early_decision_transfer_20261016/` | 2,355 条数据、来源审核、拟合与后续评估方案 |

原部署 SHA256：`7a3dad60f76060daf6b311b4bc6db921d3925ef0cdcf11683fe99e282173de11`。
参考 BC SHA256：`76bd054747c59d5cb1c625e61c9411cce10744346ea97e4b1308433d236fdbc8`。
新 2k 对照 SHA256：`7a8b576c1a29e357508cac3f473c2524bc2688c46f6f6f09be5a38d50fe88cb5`。
新 2k 保留约束 SHA256：`9012d24c36f7ec2a54455dbb4010a8755a74a348a0881e58e0222e21733bf220`。

`policy_bc.zip` 是监督初始化，追加 PPO 为 0；最终 `policy.zip` 才包含全部 PPO 更新。`policy_step_2000.zip` 在最后一次更新前保存，不能当作最终模型。恢复 ZIP 用 `experiments.stop_recovery_policy.StopRecoveryPPO.load`，运行时由 manifest 的 `required_runtime` 解析。

两组新 PPO 用时约 25.0 / 26.4 分钟，软预算约 0.35 核，峰值 RSS 约 2.35 / 2.39 GiB。初始化分别 1,060 / 1,160 帧另计，不混入各自的 2,000 个 PPO 样本。

## 5. 换机后先补齐固定评估

下面以约束组为例。验证对照组时把 `model` 改为 `w11_control2000_s231015`；两组串行，每个输出目录必须不存在。

```bash
model=w11_retained2000_s231015
python tools/prepare_native_bundle.py \
  --source models/stop_feedback_20261015/$model/portable \
  --output training/runs/dev_$model

nice -n 19 python -m experiments.evaluate_docking_navigation \
  --candidate training/runs/dev_$model --candidate-name $model \
  --cases reports/stop_feedback_20261015/w11_success_augmented_protocol_20261016/gate_cases.json \
  --output training/runs/dev_${model}_navigation --cpu-cores 0.35

nice -n 19 python -m experiments.evaluate_feedback_recovery \
  --bundle training/runs/dev_$model --label $model \
  --specifications reports/stop_feedback_20261015/frozen_recovery_cases.json \
  --output training/runs/dev_${model}_prefixes --cpu-cores 0.35

nice -n 19 python -m experiments.evaluate_feedback_recovery \
  --bundle training/runs/dev_$model --label $model \
  --specifications reports/stop_feedback_20261015/proactive_protocol_20261016/w11_prefixes.json \
  --output training/runs/dev_${model}_w11 --cpu-cores 0.35

nice -n 19 python -m experiments.evaluate_docking_navigation \
  --candidate training/runs/dev_$model --candidate-name $model \
  --cases reports/stop_feedback_20261015/w11_success_augmented_protocol_20261016/fresh_cases.json \
  --output training/runs/dev_${model}_fresh --cpu-cores 0.35
```

每组共 35 行，不是 35 张未见地图。比较原 14 项成功、六项方向停靠、合理停车、碰撞、假到达、实际等待、时间和路程，不只按奖励或到达总数选模型。同一任务重跑不能重复算成独立成功。参考 BC 的完整复现和实车同输入回放命令在 [上一份详细指南](docs/handoff/WORK_README.md)。

## 6. 后续训练入口与限制

先完成固定评估，决定保留、调整或放弃约束，不自动增步数。入口为 `python -m experiments.train_prefix_recovery`，完整 2k 配置在 [计划](reports/stop_feedback_20261015/retention_pair_20261016/PLAN.json)，完整相对路径命令在 [上一份指南第 5 节](docs/handoff/WORK_README.md)。

网络、优化器、示范、RNG、课程配置和样本均保存，可用于分析及新实验初始化；**没有半回合物理状态的精确续跑**。新实验用新目录和独立步数，不能把新步数直接接到旧 8k。观测或动作变更时同步导出和 manifest。

默认关闭的长等待探索损失尚未进行物理 PPO，不要与示范保留同时修改后归因。2,355 标签 BC 已以补充产物保存，未满足保留条件；先分析丢失的 6 条旧标签，不直接增加更新次数。新拟合参考 [方案](reports/stop_feedback_20261015/early_decision_transfer_20261016/BC_PLAN.json)，使用独立输出目录。离线监督不产生新物理 PPO 步数。原机排队脚本不能直接迁移运行。

## 7. 定位、方向停靠与实车条件

定位改善来自里程计传播、静态支持筛选、弱方向约束、时序融合与自由射线矛盾检查，并非本轮 PPO 重训了定位网络。独立时序父版 / 射线候选在 22 条冻结仿真轨迹、2,855 帧同输入对照的位置 P95 为 18.51 / 18.04 cm，航向 P95 为 1.526 / 1.368°，定位 CPU P95 为 7.88 / 14.59 ms；收益有限、耗时增加，尚未合入恢复对照。

源码、编译与验证见 [定位 README](docs/localization/README.md)。实车录包没有独立绝对位姿真值，不能据此宣称超过 AMCL、长期无漂移、搬车重定位或达到 Orin Nano 延迟标准。已知地图、边导航边建图、开放稀疏场景仍需分别验证。

底盘合同：车体约 39.39×32.95 cm、4 cm 余量；12 个离散动作，无横移，当前动作表无倒车；前进最高 0.20 m/s、转向 0.50 rad/s。220 束原始雷达、约 219.44°，保留后方盲区。恢复训练 436 维、方向运行时 439 维，区别于原部署 300 维。位置＋朝向目标已接入，同位置不同朝向可转向，六项方向停靠保留在原回归集。

训练主要使用带噪声运动反馈；若实车仍用指令积分里程计，不能认为对应误差已验收。真实 6%～12% 无效束、连续缺失、帧时间和延迟、新障碍未入静态图、低层指令超时停车仍需验证。不能通过缩小车体、取消必要停车或放宽未知区保护获得表面成绩。

## 8. 目录、存储与资源

| 目录 | 内容 |
|---|---|
| `training/neural_navigation/` | 地图、定位、路由、方向目标、反馈特征及保护 |
| `training/experiments/` | 快速二维训练、监督初始化、恢复 PPO、审计、评估 |
| `models/`、`reports/stop_feedback_20261015/` | 模型、优化器、配置、源码快照和成功/失败证据 |
| `reports/handoff_dev_20261016/` | 本次固定截止、完成及部分回合、原机状态 |
| `training/assets/localization_replay_20261015/` | 28 条冻结仿真输入，真值仅供监督及评价 |
| `training/assets/real_recording_20261014/` | 1,516 条实车决策、1,911 条反馈、轮速/IMU、map_01 及清单 |
| `deployment/ros2_recording_adapter/` | ROS 2 记录与集成接口 |
| `docs/handoff/`、`tools/` | 迁移说明、哈希、检查与原生重建工具 |

新机先单线程、`nice -n 19`、单任务软限 **0.35 核**串行执行，预留至少 4 GiB 内存余量，不要同时启动所有命令。原机主任务 0.35 核、一个辅助物理验证 0.15 核、离线拟合 0.05 核；新机速度以实测为准。

完整 `record.zip`、原始 rosbag、相机图像、虚拟环境、缓存和平台 `.so` 未上传。精简数据可用于已说明的回放；需要原始话题时另复制 `/home/ubuntu/rl_learn/record.zip` 和源录包，参见 [清单](training/assets/real_recording_20261014/MANIFEST.json)。来源中的旧绝对路径保留作审计，运行使用本文相对路径。
