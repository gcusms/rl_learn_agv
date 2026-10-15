# AGV 导航：当前进度与换机指南

本次交接分支：**[`work`](https://github.com/gcusms/rl_learn_agv/tree/work)**。实验结果固定截至 **2026-10-16 06:12:03（北京时间）**。

包含当前代码、原部署模型、最新恢复候选、已完成的 8k PPO 及优化器、训练示范、地图、精简实车数据、定位算法、完整成功/失败记录和继续开发命令，另含最新等待原因与探索不足诊断、默认关闭的训练约束实验。**目前仍是研究候选，未完成新模型的部署验收。** 原实车模型保留。

精确状态见 [SNAPSHOT.json](reports/handoff_work_20261016/SNAPSHOT.json)，文件哈希见 [FILES.json](docs/handoff/FILES.json)，迁移验证见 [VERIFICATION.json](docs/handoff/VERIFICATION.json)。`now` 分支是 05:32 的固定快照，说明保存在 [NOW_README.md](docs/handoff/NOW_README.md)，旧哈希与验证结果单独归档。更早的 `ship` 说明保存在 [SHIP_README.md](docs/handoff/SHIP_README.md)。新电脑优先使用本 `work` 分支。

## 1. 当前做到哪里

当前目标：让模型区分停车原因，在有安全可行动作时恢复前进，同时保留必要停车、避障、路径决策和位置＋方向停靠。结构是冻结基础 actor、localizer 和 router，训练有启用条件的恢复分支。继续恢复训练不能直接改变定位网络或路由网络。

| 模型 / 检查 | 当前结果 | 结论 |
|---|---|---|
| 原实车质量恢复 2k | 历史开发集 46/48 | 保留作部署基线；与下述 18 项不是同一集合 |
| 直接父模型：2,030 标签、800 次示范拟合 | 18 项：15 到达、3 碰撞；原 14 项成功全部保留 | 直接配对参考，追加 PPO 为 0 |
| **最新候选：2,089 标签、800 次示范拟合** | **18 项：16 到达、1 碰撞、1 超时；原 14 项仍 14/14** | 共同成功的 15 项总路程 −0.28%，总时间 −0.05%；安全和效率保留检查通过，全部成功门槛未通过 |
| 最新候选在 W11 的 90 秒历史后接管 | 接管后 20.4 秒到达，误差 9.39 cm，0 碰撞 | 已学习的历史，不能当未见场景泛化；该候选追加 PPO 为 **0** |
| **独立旧数据 PPO：8,000 步** | **已训练完成**；18 项：16 到达、2 碰撞；原 14 项只保留 13 项 | 旧成功任务 `motion_profile_015` 退化为碰撞；共同成功任务总耗时 +10.27%，不选作部署升级 |
| 该 8k PPO 的相关历史恢复 | 原两个前缀 2/2 到达；W11 90/95/106 秒接管均超时 | 从完整起点到达与中途恢复不是同一种能力 |
| 新增 PPO 示范保留约束 | 23 项相关实现检查通过，单个已录 rollout 更新对照通过 | 默认关闭，尚未用于新的物理 PPO；没有产生“约束版训练完成模型” |

最新 BC 候选的两个困难失败仍保留：`recovery_holdout_00` 超时，其中 34.6 秒存在合法运动却实际停车，92.8 秒只允许停止；`recovery_holdout_01` 碰撞。不能用强制最低速度、缩小车体或削弱保护掩盖失败。两版虽同为 16/18，成功任务组成不同，不能只按总成功率选模型。

新候选全部 2,089 个示范标签和导出动作一致，原 2,030 标签及 37 个合理停车标签保留；这属于训练数据拟合，不等于所有新场景都正确。W11 首 90 秒用于物理历史初始化，之后由模型和原保护接管，没有定时教师。

在本次截止时，8k PPO 的额外条件配对已完成 24/24 行：父 BC 为 8/12 到达、4 次碰撞；PPO 为 10/12 到达、1 次碰撞、1 次超时。这与原 14 项中出现退化同时成立，不能只挑增益。新 BC 原两个恢复前缀为 2/2 到达；W11 相关历史完成 2/3（90 秒到达、95 秒超时），106 秒仍在验证，直接父模型 90 秒及额外 12 条条件配对尚未完成。完整与部分结果单独冻结在 [交接评估目录](reports/handoff_work_20261016/validation/)，部分结果不算完整验收。

新等待诊断：8,000 条实际 PPO 样本中，恢复分支可训练、已等待至少 2 秒且仅允许停止或动作 9 的 160 条样本全部选择停止。另做的两个固定转向窗口诊断均超时，只有最初 4 帧允许请求动作，之后被原掩码拒绝；这说明延长窗口无效，不能说明所有恢复办法都不可行。诊断不属于学习模型的成功成绩。详见 [剩余问题报告](reports/stop_feedback_20261015/remaining_recovery_analysis_20261016/REPORT.md)。

新增可选的长等待探索损失已实现，默认关闭，独立于示范保留损失；**两种新约束均未用于新的物理 PPO**。新代码的实现检查与闭环收益分别记录，下一轮先做单项对照。实现兼容性及零激活的已录批次对照见 [实验报告](reports/stop_feedback_20261015/long_wait_exploration_protocol_20261016/REPORT.md)。

主要证据：[新 BC 完整配对汇总](reports/stop_feedback_20261015/w11_recovery_bc800_navigation_20261016/SUMMARY.json)、[8k PPO 完整配对汇总](reports/stop_feedback_20261015/proactive_augmented_ppo8000_navigation_20261016/SUMMARY.json)、[8k 采样审计](reports/stop_feedback_20261015/proactive_augmented_ppo8000_audit_20261016/RESULTS.json)、[遗忘诊断](reports/stop_feedback_20261015/ppo8000_retention_diagnostic_20261016/RESULTS.json)、[示范保留方案](reports/stop_feedback_20261015/ppo_retention_protocol_20261016/REPORT.md)。

## 2. 新电脑安装和完整性检查

先按 Ubuntu 22.04、Python 3.10、CPU 环境复现；不用复制旧虚拟环境。下列命令在同一个终端执行。

```bash
git clone --depth 1 --branch work https://github.com/gcusms/rl_learn_agv.git
cd rl_learn_agv
sudo apt-get install python3.10-venv g++ libeigen3-dev
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r training/requirements.txt
export PYTHONPATH=training:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# 输出目录须不存在；核对文件、模型、实际样本、输入合同和实现。
# 此命令不启动训练，也不向电机发送指令。
nice -n 19 python tools/check_handoff.py --output training/runs/work_check
```

建议至少预留 8 GB 安装空间，下载缓存另计。Jetson 使用与 JetPack 匹配的 PyTorch，不要直接安装上述桌面 CPU 包。`training/requirements.lock.txt` 为原机版本记录。检查不依赖 ROS 2、CUDA 或显示器，不能代替闭环导航验收。 本分支 **127 项实现检查全部通过、无跳过**，包含新增 9 项长等待损失检查；10 份反馈模型可加载，最终 8k 优化器和 8,000 条实际样本核对通过。当前 439 维原生包沿用 `now` 已重建的参考，原生一致性检查通过；本轮未重新编译全部包，具体范围见 VERIFICATION.json。

闭环运行前按下一节重建原生定位库；`.so` 与 CPU 架构、系统相关，源码已上传，编译不等于训练。`check_handoff.py --build-native` 可额外检查全部原生包，耗时更长。

## 3. 模型位置与使用区别

路径均相对仓库根目录。模型、manifest、输入特征、定位内核与运行时要配套，不能只替换 actor 文件。

| 路径 | 用途 |
|---|---|
| `models/quality_recovery_20261014/policy_2000.zip` | **原实车部署模型**，未替换 |
| `models/stop_feedback_20261015/w11_recovery_bc800_s231015/` | **最新恢复研究候选**：`policy_bc.zip`、`bc_portable/`、优化器、2,089 条示范和配置；800 次监督拟合、追加 PPO 0 步 |
| `models/stop_feedback_20261015/proactive_augmented_bc_800_s231015/` | 直接父模型，2,030 标签、800 次拟合 |
| `models/stop_feedback_20261015/proactive_augmented_ppo_8000_s231015/` | **已完成的独立 8k PPO**：最终 `policy.zip`、`portable/`、优化器、RNG、全部 8,000 个实际样本、4,060 帧初始化及源码快照 |
| `models/stop_feedback_20261015/proactive_ppo_ship_20261016/` | 上次交接的历史 7k 检查点，权重更新到 6,750 步；不是最新最终 PPO |
| `models/stop_feedback_20261015/baseline/portable/` | 303 维训练父包 |
| `models/stop_feedback_20261015/ppo_4000_s221015/initial_portable/` | 439 维冻结参考，区别于上面的父包 |
| `models/localization_mapping_20261015/ray_consistent_candidate/portable/` | 独立定位算法候选，尚未合入本轮恢复实验 |
| `models/docking_selected_5000_20261015/` | 历史位置＋方向停靠模型 |

SHA256：

- 新 BC `policy_bc.zip`：`76bd054747c59d5cb1c625e61c9411cce10744346ea97e4b1308433d236fdbc8`。
- 最终 8k PPO `policy.zip`：`855fae8bbc391c2c729f2692d88910bc9b50f5e15c697400809ee838728a2c6d`。
- 原部署 `policy_2000.zip`：`7a3dad60f76060daf6b311b4bc6db921d3925ef0cdcf11683fe99e282173de11`。

`policy_bc.zip` 是示范初始化，最终 `policy.zip` 才是 PPO 更新后的结果；`policy_step_8000.zip` 在最后一次更新之前保存，不能当作最终权重。恢复 ZIP 用 `experiments.stop_recovery_policy.StopRecoveryPPO.load`，运行时由 manifest 的 `required_runtime` 解析，见 [迁移检查入口](tools/check_handoff.py)。

本轮 8k PPO 用时 **5,567 秒，约 92.8 分钟**，平均 0.35 核、峰值 RSS 约 2.46 GiB。PPO 只计 8,000 个真实采样步；示范拟合和物理初始化不混入训练步数。

## 4. 新电脑继续闭环验证

以下命令在仓库根目录执行，使用第 2 节的环境变量。每个输出目录须不存在，同类任务串行运行。

```bash
python tools/prepare_native_bundle.py \
  --source models/stop_feedback_20261015/w11_recovery_bc800_s231015/bc_portable \
  --output training/runs/work_candidate
python tools/prepare_native_bundle.py \
  --source models/stop_feedback_20261015/proactive_augmented_bc_800_s231015/bc_portable \
  --output training/runs/work_parent

# 已完成 18 项的重现；包含原成功任务、困难条件、6 项方向停靠。
nice -n 19 python -m experiments.evaluate_docking_navigation \
  --candidate training/runs/work_candidate --candidate-name w11_bc800 \
  --baseline training/runs/work_parent --baseline-name parent_bc800 \
  --cases reports/stop_feedback_20261015/w11_success_augmented_protocol_20261016/gate_cases.json \
  --output training/runs/work_gate --cpu-cores 0.35

# 原前缀已完成，可重现；优先补全相关历史和额外条件配对。
nice -n 19 python -m experiments.evaluate_feedback_recovery \
  --bundle training/runs/work_candidate \
  --specifications reports/stop_feedback_20261015/frozen_recovery_cases.json \
  --output training/runs/work_prefixes --label w11_bc800 --cpu-cores 0.35
nice -n 19 python -m experiments.evaluate_feedback_recovery \
  --bundle training/runs/work_candidate \
  --specifications reports/stop_feedback_20261015/proactive_protocol_20261016/w11_prefixes.json \
  --output training/runs/work_w11 --label w11_bc800 --cpu-cores 0.35
nice -n 19 python -m experiments.evaluate_feedback_recovery \
  --bundle training/runs/work_parent \
  --specifications reports/stop_feedback_20261015/w11_success_augmented_protocol_20261016/w11_t90_evaluation.json \
  --output training/runs/work_parent90 --label parent_bc800 --cpu-cores 0.35
nice -n 19 python -m experiments.evaluate_docking_navigation \
  --candidate training/runs/work_candidate --candidate-name w11_bc800 \
  --baseline training/runs/work_parent --baseline-name parent_bc800 \
  --cases reports/stop_feedback_20261015/w11_success_augmented_protocol_20261016/fresh_cases.json \
  --output training/runs/work_fresh --cpu-cores 0.35
```

额外条件为已有地图上的 12 条冻结任务，双方共 24 行，不是全新地图测试。90 秒历史现用于训练；同历史复跑属于重现，不重复统计为独立成功。上述评估使用相同雷达、动作和保护约束。

原机后台任务的完成及运行状态、调用关系记录在 SNAPSHOT 中；主 8k 流程和两个晚期诊断已经结束，辅助验证及其排队任务仍在运行。**进程不会随 Git 迁移**，换电脑后使用本 README 的直接命令；原机 `run.py`、`after_preservation.py` 含机器路径和等待关系。截止后的结果不会自动进入本分支。

## 5. 下一轮训练方案和入口

推荐顺序：**补全验证 → 利用已保存的等待与探索诊断 → 确认可行恢复轨迹或提前决策 → 固定同一初始化做单项约束对照 → 完整闭环回归**。不能因为 8k 总成功率提高就直接继续加步数。

已新增训练期可选损失：`PPO 损失 + 系数 × 合法示范平衡交叉熵`，默认系数 0，建议首个对照 0.05、每批 128 条示范，停车与运动各半。只有当前批次存在可训练恢复样本才更新；示范不替换 PPO 的采样动作、旧概率、掩码、奖励或执行历史，不增加物理步数，推理时没有示范教师。代码与验证在 [demonstration_retention.py](training/experiments/demonstration_retention.py) 和 [约束报告](reports/stop_feedback_20261015/ppo_retention_protocol_20261016/REPORT.md)。

第二个实验选项为 `--long-wait-exploration-coefficient 0.001 --long-wait-minimum-seconds 2`，默认系数同样为 0；只在合法运动可选、观测有效、远离目标、已等待且停止概率饱和的恢复状态施加合法动作均匀先验 KL，停止仍可选。详见 [实现](training/experiments/long_wait_exploration.py)。该选项与下述示范保留实验分别对照：测试它时把 `--demo-retention-coefficient` 设为 `0`，使用独立输出目录，避免一次改变两个主要因素。

当前只证明实现正确及单次离线更新指标改善，**未证明新约束能改善闭环**。先完成第 4 节；确定开展独立小规模实验时，使用以下完整入口，输出到新目录：

```bash
python tools/prepare_native_bundle.py \
  --source models/stop_feedback_20261015/baseline/portable \
  --output training/runs/work_training_parent
nice -n 19 python -m experiments.train_prefix_recovery \
  --bundle training/runs/work_training_parent \
  --checkpoint models/quality_recovery_20261014/policy_2000.zip \
  --training-cases reports/stop_feedback_20261015/proactive_protocol_20261016/ppo_training_cases.json \
  --prefixes reports/stop_feedback_20261015/w11_success_augmented_protocol_20261016/ppo_training_prefixes.json \
  --demonstrations reports/stop_feedback_20261015/w11_success_augmented_protocol_20261016/demonstrations.json \
  --reference-bc models/stop_feedback_20261015/w11_recovery_bc800_s231015/policy_bc.zip \
  --bc-updates 800 --proactive-recovery --steps 2000 --seed 231015 --cpu-cores 0.35 \
  --demo-retention-coefficient 0.05 --demo-retention-batch-size 128 \
  --output training/runs/work_retention2000
```

配对的关闭约束实验采用相同配置、种子和 BC，把系数改为 `0`，输出改为 `training/runs/work_control2000`。两次各 2k，不能合称单个模型 4k。入口要求重新生成的 BC 网络和优化器与参考逐项一致，记录在 `INITIALIZATION_PARITY.json`；不一致需查原因，不能跳过。只检查初始化可加 `--initialization-only`，此时 PPO 为 0。

训练后分别运行 `experiments.audit_prefix_recovery_rollouts`、`experiments.analyze_recovery_learning`，再以双方自身 BC 和配对模型做第 4 节闭环验证。必须保留原 14 项、六项方向停靠、合理停车，并单独报告困难条件与新增条件。原任务退化、碰撞增加或长等待未改善时不自动扩步数。

最终模型和优化器可加载分析，但现训练 CLI **不支持精确恢复同一半回合的物理、雷达历史和全部内部状态**。新实验记录独立起点，不能直接把新步数接在旧 8k 后累计。旧 7k 部分快照可用 `tools/check_training_snapshot.py` 做历史完整性检查。

## 6. 定位和实车条件

定位改善来自里程计传播、静态支持筛选、弱方向约束、时序融合和自由射线矛盾检查，**并非这轮重新训练了定位网络**。独立时序父版 / 射线候选在 22 条冻结轨迹、2,855 帧同输入对照的位置 P95 为 18.51 / 18.04 cm，航向 P95 为 1.526 / 1.368°，定位 CPU P95 为 7.88 / 14.59 ms。收益有限且耗时增加，尚未合入本轮恢复模型。实现、原生编译与验证见 [定位说明](docs/localization/README.md)。

实车录包没有独立绝对位姿真值；不能用回放或仿真毫米级误差宣称实车精度、超过 AMCL、完成搬车重定位或达到 Orin Nano 耗时标准。已知地图、边导航边建图、开放稀疏场景和重新定位仍需专项验证。

硬件合同保持：车体约 39.39×32.95 cm、4 cm 余量；12 个离散动作，无横移，当前动作表无倒车；最高前进 0.20 m/s、转向 0.50 rad/s。220 束原始雷达、约 219.44°，保留后方盲区。历史 actor 按 12 m 归一化，独立定位候选按 10 m；恢复训练 436 维、方向运行时 439 维，区别于原部署 300 维。

当前闭环主要使用带噪声运动反馈，不能直接代表实车指令积分里程计。实车约 6%～12% 无效束、连续缺失、帧时间抖动、新障碍未入静态图仍需专门匹配；额外条件中的 0～4% 丢点不能替代这项验收。实车低层指令超时停车、可信运动反馈也需独立确认。

位置＋方向目标已接入，同位置、不同方向可转向；保留六项方向停靠回归。精简实车同输入回放命令：

```bash
nice -n 19 python -m experiments.evaluate_real_input_repair \
  --cache training/assets/localization_replay_20261015 \
  --analysis training/assets/real_recording_20261014/analysis \
  --record training/assets/real_recording_20261014/record \
  --hardware models/quality_recovery_20261014/portable/hardware.json \
  --output training/runs/work_localization_replay --cpu-cores 0.35
```

此回放分析原记录，不更新网络、不代表重新完成一次物理导航。

## 7. 目录和资源

| 目录 | 内容 |
|---|---|
| `training/neural_navigation/` | 地图、定位、路由、方向目标、反馈特征与保护 |
| `training/experiments/` | 快速二维闭环训练、示范审核、恢复训练、等待诊断、采样审计与评估 |
| `models/`、`reports/stop_feedback_20261015/` | 模型、优化器、配置、源码快照及完整成功/失败结果 |
| `reports/handoff_work_20261016/` | 本次固定状态、完成及部分验证行与对应原始轨迹 |
| `reports/handoff_ship_20261016/other_worktrees/` | 其他旧工作目录 20 份未提交文件和补丁，单独备份，未混入当前运行时 |
| `training/assets/localization_replay_20261015/` | 28 条冻结仿真输入，真值只供监督与评价 |
| `training/assets/real_recording_20261014/` | 1,516 条实车决策、1,911 条反馈、轮速/IMU、map_01 与来源清单 |
| `deployment/ros2_recording_adapter/` | ROS 2 记录和集成接口 |
| `docs/handoff/`、`tools/` | 哈希、迁移证据、原生重建与检查入口 |

新电脑先保持单线程、`nice -n 19`、**单任务软限 0.35 核、串行运行**，预留至少 4 GiB 内存余量。原机主任务 0.35 核、单个辅助验证 0.15 核；不要直接同时启动全部命令。新机速度以实测为准。

完整 `record.zip`、原始 rosbag、相机图像、虚拟环境、缓存和平台 `.so` 没有上传。精简数据足以运行上述定位回放；若后续需要原始 TF、图像或其他话题，应另外复制 `/home/ubuntu/rl_learn/record.zip` 及源录包，参见 [录制清单](training/assets/real_recording_20261014/MANIFEST.json)。源码快照中的旧绝对路径是来源记录，新电脑按本 README 的相对路径入口执行。

**换机后的第一步：安装并运行迁移检查，再补全冻结历史及额外条件验证。** 当前没有自动替换原部署模型，也没有宣称已经解决所有等待、定位、避障或泛化问题。
