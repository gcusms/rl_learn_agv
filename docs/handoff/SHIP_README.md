# AGV 导航：换电脑继续开发

交接分支：**[`ship`](https://github.com/gcusms/rl_learn_agv/tree/ship)**。本次固定快照截至 **2026-10-16 04:44:42（北京时间）**，包含当前源码、定位算法、原部署模型、新恢复候选、PPO 检查点与优化器、地图、精简实车数据、训练示范及成功/失败记录。

**当前可继续开发，尚未完成新模型的部署验收。** 新候选已经自主复现一个训练过的停滞恢复过程；完整 18 项回归在快照时只完成 2 项。独立 8k PPO 实验在旧机运行到 7,250 步，本分支归档最新完整的 7,000 步检查点，其权重已更新到 6,750 步。原实车部署模型保留。

精确范围：[SNAPSHOT.json](reports/handoff_ship_20261016/SNAPSHOT.json)；文件哈希：[FILES.json](docs/handoff/FILES.json)；迁移检查：[VERIFICATION.json](docs/handoff/VERIFICATION.json)。旧 `go` 快照见 [历史 README](docs/handoff/GO_README.md)，它的“运行中”和步数仅代表当时状态。

## 1. 当前解决什么，做到哪里

目标是让模型理解停车原因，在有安全可行动作时恢复运动，同时保留必要停车、避障、路径决策和位置＋方向停靠。当前采用冻结基础 actor、localizer、router，再训练有启用条件的恢复分支；仅增加恢复训练不能直接修复定位网络或路由网络。

| 版本 / 检查 | 已有结果 | 如何解读 |
|---|---|---|
| 原部署质量恢复 2k | 历史开发集 46/48 | 实车基线；与下面任务不是同一集合 |
| 2,030 标签、800 次示范拟合父模型 | 完整 18 项：15 到达、3 碰撞；其中原 14 项全部保留 | 当前完成完整回归的研究参考，追加 PPO 为 0 |
| **新 2,089 标签、800 次示范拟合候选** | 全部标签与导出动作一致；原 2,030 标签和 37 个合理停车标签保留 | 新增 59 个完整安全恢复过程的有效样本；追加 PPO 为 **0** |
| 新候选在 W11 的 90 秒历史后自主接管 | **接管后 20.4 秒到达，终点误差 9.39 cm，0 碰撞** | 同一训练历史，证明已学会该过程；不能当未见场景泛化 |
| 新候选完整任务 `motion_profile_015` | **49.2 秒、6.391 m、误差 8.78 cm，安全到达** | 父模型 48.0 秒、6.456 m；首项原成功能力保留 |
| 新候选困难任务 `recovery_holdout_00` | **140 秒超时、0 碰撞，末端距目标 3.32 m** | 父模型在 17.2 秒碰撞；停车避免碰撞，但脱困仍未解决 |
| 新候选完整回归、两个原接管、95/106 秒后期接管 | 快照时仍未完成 | 不能用上面单条成功代替完整验收 |
| **独立 8k PPO** | 旧机采集 7,250/8,000；归档 7,000 步、优化到 6,750 步 | 使用原 2,030 标签，尚未加入新 59 标签；未完成最终闭环评估 |
| 实际停车审计 | 8 项检查通过；20 条完成轨迹的旧指标保持一致 | 区分主动停止、只允许停止、选择运动但实际零指令；没有修改保护规则 |

W11 前 90 秒是固定物理历史初始化，之后 103 帧由新模型和原保护控制，没有运行时定时教师。接管后存在可行动作时的等待为 0 秒；仅允许停止时仍停车 4 秒。成功来自新增可行示范，**不是本候选又做了 PPO**。第一条“只左转再交回参考”的诊断仍超时，完整失败记录也保留。

证据：[新候选训练方案与分析](reports/stop_feedback_20261015/w11_success_augmented_protocol_20261016/REPORT.md)、[2,089 标签检查](reports/stop_feedback_20261015/w11_recovery_bc800_fit_20261016/RESULTS.json)、[实际自主恢复](reports/stop_feedback_20261015/w11_recovery_bc800_early_20261016/RESULTS.json)、[本次固定的两项完整任务](reports/handoff_ship_20261016/w11_navigation/RESULTS.json)、[父模型完整 18 项](reports/stop_feedback_20261015/proactive_augmented_bc800_gate_20261016/RESULTS.json)、[停车指标对照](reports/stop_feedback_20261015/executed_wait_metrics_20261016/RESULTS.json)。

## 2. 新电脑安装与检查

先按 Ubuntu 22.04、Python 3.10、CPU 环境复现。虚拟环境和平台二进制不跨电脑复制。

```bash
git clone --depth 1 --branch ship https://github.com/gcusms/rl_learn_agv.git
cd rl_learn_agv
sudo apt-get install python3.10-venv g++ libeigen3-dev
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r training/requirements.txt
export PYTHONPATH=training:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# 输出目录须不存在；验证文件、九份模型、真实采样和实现测试。
# 不启动训练，不向电机发布命令。
nice -n 19 python tools/check_handoff.py --output training/runs/ship_check
```

建议预留至少 8 GB 安装空间，下载缓存另计。Jetson 需使用与 JetPack 匹配的 PyTorch；`training/requirements.lock.txt` 是原机版本记录。上述检查不要求 CUDA、ROS 2 或显示器，不等于导航验收。

运行闭环导航前需要重建原生定位库，见下一节。`.so` 依赖系统架构和编译环境，源码已归档；编译不会训练网络。若要一次检查全部原生包，可给 `check_handoff.py` 加 `--build-native`，会额外编译十个包。

## 3. 模型在哪里

路径均相对仓库根目录。模型、manifest、特征与运行时需要一起使用。

| 路径 | 用途 |
|---|---|
| `models/quality_recovery_20261014/policy_2000.zip` | **原实车部署模型**，没有替换 |
| `models/stop_feedback_20261015/w11_recovery_bc800_s231015/` | **最新研究候选**：`policy_bc.zip`、`bc_portable`、优化器、2,089 条示范、配置和源码快照 |
| `models/stop_feedback_20261015/proactive_augmented_bc_800_s231015/` | 直接父模型，2,030 标签、800 次拟合，完整 18 项已完成 |
| `models/stop_feedback_20261015/proactive_ppo_ship_20261016/` | **本次运行中 PPO 的固定快照**：7k 检查点、优化器、7k 实际样本、预热、源码和 BC 初始化 |
| `models/stop_feedback_20261015/proactive_ppo_snapshot_20261016/` | 上次 `go` 的 2k 历史快照，保留 |
| `models/stop_feedback_20261015/steady_base_ppo_2000_s231015/` | 已完成的独立恢复 PPO 2k，历史对照 |
| `models/stop_feedback_20261015/baseline/portable/` | 303 维恢复训练父包 |
| `models/stop_feedback_20261015/ppo_4000_s221015/initial_portable/` | 439 维冻结配对参考，与上面父包不同 |
| `models/localization_mapping_20261015/ray_consistent_candidate/portable/` | 独立定位内核候选，尚未合入本轮恢复训练 |
| `models/docking_selected_5000_20261015/` | 历史位置＋方向停靠模型 |

新 BC ZIP SHA256：`76bd054747c59d5cb1c625e61c9411cce10744346ea97e4b1308433d236fdbc8`。

归档 7k PPO ZIP SHA256：`fd0f37302865c919e7daa9bc2873b3f24f1f5c86676ceeac0e239f103bed9e0c`。

原部署 ZIP SHA256：`7a3dad60f76060daf6b311b4bc6db921d3925ef0cdcf11683fe99e282173de11`。

`policy_bc.zip` / `bc_portable` 是示范初始化；完成 PPO 后的 `policy.zip` / `portable` 才是最终 PPO 产物。中期 `policy_step_7000.zip` 可加载分析，未完成最终导出。恢复检查点使用 `experiments.stop_recovery_policy.StopRecoveryPPO.load`；运行时按 manifest 的 `required_runtime` 调用 `resolve_runtime`，见 [check_handoff.py](tools/check_handoff.py)。

## 4. 新电脑先继续验证

以下命令在仓库根目录执行，沿用上方环境变量。每次使用不存在的输出目录；同类任务串行运行。

先重建新候选与直接父模型：

```bash
python tools/prepare_native_bundle.py \
  --source models/stop_feedback_20261015/w11_recovery_bc800_s231015/bc_portable \
  --output training/runs/ship_candidate
python tools/prepare_native_bundle.py \
  --source models/stop_feedback_20261015/proactive_augmented_bc_800_s231015/bc_portable \
  --output training/runs/ship_parent
```

完整 18 项配对验证，原成功与困难任务全部保留：

```bash
nice -n 19 python -m experiments.evaluate_docking_navigation \
  --candidate training/runs/ship_candidate --candidate-name w11_bc800 \
  --baseline training/runs/ship_parent --baseline-name parent_bc800 \
  --cases reports/stop_feedback_20261015/w11_success_augmented_protocol_20261016/gate_cases.json \
  --output training/runs/ship_gate --cpu-cores 0.35
```

然后按顺序检查两个原前缀、W11 的 90/95/106 秒相关历史、12 个预先冻结的额外条件：

```bash
nice -n 19 python -m experiments.evaluate_feedback_recovery \
  --bundle training/runs/ship_candidate \
  --specifications reports/stop_feedback_20261015/frozen_recovery_cases.json \
  --output training/runs/ship_prefixes --label w11_bc800 --cpu-cores 0.35
nice -n 19 python -m experiments.evaluate_feedback_recovery \
  --bundle training/runs/ship_candidate \
  --specifications reports/stop_feedback_20261015/proactive_protocol_20261016/w11_prefixes.json \
  --output training/runs/ship_w11 --label w11_bc800 --cpu-cores 0.35
nice -n 19 python -m experiments.evaluate_docking_navigation \
  --candidate training/runs/ship_candidate --candidate-name w11_bc800 \
  --baseline training/runs/ship_parent --baseline-name parent_bc800 \
  --cases reports/stop_feedback_20261015/w11_success_augmented_protocol_20261016/fresh_cases.json \
  --output training/runs/ship_fresh --cpu-cores 0.35
```

父模型在同一 90 秒起点的对照也尚未完成，可将 `--bundle` 改为 `training/runs/ship_parent`，使用 `w11_success_augmented_protocol_20261016/w11_t90_evaluation.json` 单独运行。不要把它与旧 1,058 标签模型的结果混淆。

这些都是开发任务，部分同源、同地图，不能声称全新地图泛化。旧电脑任务状态仅记录在快照中，进程与 systemd 服务不会随 Git 迁移。本地来源脚本 `run.py`、`after_preservation.py` 等含原机目录或等待关系，**新电脑使用本 README 的直接入口**。复跑已有任务属于重现，不是新增独立成功。

## 5. 训练如何继续

优先顺序是：**完整回归 → 分析失败中的可行动作与停车原因 → 决定训练哪个组件 → 独立微调 → 闭环验收**。若原 14 项成功、正常避障或 6 项方向停靠退化，先查原因，不直接增加步数。

先检查已归档的真实 PPO，不改变其训练状态：

```bash
nice -n 19 python tools/check_training_snapshot.py \
  --model-run models/stop_feedback_20261015/proactive_ppo_ship_20261016
```

7k 检查点是在最后 250 步更新之前保存的：采集 7,000 步，权重优化到 6,750 步。网络、优化器和 RNG 已保存，但当前训练 CLI **不支持恢复同一半回合、全部物理环境及历史状态**。不能直接把新实验步数接在 7,250 后累计；需要精确续训时先实现和验证环境状态恢复入口。

若决定复现旧机正在跑的 2,030 标签实验，先按 [go README 第 4.3 节的训练命令](docs/handoff/GO_README.md) 在新输出目录重开。它使用 `proactive_augmented_protocol_20261016` 的示范和父 BC，不含新增 59 标签。旧机本次 7,250 步用时约 83 分钟、平均 0.35 核、峰值 RSS 约 2.46 GiB；新机速度需实测，完整评估另计。

若新 2,089 标签候选通过保留检查后需要独立 PPO，使用同一入口 `experiments.train_prefix_recovery`，替换为：

- `--demonstrations reports/stop_feedback_20261015/w11_success_augmented_protocol_20261016/demonstrations.json`
- `--prefixes reports/stop_feedback_20261015/w11_success_augmented_protocol_20261016/ppo_training_prefixes.json`
- `--reference-bc models/stop_feedback_20261015/w11_recovery_bc800_s231015/policy_bc.zip`
- 保留 `--bc-updates 800 --proactive-recovery --seed 231015`，使用独立输出目录。

入口先重新生成 BC，并要求网络和优化器与指定参考逐项一致，才会进行 PPO；不应绕过 `INITIALIZATION_PARITY.json` 检查。`--initialization-only` 只复现示范拟合和烟雾检查，此时 PPO 为 0。完成训练后先运行 `experiments.audit_prefix_recovery_rollouts`、`experiments.analyze_recovery_learning`，再与自身 BC 做同输入闭环对照。不能只凭奖励、权重变化或示范拟合选模型。

## 6. 定位、硬件合同与限制

定位目前的改善来自里程计传播、静态支持筛选、弱方向约束、时序融合和自由射线矛盾检查，**并非新训练了定位网络**。22 条冻结轨迹、2,855 帧同输入对照，时序父版 / 射线候选的位置 P95 为 18.51 / 18.04 cm，航向 P95 为 1.526 / 1.368°，定位 CPU P95 为 7.88 / 14.59 ms。改善有限、耗时增加；射线候选尚未自动合入本轮恢复模型。

定位实现与验收：[docs/localization/README.md](docs/localization/README.md)。原始实车没有独立绝对位姿真值，不能据此宣称超过 AMCL。已知地图、边导航边建图、开放少特征、搬车重定位和 Jetson Orin Nano 耗时仍需要专项验证。

硬件合同保持：车体约 39.39×32.95 cm、4 cm 余量；12 个离散动作、无横移、当前动作表无倒车；最高前进 0.20 m/s、转向 0.50 rad/s。220 束雷达、约 219.44°，后方盲区保留。历史 actor 按 12 m 归一化，独立定位候选按 10 m；训练 436 维、方向运行时 439 维，与原部署 300 维合同不同，不能只换 actor 文件。

当前闭环验证主要使用带噪声运动反馈，不能直接代表实车的指令积分里程计。实车 6%～12% 无效束、连续缺失、时间抖动和新增障碍物未入静态图，仍需增加匹配条件。不能靠缩小车体、放宽掩码、关闭保护或强制最低速度换成功率。

精简实车回放入口：

```bash
nice -n 19 python -m experiments.evaluate_real_input_repair \
  --cache training/assets/localization_replay_20261015 \
  --analysis training/assets/real_recording_20261014/analysis \
  --record training/assets/real_recording_20261014/record \
  --hardware models/quality_recovery_20261014/portable/hardware.json \
  --output training/runs/ship_localization_replay --cpu-cores 0.35
```

该回放评估同一份记录，不会重演新的物理导航，也不更新网络。位置＋方向目标已接入，同位置不同朝向可转向；原方向能力须继续回归。实车低层超时停车、可信运动反馈仍应独立验证。

## 7. 目录、资源与交接边界

| 目录 | 内容 |
|---|---|
| `training/neural_navigation/` | 定位、地图、路由、反馈特征、方向目标与保护 |
| `training/experiments/` | 训练、示范审核、停车诊断、采样审计、闭环评估与测试 |
| `models/`、`reports/` | 网络、优化器、配置、源码快照与成功/失败记录 |
| `reports/handoff_ship_20261016/` | 本次时间固定的训练状态、原机任务及两项完成回归 |
| `reports/handoff_ship_20261016/other_worktrees/` | 另外两个旧工作目录中未提交的 20 份文件及补丁，按原始提交单独备份，未并入当前运行时 |
| `reports/stop_feedback_20261015/w11_success_augmented_protocol_20261016/` | 新 59 标签来源、训练方案、拟合和自主恢复证据、待完成配对计划 |
| `training/assets/localization_replay_20261015/` | 28 条冻结仿真输入，真值只供监督/评价 |
| `training/assets/real_recording_20261014/` | 1,516 条实车决策、1,911 条反馈、轮速/IMU、map_01 与来源清单 |
| `deployment/ros2_recording_adapter/` | ROS 2 数据记录与集成接口 |
| `docs/handoff/`、`tools/` | 哈希、迁移证据、原生重建和检查入口 |

新电脑先保持单线程、`nice -n 19`、单任务软限 0.35 核，内存预留 4 GiB。旧机主训练 0.35 核、辅助验证 0.15 核；新电脑先串行，确认余量再增加。

完整 `record.zip`、原始 rosbag、相机图像、虚拟环境、缓存、平台 `.so` 没有上传。仓库内精简数据足以运行上述定位回放；如需原始 TF、图像或其他话题，应另外复制录包，见 [实车清单](training/assets/real_recording_20261014/MANIFEST.json)。源码快照中可能保留旧绝对路径作为来源记录，可执行入口以上方相对路径命令为准。

**换电脑后的第一步：运行迁移检查，再完成新候选的 18 项回归与冻结配对评估。** 旧机在快照之后继续产生的结果不会自动进入此分支。发布选择必须同时看真实到达、碰撞、超时、合理停车、恢复、定位与延迟，当前没有自动替换原部署模型。
