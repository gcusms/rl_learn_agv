# AGV 导航：换电脑继续开发

交接分支：**`go`**；仓库：[gcusms/rl_learn_agv](https://github.com/gcusms/rl_learn_agv/tree/go)。本次整理截至 **2026-10-16 03:53:58（北京时间）**：源码、定位算法、原部署模型、候选模型、优化器、地图、精简实车数据、训练示范、成功和失败记录及复现入口均已归档。精确范围见 [SNAPSHOT.json](reports/handoff_go_20261016/SNAPSHOT.json)，文件 SHA256 见 [FILES.json](docs/handoff/FILES.json)。

**完整闭环评估过的最新候选是 800 次示范拟合模型；另一份 8k PPO 实验仍在训练，尚未通过部署验收。** 快照时实际已采集 **2,750/8,000 步**；归档最新完整写盘的 **2,000 步检查点**，其权重已更新到 **1,750 步**，最后 250 步只采集未更新。 固定 14 项开发回归全部保留，另 4 个不同条件仅 1 个安全到达、3 个碰撞。不能将示范拟合准确率当作导航成功率，也不能认定最新文件就是最佳部署模型。原实车模型保留。

## 1. 当前进度与关键结果

当前重点：让模型理解停车原因，在存在有效安全动作时尝试恢复，同时保留必要停车、正常导航与位置＋方向停靠。

| 版本 / 工作 | 已完成结果 | 结论与限制 |
|---|---|---|
| 原部署质量恢复 2k | 历史开发集 46/48 | 作为实车基线保留；不是下方同一套任务 |
| 冻结原参考 | 固定 14 项为 12/14 到达；6/6 方向停靠 | 配对参考 |
| 首轮恢复 PPO | BC＋2,000 步，固定 14 项为 13/14，1 碰撞 | 与自身 BC 的确定性轨迹一致，未证明 PPO 的额外收益 |
| steady-base PPO | 独立 2,000 步，同样 13/14 | 与前轮不是累计 4k；限制随机探索没有解决全部问题 |
| proactive BC，1,058 帧 | 400 次示范拟合；14/14，另 4 条件 1/4 | 改善接近障碍时的提前决策，但泛化不足 |
| 新增训练轨迹 | 8 条冻结条件：7 条完整安全、1 条超时 | 是离线合法干预示范，不是模型自主成功；失败保留 |
| 扩充 BC，2,030 帧，400 次 | 2,006/2,030 标签吻合，24 处运动错误 | 仅作拟合对照，不推荐部署 |
| **扩充 BC，2,030 帧，800 次** | **2,030/2,030 标签吻合，14/14 原任务安全；另 4 条件 1/4** | 全部 18 项完成，合计 15 到达、3 碰撞；追加 PPO **0** |
| 旧 proactive 的 w11 停滞接管 | 90 / 95 / 106 秒接管全部超时 | 同源历史，不是 3 条独立新任务；仍未解决 |
| 扩充 BC 的两个 20 秒前缀接管 | **2/2 安全到达**，46.0 / 47.4 秒 | 前缀初始化不计自主能力；满足进入实验的保留条件 |
| **实际 8k PPO** | 快照时采集 **2,750/8,000**；保存 **2k 检查点、优化器及 2k 真实样本** | 仍在旧机运行；归档检查点优化到 1,750 步，未做完整导航验收 |
| PPO 首个 1k 检查点诊断 | 710 个可训练状态中 25 个贪心动作改变；156 个长等待状态无停止→运动改善 | 确实在学习，但不能据此说等待问题已解决 |
| w11 提前原地左转诊断 | 第一个完整继续运行仍超时；第二个在快照时未完成 | 合法动作不一定能脱困，诊断不是训练模型成绩 |

扩充数据保留原 1,058 帧不变，再加入 972 帧，共 1,993 个运动、37 个合理停车标签。800 次拟合用时 **123.69 秒**、峰值 RSS **1,770.78 MiB**、平均约 **0.35 核**；最低运动标签概率 **0.519**，未达到 0.9 置信目标。所有 2,030 帧检查点与 TorchScript 动作一致、基础 actor 未改变；这只是拟合和导出证据。

当前 800 次候选将 `motion_profile_015` 从原参考 11.6 秒碰撞改善为 **48.0 秒安全到达**，行驶 **6.456 m**，终点误差 **9.77 cm**。四个不同条件的成功任务从上一 proactive 的条件 01 换为条件 02，整体仍只有 1/4，未形成全面改进。

完整证据：[18 项导航](reports/stop_feedback_20261015/proactive_augmented_bc800_gate_20261016/RESULTS.json)、[配对汇总](reports/stop_feedback_20261015/proactive_augmented_bc800_gate_20261016/SUMMARY.json)、[2,030 帧检查](reports/stop_feedback_20261015/proactive_augmented_fit800_20261016/RESULTS.json)、[训练来源及选择](reports/stop_feedback_20261015/proactive_augmented_protocol_20261016/DATA.json)、[8 条含失败示范](reports/stop_feedback_20261015/proactive_training_probes_20261016/RESULTS.json)、[w11 三次超时](reports/stop_feedback_20261015/proactive_bc_w11_recovery_20261016/RESULTS.json)。这些是反复使用的开发任务，不能据此声称未见地图泛化或超越 A* / TEB。

最新实际证据：[两个安全接管](reports/stop_feedback_20261015/proactive_augmented_bc800_prefixes_20261016/RESULTS.json)、[8k 实验准入记录](reports/handoff_go_20261016/PPO_ADMISSION.json)、[实际 500 帧时序与概率审计](reports/stop_feedback_20261015/proactive_actual500_audit_20261016/RESULTS.json)、[1k 检查点的动作变化](reports/stop_feedback_20261015/proactive_interim1000_transitions_20261016/RESULTS.json)、[缺失转向示范诊断](reports/stop_feedback_20261015/interim_learning_protocol_20261016/REPORT.md)、[12 个预先冻结任务的物理可行性](reports/stop_feedback_20261015/proactive_fresh_geometry_20261016/RESULTS.json)。这些诊断不等于新一轮闭环成功率。当前可行动作全合法的同一恢复起点，1k 检查点仍以约 99.7% 概率选择前进左转，原地左转概率约 5.8×10⁻¹⁰；示范中该动作只有 2/2,030 帧。是否补示范必须先验证完整物理轨迹，不能直接把失败左转当专家答案。

## 2. 新电脑安装与完整性检查

先在 Ubuntu 22.04、Python 3.10 上复现。CPU 检查不依赖 CUDA、ROS 2 或显示器，不复制旧虚拟环境。

```bash
git clone --depth 1 --branch go https://github.com/gcusms/rl_learn_agv.git
cd rl_learn_agv
sudo apt-get install python3.10-venv g++ libeigen3-dev
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r training/requirements.txt
export PYTHONPATH=training:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# 输出目录须不存在；不连接电机、不启动训练。
python tools/check_handoff.py --build-native --output training/runs/go_check
```

预留至少 **8 GB** 安装空间，下载缓存另计。Jetson 使用与 JetPack 匹配的 PyTorch；`requirements.lock.txt` 是原机版本记录。原生 `.so` 与平台有关，必须用同版本源码在目标电脑重建；编译不会训练模型。检查会加载 **8 份反馈检查点及 TorchScript**、重建 **9 个原生包**并运行实现测试。另检查归档 PPO 的优化器、2,000 帧采样、动作掩码及基础 actor 是否保留。实际迁移检查记录见 [VERIFICATION.json](docs/handoff/VERIFICATION.json)，与闭环导航验收分开。

## 3. 模型、合同与硬件条件

以下路径相对仓库根目录：

| 路径 | 用途 |
|---|---|
| `models/quality_recovery_20261014/policy_2000.zip` | **原实车部署基线**，保留 |
| `models/stop_feedback_20261015/proactive_ppo_snapshot_20261016` | **运行中实验的只读快照**：`policy_step_2000.zip`、优化器、原始采样、预热历史；无最终部署导出 |
| `models/stop_feedback_20261015/proactive_augmented_bc_800_s231015` | **当前继续研究起点**：`policy_bc.zip`、`bc_portable`、优化器、示范、配置、拟合记录 |
| `models/stop_feedback_20261015/proactive_augmented_bc_s231015` | 同一 2,030 帧的 400 次拟合对照 |
| `models/stop_feedback_20261015/proactive_bc_s231015_prepared` | 上一 1,058 帧 BC，14/14 开发回归 |
| `models/stop_feedback_20261015/steady_base_ppo_2000_s231015` | 最新已经完整结束的恢复 PPO，独立追加 2k |
| `models/stop_feedback_20261015/prefix_ppo_2000_s231015` | 首轮恢复 PPO，独立追加 2k |
| `models/stop_feedback_20261015/baseline/portable` | 恢复训练的 303 维父包，训练前在新电脑重建原生库 |
| `models/stop_feedback_20261015/ppo_4000_s221015/initial_portable` | **439 维冻结配对参考**，不是上述 303 维父包 |
| `models/localization_mapping_20261015/ray_consistent_candidate/portable` | 独立定位算法候选，未自动合入本轮恢复训练 |
| `models/docking_selected_5000_20261015` | 历史位置＋方向停靠候选 |

最新 800 次 BC ZIP SHA256：`2062c2ae96aaa9837923c1cc717196875336b6716e18203013405ca2c1cefab0`。原部署 ZIP SHA256：`7a3dad60f76060daf6b311b4bc6db921d3925ef0cdcf11683fe99e282173de11`。

`policy_bc.zip` / `bc_portable` 表示示范初始化；只有完成 PPO 后的 `policy.zip` / `portable` 才是最终 PPO 产物。`policy_migrated.zip` 是零残差初始模型，不能当最新训练成果。恢复检查点用 `StopRecoveryPPO.load`；运行时按 manifest 的 `required_runtime` 调用 `resolve_runtime`，示例见 [check_handoff.py](tools/check_handoff.py)。

原部署观测 300 维，方向父包 303 维；停车反馈训练 436 维、方向运行时 439 维。**模型、特征、manifest、运行时一起迁移**，不能只换 actor。12 个离散动作，无横移、无倒车；最高前进 0.20 m/s，转向 0.50 rad/s。车体约 39.39×32.95 cm，4 cm 保护余量。雷达 220 束、约 219.44°，保留后方盲区；定位候选按 10 m，历史 actor 按 12 m 归一化。当前没有修改这些条件。

## 4. 新电脑从哪里继续

先检查包和完整回归，再决定启动独立训练。旧电脑的进程、systemd 服务、内存中的物理状态不会随 Git 迁移；本分支对未完成工作保存了只读快照，**当前入口不支持恢复到同一半回合和同一物理状态**。检查点可以加载、分析和作为后续适配入口的起点；下方正式训练命令会从已验证 BC 重开独立 8k 实验，不是从 2,750 步接着加计。若旧电脑仍在训练，不要在两台机器上重复启动同名实验后混合日志。

### 4.1 重建当前模型并复评

下方命令均在仓库根目录、激活虚拟环境并设置上方环境变量后执行；每次用不存在的输出目录，一次一个任务。

```bash
python tools/prepare_native_bundle.py \
  --source models/stop_feedback_20261015/proactive_augmented_bc_800_s231015/bc_portable \
  --output training/runs/go_bc

# 全部 18 条：原 14 项和全部四个不同条件，失败不筛掉。
nice -n 19 python -m experiments.evaluate_docking_navigation \
  --candidate training/runs/go_bc \
  --cases reports/stop_feedback_20261015/proactive_augmented_protocol_20261016/gate_cases.json \
  --output training/runs/go_navigation --candidate-name augmented_bc800 --cpu-cores 0.35

# 两个 20 秒物理前缀之后接管；初始化不计模型自主成绩。
nice -n 19 python -m experiments.evaluate_feedback_recovery \
  --bundle training/runs/go_bc \
  --specifications reports/stop_feedback_20261015/frozen_recovery_cases.json \
  --output training/runs/go_prefixes --label augmented_bc800 --cpu-cores 0.35

# 拟合 / 导出检查，不等于导航成功。
nice -n 19 python -m experiments.inspect_recovery_initialization \
  --model models/stop_feedback_20261015/proactive_augmented_bc_800_s231015 \
  --demonstrations reports/stop_feedback_20261015/proactive_augmented_protocol_20261016/demonstrations.json \
  --output training/runs/go_fitcheck --cpu-cores 0.35
```

若比较冻结参考，先重建 `models/stop_feedback_20261015/ppo_4000_s221015/initial_portable`，再给导航评估增加 `--baseline 新重建目录 --baseline-name frozen_reference`。完整 18 条已有记录；新电脑复评是重现，不能算新增独立证据。

### 4.2 继续开发前，先检查已归档的实际 PPO

先验证归档检查点和优化器，再重现已有中期分析。它不会恢复机器人运动、启动训练或生成新的成功率：

```bash
python tools/check_training_snapshot.py \
  --model-run models/stop_feedback_20261015/proactive_ppo_snapshot_20261016
nice -n 19 python -m experiments.analyze_recovery_learning \
  --model-run models/stop_feedback_20261015/proactive_ppo_snapshot_20261016 \
  --through-steps 1000 --output training/runs/go_interim1000 --cpu-cores 0.15
```

当前工作顺序：保留原模型 → 核对 2k 快照 → 等旧机 8k 及完整配对评估，或在新电脑独立复现 → 确认有真实收益再选择模型。w11 原地左转的第一个完整诊断仍超时，另一个快照时未完成；不因某一帧动作合法就新增运行时强制规则。已有 14 项导航和 6 项方向能力必须继续回归，不能通过缩车体、放宽碰撞检查或强制最低速度解决等待。

### 4.3 如需在新电脑重新开展完整实验：800 次 BC ＋ 8,000 步 PPO

实验计划：[PPO_PLAN.json](reports/stop_feedback_20261015/proactive_augmented_protocol_20261016/PPO_PLAN.json)。**进入实验的条件**是保留原 14 项成功及两个安全接管；四个困难条件必须完整报告，作为学习问题继续处理。这不是部署合格条件。原任务退化时先排查，不自动扩步数。

本分支包含全部训练来源：12 个训练条件、3 个真实物理前缀、2,030 条示范，以及 8 条新来源中的失败轨迹。w11 的 90 秒前缀现在用于训练演练，90 / 95 / 106 秒同源接管不能再称作未见留出测试。12 条额外评估条件在 PPO 前冻结，见 [FRESH_PLAN.json](reports/stop_feedback_20261015/proactive_augmented_protocol_20261016/FRESH_PLAN.json)；仍涉及已用地图，不声称全新地图泛化。

确认回归满足实验条件后，在新输出目录启动：

```bash
python tools/prepare_native_bundle.py \
  --source models/stop_feedback_20261015/baseline/portable \
  --output training/runs/go_parent

nice -n 19 python -m experiments.train_prefix_recovery \
  --bundle training/runs/go_parent \
  --checkpoint models/quality_recovery_20261014/policy_2000.zip \
  --training-cases reports/stop_feedback_20261015/proactive_protocol_20261016/ppo_training_cases.json \
  --prefixes reports/stop_feedback_20261015/proactive_augmented_protocol_20261016/ppo_training_prefixes.json \
  --demonstrations reports/stop_feedback_20261015/proactive_augmented_protocol_20261016/demonstrations.json \
  --reference-bc models/stop_feedback_20261015/proactive_augmented_bc_800_s231015/policy_bc.zip \
  --bc-updates 800 --proactive-recovery --steps 8000 --seed 231015 --cpu-cores 0.35 \
  --output training/runs/go_ppo8000
```

入口从原基线重新生成 BC，**必须先与已验证 BC 的网络和优化器逐项字节一致**，才会开始 PPO；不一致会停止，不能跳过检查后宣称同初始化对照。`INITIALIZATION_PARITY.json` 记录一致性。加 `--initialization-only` 可以只复现 BC 和两帧动作；此时实际 PPO 为 0，不是训练完成。

原机 2k 实际 PPO 用时约 20 分钟；按相同 0.35 核预算，8k 主训练粗估 **80～110 分钟**，完整配对评估另计。新机器以实测速率为准。优化器和 RNG 已保留，但 CLI 没有精确恢复半回合及全部物理状态的功能；新实验应记录新的起点。

训练完成后先做真实采样概率审计，再做与自身 BC 的配对闭环比较：

```bash
python -m experiments.audit_prefix_recovery_rollouts \
  --model-run training/runs/go_ppo8000 --output training/runs/go_audit
nice -n 19 python -m experiments.analyze_recovery_learning \
  --model-run training/runs/go_ppo8000 --output training/runs/go_learning --cpu-cores 0.35
nice -n 19 python -m experiments.evaluate_docking_navigation \
  --candidate training/runs/go_ppo8000/portable --bc training/runs/go_ppo8000/bc_portable \
  --cases reports/stop_feedback_20261015/proactive_augmented_protocol_20261016/gate_cases.json \
  --output training/runs/go_paired --candidate-name ppo8000 --cpu-cores 0.35
```

随后用相同两份包分别检查两个原前缀、w11 相关历史，以及 `fresh_cases.json` 中全部 12 个条件。实际 PPO 步数、动作变化、合理等待、碰撞、超时、位置/方向误差、保护介入、推理延迟与原成功任务退化一起判断；不能只凭奖励上涨或权重变化选模型。

`reports/.../after_validation.py`、`validate_bc800.py`、`probe_w11_after_reference.py` 是**旧电脑任务来源记录**，会等待原机的 systemd invocation；新电脑请使用上面的直接入口，不执行这些等待脚本。

## 5. 定位、方向停靠和仍未解决的问题

定位改善来自里程计传播、静态支持筛选、弱方向约束、时序融合和自由射线矛盾检查，**没有新训练定位网络**。22 条冻结轨迹、2,855 帧同输入对照，时序父版 / 射线候选位置 P95 为 **18.51 / 18.04 cm**，航向 P95 为 **1.526 / 1.368°**，终点平均误差 **3.42 / 3.27 cm**；Python 定位 CPU P95 **7.88 / 14.59 ms**。改善有限，耗时增加。恢复训练仍使用时序父包，没有自动合并射线定位候选。

实车没有独立绝对位姿真值，不能声称精度超过 AMCL。位置＋方向目标已接入，同位置不同方向可转向；方向回归 6/6 不等于所有陌生环境都合格。已知地图、边导航边建图、开放少特征场景、搬车重定位和 Orin Nano 端到端预算仍需专项验收。

w11 的离线几何分析见 [RESULTS.json](reports/stop_feedback_20261015/w11_stopping_geometry_20261016/RESULTS.json) 和 [geometry.png](reports/stop_feedback_20261015/w11_stopping_geometry_20261016/geometry.png)。问题不只是静态墙，停车障碍物、接近阶段决策及安全记忆约束均可能参与。更早左转是待验证假设，不能因后期可选动作少就关闭保护、强制最低速度或模仿失败轨迹。

定位代码与验证：[docs/localization/README.md](docs/localization/README.md)；实车输入修复：[REPORT.md](reports/localization_mapping_20261015/real_input_repair_20261015/REPORT.md)；轮速＋IMU：[measured_motion](deployment/ros2_recording_adapter/examples/measured_motion/README.md)。

```bash
# 同一份真实输入回放；不更新模型、不改变当时的物理轨迹。
nice -n 19 python -m experiments.evaluate_real_input_repair \
  --cache training/assets/localization_replay_20261015 \
  --analysis training/assets/real_recording_20261014/analysis \
  --record training/assets/real_recording_20261014/record \
  --hardware models/quality_recovery_20261014/portable/hardware.json \
  --output training/runs/go_localization_replay --cpu-cores 0.35
```

## 6. 资源、目录与交接范围

保持单线程、低优先级。原机主任务软限 **0.35 核**，辅助 **0.15 核**，合计约半核；单任务建议内存上限 **4 GiB**。先串行运行，再按新电脑实测余量调整。原机后台任务仍可能继续，**本分支严格表示 03:53:58 的快照**，之后产生的结果不会自动同步。

| 目录 | 内容 |
|---|---|
| `training/neural_navigation/` | 定位、地图、路由、反馈特征、方向目标和运行时保护 |
| `training/experiments/` | 训练、示范审核、诊断、采样审计、闭环评估、测试 |
| `models/`、`reports/` | 模型、优化器、配置、源代码快照、成功与失败证据 |
| `training/assets/localization_replay_20261015/` | 28 条冻结仿真输入，真值仅供训练监督/评估 |
| `training/assets/real_recording_20261014/` | 精简实车记录：1,516 条决策、1,911 条反馈、轮速/IMU、map_01 |
| `deployment/ros2_recording_adapter/` | ROS 2 记录和集成接口 |
| `reports/handoff_go_20261016/` | 当前精确交接快照、旧机任务状态、准入记录及进行中左转诊断 |
| `reports/handoff_carry_20261016/` | 前一交接的历史快照，日期状态不代表现在 |
| `docs/handoff/FILES.json`、`VERIFICATION.json` | 当前文件哈希与迁移检查 |

未包含完整 `record.zip`、原始 rosbag、相机图像、虚拟环境、缓存和平台 `.so`。精简定位数据足以运行本 README 的回放；分析原始 TF、图像或其他话题时需另行复制原始录包，详见 [实车数据清单](training/assets/real_recording_20261014/MANIFEST.json)。

上一交接：[carry README](docs/handoff/CARRY_README.md)、[carry 检查](docs/handoff/CARRY_VERIFICATION.json)。早期分支、旧绝对路径和历史“运行中”只代表当时来源；以本 README、当前 manifest 与快照为准。下一步优先完成当前 8k PPO 和预先约定的配对评估，核对长等待是否改善；如需补训练示范，先要求完整安全闭环，再做独立实验，保留普通导航、避障和方向停靠的回归记录。
