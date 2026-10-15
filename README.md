# AGV 导航：换电脑继续开发

交接分支：**`recovery`**，整理日期：**2026-10-16（北京时间）**。包含源码、基线与候选模型、优化器、必要地图、精简实车录制、完整失败对照及验证入口。当前状态以[本次快照](reports/handoff_recovery_20261016/SNAPSHOT.json)为准。上一份交接保留在 `resume` 分支（`a991e91`）。

**原部署基线保留；新恢复模型仍是实验候选。** 当前重点是解决“模型不知道为什么被停车”和“有运动选项却长期不尝试”。最新三模型对照已完成 42/42 条：固定 14 例中原参考安全到达 12 例，示范初始化版及追加 2k PPO 版均为 13 例；6 例方向停靠均保留。两种新模型的确定性轨迹相同，尚未证明 PPO 在初始化之外带来额外收益，且仍有碰撞与另一类训练停滞待解决。

## 1. 新电脑恢复

```bash
git clone --depth 1 --branch recovery https://github.com/gcusms/rl_learn_agv.git
cd rl_learn_agv
# 推荐先用 Ubuntu 22.04、Python 3.10 复现；不要复制旧 .venv。
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r training/requirements.txt
export PYTHONPATH=training:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# 输出目录须不存在。只检查，不连接电机、不训练。
python tools/check_handoff.py --output training/runs/recovery_check
```

CPU 离线检查不需要 CUDA、ROS 2 或显示器。建议预留至少 8 GB 安装空间，虚拟环境、下载缓存另计。`training/requirements.lock.txt` 是旧 ROS 主机记录，不是跨电脑安装清单。Jetson 先安装与其 JetPack 匹配的 PyTorch。

停车反馈包使用原生定位内核；平台二进制不上传，目标电脑重建。完整检查命令：

```bash
sudo apt-get install g++ libeigen3-dev
python tools/check_handoff.py --build-native --output training/runs/recovery_check_native
```

该入口检查文件哈希、四类 PPO、436/439 维合同、TorchScript、冻结基础 actor、目标方向及 100 项功能测试。实际执行结果见[VERIFICATION.json](docs/handoff/VERIFICATION.json)。这些是迁移与实现检查，不是实车精度或安全验收。

## 2. 模型选用

| 路径（相对仓库根目录） | 用途与结论 |
|---|---|
| [quality_recovery_20261014](models/quality_recovery_20261014) | 原实车部署基线，300 维，追加恢复训练 2k；历史开发集 46/48；本轮不替换 |
| [ray_consistent_candidate/portable](models/localization_mapping_20261015/ray_consistent_candidate/portable) | 最新射线一致性定位算法候选，303 维；尚非实车验收 |
| [ppo_4000_s221015](models/stop_feedback_20261015/ppo_4000_s221015) | 停车反馈 V1；回归退化，拒绝 |
| [ppo_v2_4000_s221015](models/stop_feedback_20261015/ppo_v2_4000_s221015) | V2 因果等待计时，独立 4k；回归退化，拒绝 |
| [ppo_explore_4000_s221015](models/stop_feedback_20261015/ppo_explore_4000_s221015) | V2 ＋有界温度探索，独立 4k；完整回归退化，拒绝 |
| [prefix_ppo_2000_s231015](models/stop_feedback_20261015/prefix_ppo_2000_s231015) | 最新：冻结原 actor＋学习恢复残差，示范初始化再做专项 PPO；14 例为 13 安全到达、1 碰撞，尚未部署验收 |
| [docking_selected_5000_20261015](models/docking_selected_5000_20261015) | 历史位置＋方向停靠模型；不能凭停靠成绩替换整机 |

原基线 `policy_2000.zip` SHA256：`7a3dad60f76060daf6b311b4bc6db921d3925ef0cdcf11683fe99e282173de11`。

最新恢复目录：`policy_migrated.zip` 为零残差初始化，`policy_bc.zip` / `bc_portable` 为示范初始化版，**`policy.zip` / `portable` 为最终 PPO 更新后产物**。`policy_step_1000.zip`、`policy_step_2000.zip` 是更新前回调检查点，只能用于对应采样概率审计。本次专项 PPO 已完成 **2,000 步**，总用时 **1,257.38 s（约 21 分钟）**，峰值 RSS **2,393.43 MiB**；另有 600 帧物理初始化不计入步数。训练统计不是验证成绩；资源与哈希见本次快照和模型 `status.json`。

**配套加载：**原基线 300 维，方向候选 303 维；停车反馈训练 436 维、方向运行时 439 维。根据 manifest 的 `required_runtime` 使用 `neural_navigation.goal_pose_runtime.resolve_runtime(...)`，先调用 `check_files(bundle, manifest)`。新恢复检查点用 `StopRecoveryPPO.load`，旧三种反馈模型用 `StopFeedbackPPO.load`。不能只换 `actor.pt` 或重命名合同；完整例子在[检查入口](tools/check_handoff.py)。

车体约 39.39×32.95 cm，保护余量 4 cm；12 动作，前进上限 0.20 m/s、转向上限 0.50 rad/s，不含横移或倒车。220 束雷达、约 219.44° 视野，保留后方盲区；最新定位按 10 m 有效范围处理。没有缩小车体或放宽安全条件。

## 3. 当前结果

### 普通导航：保留失败对照

| 同一 14 例开发集 | 原基线 | V2 4k | 温度探索 4k |
|---|---:|---:|---:|
| 安全到达 | 12 | 11 | 10 |
| 碰撞 | 2 | 3 | 3 |
| 超时 | 0 | 0 | 1 |
| 假到达 | 0 | 0 | 0 |

两种新训练都拒绝。探索版的 rotation 前缀在 20 s 接管后，有运动选项仍等待 120 s，最终 140 s 超时。报告：[V2](reports/stop_feedback_20261015/pilot_v2_4000_final/SUMMARY.json)、[探索版](reports/stop_feedback_20261015/pilot_explore_4000_final/SUMMARY.json)、[探索恢复](reports/stop_feedback_20261015/exploration_recovery_4000/RESULTS.json)。集合与历史 46/48 不同，不能合并统计。

### 新方法：冻结基础策略，学习恢复

旧 300 维 actor 冻结；增加 436 维输入、64×64 恢复网络。仅在旧策略选停、有运动选项、感知/预测有效且距目标超过 0.3 m 时影响选择，停车始终可选。旧策略正常行驶和近目标分支输出保留，没有按时间强制前进或指定方向。

先用一条经物理检查安全到达的干预轨迹提取 **17 状态：3 行驶、14 停车标签**初始化，再进行真实闭环掩码 PPO。半数回合实际重放 100 帧合法前缀，到 20 s 后交给模型；扫描、里程计、反馈、掩码与物理位姿逐帧核验，**初始化不计入 PPO 步数或模型导航成绩**。其余回合保留一般任务与原难例。

| 同一物理前缀，20 s 后接管 | 历史原策略参考 | 示范初始化版 | 最终 2k PPO |
|---|---|---|---|
| rotation_stall_t20（训练相关） | 140 s 超时；接管后最长主动等待 120 s | **45.8 s 安全到达**；最长主动等待 0 s | **45.8 s 安全到达**；最长主动等待 0 s |
| arc_stall_t20（同任务另一前缀） | 88.6 s 安全到达；最长主动等待 7.8 s | **46.2 s 安全到达**；最长主动等待 0 s | **46.2 s 安全到达**；最长主动等待 0 s |

总时间含 20 s 初始化。参考为已经验证的历史轨迹，未算作新的独立重复执行；两例同源，不是泛化或部署验收。BC 与最终 PPO 的动作、扫描、观测和物理轨迹逐帧一致，不能把这部分改善额外归因于 PPO。见[最终完整接管](reports/stop_feedback_20261015/prefix_ppo_recovery_2000_20261016/RESULTS.json)、[逐帧对照](reports/stop_feedback_20261015/prefix_recovery_comparison_20261016/RESULTS.json)、[历史参考](reports/stop_feedback_20261015/historical_recovery_reference/RESULTS.json)。

### 最新完整回归：同一 14 例、三种模型

| 固定开发集（已复用，非最终测试集） | 冻结原参考 | BC 初始化 | BC＋2k PPO |
|---|---:|---:|---:|
| 安全到达 | 12/14 | 13/14 | 13/14 |
| 碰撞 | 2 | 1 | 1 |
| 超时 / 假到达 | 0 / 0 | 0 / 0 | 0 / 0 |
| 其中方向停靠 | 6/6 | 6/6 | 6/6 |

`train_failure_stop_go_variation_0` 从原参考 12.6 s 碰撞改善为 33.6 s 安全到达；原有成功任务均保留。共同成功的 12 例，路程合计增加约 **0.21%**、时间减少约 **0.08%**，符合预先设置的开发集回归阈值。`motion_profile_015` 三种模型仍在 11.6 s 碰撞；因此扩大训练及部署门槛仍未通过。

BC 与 PPO 在全部 14 例的动作、传感器和物理轨迹一致。PPO 恢复参数确实更新，但对 2,000 个实际采样状态的确定性选择也未改变；不能仅凭权重变化宣称训练成功。详见[完整 42 条结果](reports/stop_feedback_20261015/prefix_recovery_threeway_20261016/RESULTS.json)、[配对门槛与逐帧对照](reports/stop_feedback_20261015/prefix_recovery_threeway_20261016/SUMMARY.json)、[学习诊断](reports/stop_feedback_20261015/prefix_learning_analysis_20261016/RESULTS.json)。

另外，4 个预先冻结的未训练条件中，原参考 3 例初始化后成功，1 例在 15.4 s、未到接管点前碰撞；3 个有效例均未形成持续等待，不能证明恢复泛化。候选对照尚未在本快照完成。[四例记录](reports/stop_feedback_20261015/recovery_holdout_reference_20261016/RESULTS.json)保留失败初始化。

另一条实际 PPO 随机训练轨迹已完整重放 700 帧，逐帧校验观测、掩码、命令和时间；最终 140 s 超时，距目标约 1.80 m，后段存在右弧可选而一直停车。它不同于同名任务的普通确定性执行，不能用后者成功掩盖这条失败。[重放证据](reports/stop_feedback_20261015/training_stall_source_20261016/RESULTS.json)仅用于诊断，尚非新模型恢复成绩。

### 定位和方向停车

| 22 条冻结轨迹、2855 帧同输入对照 | 时序定位父版 | 射线一致性候选 |
|---|---:|---:|
| 位置误差 P95 | 18.51 cm | 18.04 cm |
| 航向误差 P95 | 1.526° | 1.368° |
| 终点位置误差均值 | 3.42 cm | 3.27 cm |
| 本机 Python 定位 CPU P95 | 7.88 ms | 14.59 ms |

改善来自里程计传播、静态支持筛选、弱方向约束、时序融合与自由射线矛盾检查，**未训练新定位网络**。已有仿真回归改善小、耗时增加；实车录制无独立绝对位姿真值，不能声称超过 AMCL。恢复实验使用时序定位父包，未自动融合最新射线候选。

方向运行时接受 `(x,y,yaw)`，同位置不同方向可转向。开放区域、在线建图、搬车自动重定位及 Orin Nano 端到端耗时仍未验收。见[定位说明](docs/localization/README.md)、[实车输入分析](reports/localization_mapping_20261015/real_input_repair_20261015/REPORT.md)。

## 4. 下一步：先复评，不盲目增加步数

1. 在新机完成完整性检查并重建原生包；后台进程不随 Git 迁移。
2. 已完成的 14 例三模型回归及两个完整前缀无需因换机重复训练；若改变环境或代码，使用下面的相同任务复评。
3. 优先解决已复现的另一类训练停滞：相同物理前缀下，对照原参考与短右弧，确认是否存在完整、安全且当前残差可学习的恢复序列。保留全部失败，不把定时干预当作模型能力。
4. 完成 3 个可初始化的未训练条件候选对照，并保留第 4 例初始化失败；这些条件尚不足以覆盖长等待，需要另补未训练停滞家族。
5. 只有物理诊断确实安全，才用其接管后的合法后缀补充示范，再单独训练并检验普通导航、停车、碰撞和等待。新加载器的 `minimum_sensor_timestamp_s` 排除初始化段；原示范在 20 s 后仍保留 3 个前进、10 个停车状态。此时间仅选取离线样本，不进入策略输入或部署规则。
6. 处理共同碰撞 `motion_profile_015`、补充未见地图后，再考虑扩量。定位仍需独立实车参考与开放区域试验。详见[继续工作计划](reports/stop_feedback_20261015/prefix_recovery_comparison_20261016/PLAN.md)。

```bash
# 以下输出目录均须不存在；先安装编译依赖。
python tools/prepare_native_bundle.py \
  --source models/stop_feedback_20261015/baseline/portable \
  --output training/runs/resume_baseline
python tools/prepare_native_bundle.py \
  --source models/stop_feedback_20261015/prefix_ppo_2000_s231015/portable \
  --output training/runs/resume_candidate
python tools/prepare_native_bundle.py \
  --source models/stop_feedback_20261015/ppo_4000_s221015/initial_portable \
  --output training/runs/resume_reference
python tools/prepare_native_bundle.py \
  --source models/stop_feedback_20261015/prefix_ppo_2000_s231015/bc_portable \
  --output training/runs/resume_bc
# 按已完成实验复评：14 例×三个模型，共 42 条闭环。
nice -n 19 python -m experiments.evaluate_docking_navigation \
  --candidate training/runs/resume_candidate --baseline training/runs/resume_reference \
  --bc training/runs/resume_bc \
  --cases reports/stop_feedback_20261015/feedback_development_cases.json \
  --output training/runs/resume_paired --cpu-cores 0.35 \
  --candidate-name recovery_ppo --baseline-name frozen_reference
nice -n 19 python -m experiments.evaluate_feedback_recovery \
  --bundle training/runs/resume_candidate \
  --specifications reports/stop_feedback_20261015/frozen_recovery_cases.json \
  --output training/runs/resume_recovery --label recovery_ppo --cpu-cores 0.15
```

参考的 439 维原策略包为 `models/stop_feedback_20261015/ppo_4000_s221015/initial_portable`；接管评估不能直接用 303 维基线。`resume_baseline` 的 303 维父包仅用于下方独立重现训练。

```bash
# 参考已完成：3 个有效前缀；第 4 例初始化失败另有完整记录。
nice -n 19 python -m experiments.evaluate_feedback_recovery \
  --bundle training/runs/resume_candidate \
  --specifications reports/stop_feedback_20261015/recovery_holdout_reference_20261016/prefixes.json \
  --output training/runs/resume_holdout_candidate --label recovery_ppo --cpu-cores 0.15
# 另一个训练停滞的物理对照：严格核验前 106 s，不是模型成绩。
nice -n 19 python -m experiments.probe_recovery_prefix \
  --bundle training/runs/resume_reference \
  --prefix reports/stop_feedback_20261015/prefix_recovery_comparison_20261016/training_stall_prefix.json \
  --specifications reports/stop_feedback_20261015/prefix_recovery_comparison_20261016/training_stall_probe_windows.json \
  --output training/runs/resume_stall_probes --cpu-cores 0.15
```

参考生成中固定动作只是初始化；初始化失败、所有实际到达的状态都要统计，不得只选可恢复例。预先冻结记录：[HOLDOUT.json](reports/stop_feedback_20261015/prefix_recovery_20261015/HOLDOUT.json)。旧电脑正在进行的对照与队列状态已记入本次快照；未完成输出没有作为完整结果上传。下一台电脑应使用新的输出目录运行上述命令，不能假设后台服务已迁移。

```bash
# 审计真实 PPO 因果历史与更新前采样概率；与导航验收分开。
python -m experiments.audit_prefix_recovery_rollouts \
  --model-run models/stop_feedback_20261015/prefix_ppo_2000_s231015 \
  --output training/runs/resume_sampling_audit
# 独立重现：从原 300 维基线初始化，不是续跑 436 维权重。
nice -n 19 python -m experiments.train_prefix_recovery \
  --bundle training/runs/resume_baseline \
  --checkpoint models/quality_recovery_20261014/policy_2000.zip \
  --training-cases reports/localization_mapping_20261015/blind_repair_20261015/stop_go_training_cases.json \
  --prefixes reports/stop_feedback_20261015/prefix_recovery_20261015/training_prefixes.json \
  --demonstrations reports/stop_feedback_20261015/prefix_recovery_20261015/demonstration_sources.json \
  --output training/runs/resume_repeat --steps 2000 --seed 231015 --cpu-cores 0.35
```

**续训限制：** `StopRecoveryPPO.load(path, device='cpu')` 能恢复网络和优化器；当前 CLI 只提供 300 维初始化，未提供直接续训 436 维参数。继续更新最新权重前，应实现合同检查与环境重建的续训入口，记录环境重置、课程与累计步数。虽保存优化器、随机数及课程记录，但未保存完整实时物理环境，不承诺逐步精确续跑。不要直接把新模型传给上述 `--checkpoint`。

## 5. 保守资源限制

本机主训练软限 0.35 核、硬限半个核、单线程、低优先级、内存上限 4 GiB；加辅助评估软限合计约半核。新机训练速度需实测。硬限前缀如下，后接完整 `python -m ...` 命令：

```bash
systemd-run --user --unit=agv-recovery \
  --property=CPUQuota=50% --property=MemoryHigh=3G --property=MemoryMax=4G \
  --property=Nice=19 --working-directory="$PWD" \
  --setenv=PATH="$PATH" --setenv=PYTHONPATH=training:. \
  --setenv=OPENBLAS_NUM_THREADS=1 --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1
```

`CPUQuota=50%` 为半个 CPU 核。用 `systemctl --user status agv-recovery`、`journalctl --user -u agv-recovery` 检查；不要两台电脑同时向同一结果目录写入。

## 6. 数据、目录与未上传内容

| 路径 | 内容 |
|---|---|
| `training/neural_navigation/` | 定位、地图、路由、方向目标、运行时与保护 |
| `training/experiments/` | 训练、采样审计、闭环评估、恢复方法和测试 |
| `models/`、`reports/` | 模型、优化器、输入合同、来源与逐例证据 |
| `training/assets/localization_replay_20261015/` | 28 条冻结仿真输入，真值仅供监督/评估 |
| `training/assets/real_recording_20261014/` | 1516 条实车决策、1911 条反馈、轮速/IMU、map_01 PGM/YAML |
| `deployment/ros2_recording_adapter/` | 实车记录与模型接口，需接入实际 ROS 工程 |
| [measured_motion](deployment/ros2_recording_adapter/examples/measured_motion/README.md) | 编码器＋IMU、唯一 EKF 配置 |
| [FILES.json](docs/handoff/FILES.json) | 本次文件 SHA256 清单 |

**未上传完整 `record.zip`、原始 rosbag、相机图片、虚拟环境、缓存与平台 `.so`。** 精简定位回放数据已包含；若重新提取其他话题、分析原始 TF 或图像，请另行复制原始录包。见[实车数据清单](training/assets/real_recording_20261014/MANIFEST.json)。

```bash
# 固定输入定位回放，不更新模型、不改变车辆轨迹。
nice -n 19 python -m experiments.evaluate_real_input_repair \
  --cache training/assets/localization_replay_20261015 \
  --analysis training/assets/real_recording_20261014/analysis \
  --record training/assets/real_recording_20261014/record \
  --hardware models/quality_recovery_20261014/portable/hardware.json \
  --output training/runs/resume_localization_replay --cpu-cores 0.35
```

历史交接：[learn README](docs/handoff/LEARN_README.md)、[learn 检查](docs/handoff/LEARN_VERIFICATION.json)、[更早工作目录补丁](docs/handoff/legacy_wip/README.md)。历史文件内的“进行中”和绝对路径仅代表当时来源；当前进度以本次快照为准，不要用旧补丁覆盖当前主线。
