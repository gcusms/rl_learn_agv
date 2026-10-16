# AGV 强化学习导航：换电脑继续开发

交接分支：**[handoff3](https://github.com/gcusms/rl_learn_agv/tree/handoff3)**。进度冻结于 **2026-10-16 14:06:06（北京时间）**，接续 `handoff2`，包含导航与定位源码、原实车及研究模型、优化器与随机状态、必要地图、精简实车数据、成功和失败的评估轨迹。

**当前没有正在运行的训练或物理评估。最新候选在原 4 项动态任务中 4/4 安全到达，但在另外 4 个噪声种子下只有 1/4 到达、3 次碰撞，未通过验收。原实车模型未替换；“最新”不等于“全面最好”。** 后续微调新增 PPO 为 0 步。

以本页、[本次快照](reports/handoff3_20261016/SNAPSHOT.json)和[下一步任务](docs/handoff/NEXT_STEPS.md)为当前状态。旧报告中的“运行中”“排队中”“未上传”仅描述各自生成时刻，不能当作新电脑的实际进程状态。

## 1. 新电脑恢复步骤

建议先用 Ubuntu 22.04、Python 3.10、CPU 单线程复现。私有仓库需要你的 GitHub 访问权限。

```bash
git clone --depth 1 --branch handoff3 https://github.com/gcusms/rl_learn_agv.git
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

第一项只用 Python 标准库核验文件、模型和完成轨迹哈希；第二项加载五份近期检查点、Adam 和随机状态，复算示范、动作掩码及导出结果，不训练、不运行物理导航。见[文件检查](docs/handoff/HANDOFF3_TRANSFER_CHECK.json)和[模型检查](reports/handoff3_20261016/MODEL_CHECK.json)。

平台相关 `.so` 不随 Git 迁移。完整运行前，用已上传的 C++ 定位源码生成独立包：

```bash
python tools/prepare_native_bundle.py \
  --source models/stop_feedback_20261015/recovery_followup_bc400_s231016/bc_portable \
  --output training/runs/handoff3_native
```

输出目录必须不存在；编译适配新硬件，不重新训练、不改归档权重。内核在上一份交接中已重编译并通过 439 维接口烟测，见[历史检查](reports/handoff2_20261016/NATIVE_SMOKE.json)，不代表 Jetson 已验收。Jetson 应使用与 JetPack 匹配的 PyTorch，不能照搬上述 x86 CPU 安装命令。

仓库文件约 1.3 GB，建议另留至少 8 GB 安装环境和缓存。保持单线程、单物理主任务、Nice19、软预算 **0.35 核**。可用以下有硬限制的命令复现一项既有正常任务；它是环境检查，不是新的泛化成绩：

```bash
systemd-run --user --scope -p CPUQuota=50% -p MemoryHigh=3G -p MemoryMax=4G \
  nice -n 19 python -m experiments.evaluate_docking_navigation \
  --candidate training/runs/handoff3_native \
  --candidate-name recovery_followup_bc400 \
  --cases reports/handoff2_20261016/smoke_cases.json \
  --output training/runs/handoff3_smoke --cpu-cores 0.35
```

## 2. 模型选择与比较

研究目录均位于 `models/stop_feedback_20261015/`。各列任务不同，不能跨任务集直接排序。

| 模型 | 目录 | 已完成结果 | 限制 |
|---|---|---|---|
| 原实车版 | `models/quality_recovery_20261014/policy_2000.zip`，相对仓库根目录 | 部署原文件保留 | 未替换，作为实车回退基线 |
| 保留式 2k PPO | `w11_retained2000_s231015/` | 常规 16/18；额外动态 11/12 | 额外组较好，常规仍有碰撞、超时；不是全面最优 |
| **完成较完整回归的研究参考** | **`recovery_followup_bc400_s231016/`** | **常规 18/18、额外动态 8/12；原恢复 2/2、方向探针 2/2** | 4 次动态碰撞；额外组相对诊断父版路程 +2.0113%，略超 2% 门槛；106 秒停滞未解决 |
| 提前让行第一版 | `early_yield_bc400_s231016/` | 原 4 条件中 2/4 到达、2 次碰撞 | 400 次监督更新，未完成全面回归 |
| **最新候选** | **`early_yield_remaining_bc600_s231016/`** | 原 4 条件 **4/4**；另 4 噪声条件 **1/4**、3 次碰撞 | 600 次监督更新，泛化门槛失败，扩大回归跳过 |

关键检查点 SHA256：

```text
原实车 policy_2000.zip
7a3dad60f76060daf6b311b4bc6db921d3925ef0cdcf11683fe99e282173de11

recovery_followup_bc400_s231016/policy_bc.zip
dd29ed24e25191af8265d832dd0edda072160d1204f4a18b280dbf265eb3e212

early_yield_bc400_s231016/policy_bc.zip
1b55a06047a25f45e4290ea294ca8bf8fe9310c38d1e5517da4ad5765a43ffd2

early_yield_remaining_bc600_s231016/policy_bc.zip
d736f543da3b672b20b63b61b1943b114851cd66d951a5534658bee10ea0c143
```

目录保留检查点、`bc_portable/`、示范、配置、Adam／随机状态和来源。近期 BC 检查点用 `experiments.stop_recovery_policy.StopRecoveryPPO.load(..., device='cpu')` 加载；运行使用配套包及 manifest，不能仅替换 `actor.pt`。新增监督更新不能计作 PPO 步数。

## 3. 最新进度与失败原因

取得了 3 条完整安全的提前让行教师轨迹。教师只根据当时雷达跟踪、里程计、反馈速度和掩码选择停车，连续 3 帧确认风险解除后交回模型，不强制运动。5 秒预测仅用于研究教师；部署 guard、车体、4 cm 余量和后方盲区记忆保持不变。真值只供独立评分和示范准入，不进入教师或模型输入。

第一版新增 133 条示范、400 次监督更新；最新一版再增加 277 条、600 次更新，原 2,638 条输入／掩码／标签及 95 条停车保留。**全部 2,915 条训练标签正确，但闭环噪声测试仍失败。** 基础 actor、localizer、router、方向停靠均冻结，新增 PPO 为 0。

| 同图新噪声条件 | 第一版 `1b55` | 最新 `d736` |
|---|---|---|
| `02_0` | 10.0 秒碰撞 | 10.2 秒碰撞 |
| `02_1` | 10.0 秒碰撞 | 10.4 秒碰撞 |
| `05_0` | 140 秒超时 | 14.4 秒碰撞 |
| `05_1` | 38.2 秒安全到达 | 34.0 秒安全到达 |

噪声种子在拟合前冻结、未进入训练，仍属于同一开发地图，不是陌生地图泛化。最多 16 项的有界验证在完成 12 项后按门槛停止；剩余 4 项及扩大回归均未执行，不能记为全部完成或验收通过。

103 帧回放逐帧复现了观测、动作、执行和定位。两个失败条件分别在 3.6／5.2 秒已有“运动预测会接触、停车未预测接触”的信号，恢复分支已启用，但停车概率只有约 `1.76e-5`／`1.02e-4`。下一步应验证逐动作预测风险和时间历史能否改善决策，不直接增加相同小样本更新。

证据：[新增来源准入](reports/stop_feedback_20261015/early_yield_remaining_fit_20261016/SOURCE_ADMISSION.json)、[拟合](reports/stop_feedback_20261015/early_yield_remaining_fit_20261016/RESULTS.json)、[12 项闭环结果](reports/stop_feedback_20261015/early_yield_remaining_validation_20261016/RESULTS.json)、[扩大回归跳过记录](reports/stop_feedback_20261015/early_yield_remaining_broad_20261016/RESULTS.json)、[103 帧回放](reports/stop_feedback_20261015/early_yield_holdout_replay_20261016/RESULTS.json)。

## 4. 换机后接着做什么

先完成环境检查，再按[详细任务](docs/handoff/NEXT_STEPS.md)推进：

1. 验证批量风险预测与原参考实现逐帧等价，并测 CPU P95/P99。`training/neural_navigation/long_motion_risk.py` **仅为未验证的独立原型，尚未接入任何模型或运行时，不代表性能已提升**。
2. 先冻结新测试种子，再设计逐动作风险及固定物理间隔的时间历史，版本化训练、运行、导出与 manifest。481 维运行时／478 维训练输入只是待验证设计，尚无该维度模型；旧模型不能直接接新输入。
3. 仅用完整安全来源补训小型风险感知分支，保留正常导航、必要停车和恢复运动。已经分析过的 4 个噪声失败条件属于开发集，不再作为未触碰的最终测试集。
4. 纯模型无教师通过小门槛后，完成常规 18 项、额外动态 12 项、90／95 秒恢复、额外等待、原恢复与目标朝向回归。安全与效率都通过，再决定是否进行 PPO。

106 秒停滞仍无完整安全恢复证据。动作合法、局部转动、零碰撞超时均不算成功，失败轨迹不作正示范，不能清空盲区记忆、缩小车体或降低 4 cm 余量。

历史 `reports/**/run.py`／`replay.py` 常含原机 systemd 依赖、固定输出目录和哈希，**不要原地直接重跑**。新机需建新目录、用重建包、以完成结果验收替代服务等待、重新固定输入哈希。Adam／RNG 已保存，但未保存全部半回合物理状态与扫描历史；当前 CLI 不支持任意检查点无缝续训，`--reference-bc` 也不是通用恢复开关。

## 5. 定位、地图、实车条件和目录

| 路径 | 内容 |
|---|---|
| `training/neural_navigation/` | 定位、路由、雷达、安全、停车反馈、朝向运行时 |
| `training/experiments/` | BC／PPO、回放、固定任务与配对验收 |
| `docs/localization/README.md` | 定位改进、编译、已知地图／在线建图及验证 |
| `models/localization_mapping_20261015/ray_consistent_candidate/portable/` | 独立定位候选，未混入本次恢复对照 |
| `models/docking_selected_5000_20261015/` | 位置＋目标朝向停靠模型 |
| `training/assets/real_recording_20261014/` | map_01、1,516 条实车决策、1,911 条反馈及精简轮速／IMU |
| `training/assets/localization_replay_20261015/` | 定位回放及独立评价输入 |
| `deployment/ros2_recording_adapter/` | ROS 2 录制与接入，导航控制不依赖 Nav2 行为树 |
| `reports/stop_feedback_20261015/` | 正负实验、完整轨迹和数据来源 |

定位改善主要来自里程计传播、静态支持筛选、弱方向约束、时序融合与自由射线检查，不是本轮恢复 BC 更新定位网络。独立 22 条、2,855 帧回放中，时序父版／射线候选位置 P95 为 18.51／18.04 cm，CPU P95 为 7.88／14.59 ms：精度小幅改善、计算时间增加。实车没有独立绝对位姿真值，尚未证明超过 AMCL；开放稀疏区、边建图边导航和搬车重定位仍需独立验收。

当前恢复模型车体约 39.39×32.95 cm，安全余量 4 cm；220 束、约 219.44° 雷达视野，保留后方盲区。12 个离散 `(vx,wz)` 动作，无横移、无倒车，最高 0.20 m/s、0.50 rad/s。恢复训练 436 维、含朝向运行时 439 维、原实车版 300 维，权重与合同必须配套。支持位置＋朝向目标及原地转向停靠，最新候选尚未完成该项回归。

原始 `record.zip`、完整 rosbag／相机、虚拟环境、缓存和平台 `.so` 未上传。本轮所需精简数据已包含；重新提取完整话题时需另行复制原机 `/home/ubuntu/rl_learn/record.zip`，不能从精简数据恢复原始录像。

## 6. 交接完整性

[新增文件来源](reports/handoff3_20261016/UPDATE_CAPTURE.json)、[源机进程状态](reports/handoff3_20261016/SOURCE_PROCESS.json)、[全量哈希](docs/handoff/FILES.json)用于确认换机版本。上一份说明保存在[历史 README](docs/handoff/HANDOFF2_README.md)。本分支为开发交接快照，研究目标仍未完成，不自动替换实车模型。
