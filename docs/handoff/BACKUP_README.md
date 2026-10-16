# AGV 强化学习导航：换机继续开发

本次交接分支：**[`backup`](https://github.com/gcusms/rl_learn_agv/tree/backup)**。结果固定截至 **2026-10-16 09:38:03（北京时间）**；这是 `save` 分支之后的完整进度归档。已包含训练和运行源码、定位内核源码、地图、原实车模型、研究候选、优化器、随机状态、训练样本、精简实车记录，以及成功和失败轨迹。

**两组各 2,000 步 PPO 已完成，配套四类评估均已结束；新候选仍未通过部署升级要求。当前没有新一轮 PPO 在训练。** 另一组诊断 BC 的正常任务回归完成 8/18，剩余 10 项。原机后台进程不会迁移到新电脑，此后生成的结果也不属于本次固定快照。

精确状态见 [SNAPSHOT.json](reports/handoff_backup_20261016/SNAPSHOT.json)，文件哈希见 [FILES.json](docs/handoff/FILES.json)，完整性验证见 [VERIFICATION.json](docs/handoff/VERIFICATION.json)。08:40 的交接说明保留为 [SAVE_README.md](docs/handoff/SAVE_README.md)。历史报告中的“当前”均指各自截止时间。

## 1. 现在做到哪里

当前重点是解决“模型不知道为什么被停车”和“有合法运动动作却不尝试”。已经增加逐动作停车原因、动作可行性、雷达质量、预测风险和等待历史反馈。现结构冻结基础 actor、localizer、router 及终端方向停靠部分，只训练按条件启用的恢复 residual；追加恢复 PPO 不会直接训练定位器或路由器。

下表的四类任务分别统计，不能将 35 行合成独立泛化成功率。额外条件来自已使用地图；初始化历史恢复不是全程自主任务。

| 模型 | 完整任务 18 项：到达/碰撞/超时 | 原成功任务保留 | 原恢复历史 2 项 | W11 历史 3 项 | 额外条件 12 项：到达/碰撞/超时 |
|---|---|---|---|---|---|
| 参考 BC，2,089 标签、800 次监督更新 | 16 / 1 / 1 | **14/14** | 2 到达 | 1 到达、2 超时 | 7 / 5 / 0 |
| 新对照 PPO，独立 2,000 步 | 17 / 0 / 1 | **13/14**，W11 超时 | 2 到达 | 3 超时 | 7 / 4 / 1 |
| 新约束 PPO，独立 2,000 步 | 16 / 1 / 1 | **13/14**，`motion_profile_002` 超时 | 2 到达 | 1 到达、2 超时 | **11 / 1 / 0** |

配对依据见 [FULL35_PAIR.json](reports/stop_feedback_20261015/retention_followup_20261016/FULL35_PAIR.json)。两组 PPO 初始化、网络、课程和保护一致，主要对照是示范保留损失系数 `0` / `0.05`；长等待探索损失均关闭。两组各 2k 不能写成一个模型训练了 4k。

约束组在额外条件中改善明显，但仍损失原有成功任务，不能据此替换实车模型。6 项位置＋方向停靠回归保留；旧独立 8k 的完整任务为 16 到达、2 碰撞，原成功保留 13/14，也未升级部署。原实车质量恢复 2k 的历史开发集为 46/48，与本表集合不同，不能直接按分数排名。

另有 2,355 标签、800 次监督更新的诊断 BC：专项 **6/6 到达**；完整任务已得到 **8/18 到达**，其余 10 项尚未评估。8 项包含已完成专项中复用的 3 项和新测的 5 项。它在预设全标签检查中有 6 条旧动作不一致，原准入检查仍未通过；局部成功不代表验收完成。

## 2. 新增结论与尚未解决的问题

- 完整 W11：参考 BC 约 37 秒、4.62 米安全到达；对照 2k 超时 140 秒；约束 2k 约 44 秒、4.75 米安全到达。这只证明完整起点任务的表现，不能外推到停滞后的脱困。
- **实际停滞历史仍未解决。** W11 的 32 秒历史接管测试中，参考 BC 也超时到 140 秒；接管后 540 帧里，515 帧只剩停止可选、18 帧有运动可选但等待、7 帧发出非零动作。合法动作不等于能完成脱困，失败轨迹未用作成功示范。
- 135 条已有安全轨迹、20,898 条实际模型控制帧中，没有找到符合当前恢复分支条件的“此前合法等待至少 2 秒后运动”的样本。该结论仅覆盖报告列出的轨迹集合。
- 新增两条真实执行停车 2.4 秒后的物理续航探针，随后均由参考模型自主到达。仅 W11 探针包含可训练恢复分支内的长等待后运动样本；另一条未启用该分支。**对照推理发现三份现有模型在这两个释放状态都已选择运动，因此尚未复现实际卡住的问题。** 不能把容易状态的成功当作修复证据。
- 已整理 **2,096 条示范**：原 2,089 条逐字节保留，新增 7 条运动标签，包含 1 条长等待后运动，原 37 条必要停车标签保留，无相同输入的冲突标签。这里只完成数据评估，**未据此启动新 PPO 或发布新模型**。

证据：[停滞接管](reports/stop_feedback_20261015/reference_control_wait_t32_20261016/RESULTS.json)、[物理停车探针](reports/stop_feedback_20261015/feasible_wait_probe_navigation_20261016/RESULTS.json)、[示范与同输入推理](reports/stop_feedback_20261015/legal_wait_data_20261016/RESULTS.json)、[安全轨迹覆盖检查](reports/stop_feedback_20261015/safe_wait_complete_inventory_20261016/RESULTS.json)。

新电脑建议按以下顺序继续：

1. 完成诊断 BC 剩余 10 项正常任务，和参考 BC 配对比较原 14 项成功、碰撞、假到达、时间、路程和方向停靠；不要重复计算已结束任务。
2. 在实际失败状态分开统计：只剩停止、策略主动停车、外部保护停车。继续检查恢复分支启用、冻结基础 logits 与 residual 修正、实际参与更新的状态覆盖。
3. 找到“实际失败状态＋相同约束下完整安全恢复”的证据后，再做有针对性的示范初始化或小规模 PPO；保留正常任务和必要停车。现有两条容易停车探针只补覆盖，不能单独证明方向正确。
4. 通过旧成功与效率回归后再增大训练量。保持实车轮廓、动作集合、雷达盲区和保护参数，不用强制前进或放宽碰撞条件换取成功率。

## 3. 新电脑拉取、安装与完整性检查

建议先以 Ubuntu 22.04、Python 3.10、CPU 环境复现。以下在同一终端执行；私有仓库需要该电脑已有 GitHub 访问权限。

```bash
git clone --depth 1 --branch backup https://github.com/gcusms/rl_learn_agv.git
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

# 可选完整安装检查；输出目录必须不存在。
nice -n 19 python tools/check_handoff.py --output training/runs/backup_check
```

`check_transfer.py` 仅用 Python 标准库核对文件、模型与已结束轨迹，不启动训练或导航。`check_handoff.py` 重建定位参考包，检查 13 份反馈模型、优化器、输入合同与实现。算法源码、模型及其合同与 `save` 相同；此前干净导出验证的 **127 项实现检查通过、0 跳过**，证据保留在 [SAVE_VERIFICATION.json](docs/handoff/SAVE_VERIFICATION.json)。本次新增的是固定研究结果、交接说明，以及支持当前分支元数据的完整性检查器；没有宣称重新做了实车验收。

仓库文件约 1.2 GB；环境安装建议另预留至少 8 GB，下载缓存另计。先单任务、单线程、`nice -n 19`，闭环使用 `--cpu-cores 0.35`，并保留至少 4 GiB 可用内存。软 CPU 预算不等于系统硬配额；需要硬限制时可用 `systemd-run --user --scope -p CPUQuota=50% ...`。Jetson 应安装与 JetPack 匹配的 PyTorch，不能照搬桌面 CPU wheel。

虚拟环境、缓存与平台 `.so` 未迁移。闭环前由 `tools/prepare_native_bundle.py` 从已上传 C++ 源码重建需要的包；编译不是重新训练。射线一致性定位候选是 Python 实现，原生一致性检查使用独立的 temporal 参考包，检查器已处理两者区别。

## 4. 关键模型、源码和数据

下列路径均相对仓库根目录；权重、manifest、观测语义和运行配置必须配套使用。

| 路径 | 用途 |
|---|---|
| `models/quality_recovery_20261014/policy_2000.zip` | 原实车部署模型 |
| `models/stop_feedback_20261015/w11_recovery_bc800_s231015/` | 本轮参考 BC：`policy_bc.zip`、`bc_portable/`、优化器与示范 |
| `models/stop_feedback_20261015/w11_control2000_s231015/` | 对照 2k：最终 `policy.zip`、`portable/`、优化器和实际 rollout |
| `models/stop_feedback_20261015/w11_retained2000_s231015/` | 示范保留系数 0.05 的独立 2k，完整训练状态同上 |
| `models/stop_feedback_20261015/early_transfer_bc800_s231015/` | 2,355 标签诊断 BC，PPO 0 步，尚待正常任务回归 |
| `models/stop_feedback_20261015/proactive_augmented_ppo_8000_s231015/` | 历史独立 8k 模型及实际样本 |
| `models/localization_mapping_20261015/ray_consistent_candidate/portable/` | 独立定位算法候选，未合入本轮恢复对照 |
| `models/docking_selected_5000_20261015/` | 历史位置＋朝向停靠模型 |
| `training/neural_navigation/` | 定位、地图、路由、反馈、安全检查和方向目标运行时 |
| `training/experiments/` | 监督初始化、PPO、实际样本审计和闭环评估入口 |
| `training/assets/real_recording_20261014/` | map_01、1,516 条实车决策、1,911 条反馈与轮速/IMU 精简数据 |
| `training/assets/localization_replay_20261015/` | 28 条冻结仿真回放；真值仅供监督及评价 |
| `reports/stop_feedback_20261015/` | 训练协议、完整正负结果与轨迹；各子目录有独立完成状态 |
| `reports/handoff_backup_20261016/` | 本次快照、剩余任务和原机进程状态 |
| `deployment/ros2_recording_adapter/` | ROS 2 数据记录与模型集成接口 |

原部署 SHA256：`7a3dad60f76060daf6b311b4bc6db921d3925ef0cdcf11683fe99e282173de11`。其余当前模型哈希见 SNAPSHOT。`policy_bc.zip` 是监督初始化；最终 `policy.zip` 包含完整 PPO 更新。`policy_step_2000.zip` 在最后一次更新前保存，不能代替最终文件。恢复模型使用 `experiments.stop_recovery_policy.StopRecoveryPPO.load`，运行时按 manifest 的 `required_runtime` 解析。

## 5. 先继续剩余评估

激活环境并设置第 3 节环境变量后，在仓库根目录串行执行。输出目录必须不存在。剩余任务已按本次固定完成记录筛选到 [pending_diagnostic_cases.json](reports/handoff_backup_20261016/pending_diagnostic_cases.json)。

```bash
python tools/prepare_native_bundle.py \
  --source models/stop_feedback_20261015/early_transfer_bc800_s231015/bc_portable \
  --output training/runs/backup_diagnostic_native
nice -n 19 python -m experiments.evaluate_docking_navigation \
  --candidate training/runs/backup_diagnostic_native \
  --candidate-name early_transfer_diagnostic \
  --cases reports/handoff_backup_20261016/pending_diagnostic_cases.json \
  --output training/runs/backup_diagnostic_pending --cpu-cores 0.35
```

剩余 10 项完成后，与 [已结束原任务行](reports/handoff_backup_20261016/diagnostic_finished_original_rows.json) 按 `case_id` 合并，核对恰好 18 个唯一任务。参考组为 `reports/stop_feedback_20261015/w11_recovery_bc800_navigation_20261016/RESULTS.json`；配对函数为 `experiments.paired_navigation_gate.compare`。新结果写入新目录，保留本快照的未完成状态作为审计。

两组 2k 已有完整结果，无需先重跑。需要复现实验时，完整四类评估命令见 [SAVE_README 第 5 节](docs/handoff/SAVE_README.md)；将输出目录改为新的 `training/runs/backup_*`，避免覆盖。旧文档的结果状态和分支名只对应其历史快照。

## 6. 训练入口与续训边界

入口：`python -m experiments.train_prefix_recovery`；完整 2k 协议见 [PLAN.json](reports/stop_feedback_20261015/retention_pair_20261016/PLAN.json)，相对路径示例见 [历史详细指南](docs/handoff/WORK_README.md)。先重建父运行包，新实验使用新输出目录。

网络、优化器、RNG、示范、课程配置和实际 rollout 已保留，可以加载分析并设计新实验。当前 CLI **不支持精确接回同一半回合的物理状态、雷达历史和全部记忆**；`--reference-bc` 只核对 BC 初始化，不是任意最终 PPO 检查点的续训开关。独立初始化实验不能写成无缝追加旧模型步数。原机 systemd 的 PID、依赖和调度脚本只作审计；新机直接使用上述入口。

原机两组 2k 分别约 25.0 / 26.4 分钟，软预算 0.35 核，峰值 RSS 约 2.35 / 2.39 GiB。新机时间应实测；监督初始化、物理探针和评估不计为 PPO 步数。当前 2,096 条示范尚未用于新模型训练，后续须先验证实际困难状态的覆盖和回归。

## 7. 定位、停靠与实车合同

定位改进来自里程计传播、静态支持筛选、弱方向约束、时序融合与自由射线矛盾检查，本轮恢复 PPO 没有更新定位网络。独立定位对照在 22 条冻结轨迹、2,855 帧中，时序父版 / 射线候选的位置 P95 为 18.51 / 18.04 cm，航向 P95 为 1.526 / 1.368°，CPU P95 为 7.88 / 14.59 ms；收益有限、耗时增加，尚未合入恢复对照。源码和验证见 [定位说明](docs/localization/README.md)。

实车录包没有独立绝对位姿真值；尚未验证超过 AMCL、长期无漂移、任意搬车重定位或 Jetson Orin Nano 端到端时延。已知地图、边导航边建图、开放稀疏场景仍需分别验收。

保持约 39.39×32.95 cm 车体、4 cm 安全余量；12 个离散动作，无横移和倒车，最高前进 0.20 m/s、转向 0.50 rad/s。原始雷达 220 束、约 219.44°，保留后方盲区。恢复训练观测 436 维、带方向运行时 439 维，与原部署 300 维不同。位置＋朝向目标已接入，同位置不同朝向可转向，6 项方向停靠属于固定回归。

当前闭环主要使用带噪声运动反馈；若实车仍为指令积分里程计，不能认为条件已匹配。实车 6%～12% 无效束、连续缺失、历史帧时间、10/12 米量程差异、新增障碍未入静态图及下位机超时停车仍需真实采集和验证。

## 8. 未随 Git 迁移的内容

完整 `record.zip`、原始 rosbag/相机图像、虚拟环境、下载缓存和平台二进制未上传。当前分析所需的精简数据已包含；如果新机需要重新提取原始话题，请另复制原机 **`/home/ubuntu/rl_learn/record.zip`**，源清单见 [MANIFEST.json](training/assets/real_recording_20261014/MANIFEST.json)。

本次仅归档完整完成的轨迹和快照中已结束的评估行。原机后台评估可以继续，但之后的进度不会自动出现在这个分支。保留的旧绝对路径、PID、服务状态和计划哈希用于审计；换机运行以本文相对路径及新输出目录为准。
