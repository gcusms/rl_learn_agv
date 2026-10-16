# AGV 强化学习导航：当前进度与换机续接

交接分支：**[`continue`](https://github.com/gcusms/rl_learn_agv/tree/continue)**。本分支接续 `backup` 的 2026-10-16 09:38 快照，包含随后完成的代码、模型、训练状态、成功／失败轨迹和评估结果。**准确截止时间、原机任务状态及模型 SHA256 见 [SNAPSHOT.json](reports/handoff_continue_20261016/SNAPSHOT.json)**；更早文档中的“当前”仅指各自时间。

**最新候选已改善一个真实复现的停车状态；106 秒停滞仍未解决，尚未替换实车模型。目前没有新一轮 PPO 在训练，原机本轮物理评估也均已结束。** 换机后先检查文件与模型，再完成剩余回归；无需重复已经完成的全部训练。

## 1. 当前成果与边界

当前研究目标是解决“模型不知道为什么停车”和“有可行动作却不尝试”。观测已加入逐动作拒绝原因、雷达质量、预测风险及实际停车历史。恢复分支采用可训练 residual；基础 actor、localizer、router 和方向停靠部分冻结。**继续恢复训练不会直接改善冻结的定位网络或路由网络。**

| 版本 | 训练／数据 | 已完成的主要评估 | 主要限制 |
|---|---|---|---|
| 原实车质量恢复模型 | 历史 PPO 2k | 历史开发记录 46/48 | 与下面任务集不同，不可直接排名；保留原文件 |
| 参考 BC | 2,089 标签，800 次监督更新 | 完整任务 16/18 安全到达；原成功保留 14/14 | 额外条件 7/12，到达外有 5 次碰撞 |
| 对照 PPO | 从参考 BC 独立追加 2,000 步 | 完整任务 17/18；原成功保留 13/14 | W11 回退，未通过升级 |
| 示范保留 PPO | 独立追加 2,000 步，保留系数 0.05 | 完整任务 16/18；额外条件 11/12 | 丢失原成功任务，未通过升级 |
| 诊断父 BC | 2,355 标签，800 次监督更新 | **完整任务 18/18**；原恢复历史 2/2；额外条件 8/12；W11 历史 1/3 | 额外条件仍有 4 次碰撞，95/106 秒历史接管超时；原全标签准入失败记录保留 |
| **最新 actual95 候选** | 父 BC ＋66 条成功恢复决策，追加 **400 次监督更新，新增 PPO 0 步** | 95 秒接管安全到达；4 项旧任务全部安全、时间及路程与父版一致；额外停车 2.4 秒后仍自主到达 | **106 秒仍超时；完整 18 项中仅检查了 4 项，额外 12 项及方向停靠待回归** |

上述“额外条件”仍来自使用过的地图；历史接管是带固定初始化的恢复测试。不能合成独立泛化成功率，也不能据此宣称实车可部署或定位优于 AMCL。

完整证据：[两组 PPO 的 35 行分类对照](reports/stop_feedback_20261015/retention_followup_20261016/FULL35_PAIR.json)、[诊断父版完整 18 项](reports/stop_feedback_20261015/early_transfer_regression_20261016/RESULTS.json)、[诊断父版剩余分类](reports/stop_feedback_20261015/actual_stop_validation_20261016/RESULTS_main.json)。

## 2. 最新恢复实验具体改善了什么

在同一 W11 场景，先按原轨迹重建 475 帧传感器与动作历史，到 95 秒后由候选模型独立接管：

| 检查 | 最新候选 | 匹配追加 400 次更新、仍使用旧数据的对照 |
|---|---|---|
| 95 秒接管 | **114.8 秒到达，10.277 米**；接管后 19.8 秒；0 碰撞、0 外部保护接管 | 超时 140 秒；有运动可选却等待 170 帧 |
| 95 秒后实际再停车 2.4 秒，97.4 秒释放 | **138.0 秒到达，12.324 米**；释放后 40.6 秒；有运动可选却等待 1 帧 | 超时 140 秒；有运动可选却等待 148 帧 |
| 106 秒接管 | 超时，接管后 170 帧全部停车 | 超时，接管后 170 帧全部停车 |

第二行的初始化确实执行了零速度并更新传感器／物理状态；释放后没有插入专家动作。两组初始化已逐帧配对。它只检查复用任务的新等待时序，不是陌生地图泛化；路径也仍较长，尚不满足“最快、最短”的目标。

最新模型拟合 **2,421/2,421** 标签正确，原 2,355 条样本逐字节保留，55 条必要停车均保留。完整成功探针中提取了 66 条新增决策（64 条运动、2 条停车）；失败的 106 秒尝试未作为正示范。独立导出检查复算了 95 秒接管后全部 100 个动作，与保存的模型完全一致。

证据：[训练与来源](reports/stop_feedback_20261015/actual95_transfer_20261016/README.md)、[模型独立接管及 4 项回归](reports/stop_feedback_20261015/actual95_learned_validation_20261016/RESULTS.json)、[导出动作一致性](reports/stop_feedback_20261015/actual95_transfer_20261016/verified_actual95/RESULTS.json)、[等更新次数对照](reports/stop_feedback_20261015/actual95_matched_control_20261016/physical/RESULTS.json)、[2.4 秒停车配对](reports/stop_feedback_20261015/actual95_wait_verified_20261016/RESULTS.json)。

另有等待分布温度校准原型，仅完成离线分布／导出一致性检查，没有用于上述候选、闭环、PPO 或部署；不要与本次新增示范的收益混为一项。

另有 106 秒近场记忆的传感器回放：原拒绝动作得到复现，3/4 厘米连续自由证据参数也未立即打开有效脱困通道。仅有当前扫描时部分动作看似可行，不能据此删除历史障碍记忆。该项没有修改运行时、放宽保护或产生新物理训练步，见 [审计结果](reports/stop_feedback_20261015/actual106_memory_replay_20261016/RESULTS.json)。

## 3. 新电脑拉取与安装

建议先在 Ubuntu 22.04、Python 3.10、CPU 环境复现。私有仓库需要新电脑已有 GitHub 访问权限。

```bash
git clone --depth 1 --branch continue https://github.com/gcusms/rl_learn_agv.git
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
nice -n 19 python tools/check_continue_models.py
```

`check_transfer.py` 仅依赖 Python 标准库，核对文件、快照、模型与完成轨迹的哈希；`check_continue_models.py` 再加载新增两份模型和优化器，核对训练／动作合同、原样本、必要停车和导出动作一致性。**两者都不启动训练或物理导航。** 本次两份新增检查点的原样本、优化器和共 192 条导出动作接口检查通过；还从已上传源码重建定位库并通过一帧 439 维运行时检查，见 [模型检查](reports/handoff_continue_20261016/MODEL_CHECK.json) 与 [重建检查](reports/handoff_continue_20261016/NATIVE_SMOKE.json)。 此前算法实现 127 项检查通过的记录保留在 `docs/handoff/SAVE_VERIFICATION.json`；该历史记录不代表本次重新进行实车验收。

仓库约 1.3 GB；新机建议另留至少 8 GB 安装环境及缓存。先单进程、单线程、`nice -n 19`，闭环软预算 **0.35 核**，至少保留 4 GiB 可用内存。需要硬限制时，在相关命令前加 `systemd-run --user --scope -p CPUQuota=50% -p MemoryMax=4G`。Jetson 应使用与 JetPack 匹配的 PyTorch，不能直接使用桌面 CPU wheel；目标板端 P95/P99 尚待实测。

虚拟环境、缓存和平台 `.so` 不随 Git 迁移。运行完整导航前，从上传的 C++ 源码重建本机定位包：

```bash
python tools/prepare_native_bundle.py \
  --source models/stop_feedback_20261015/actual95_transfer_bc400_s231016/bc_portable \
  --output training/runs/continue_latest_native
```

输出目录必须不存在；该命令生成新包和对应 manifest，不修改已归档模型。编译用于匹配新机架构，不是重新训练。历史计划中的原机绝对路径、PID 和 `.so` 哈希只作审计；换机应使用相对路径及重建后的新包。

## 4. 关键模型、代码与数据位置

| 相对路径 | 用途 |
|---|---|
| `models/quality_recovery_20261014/policy_2000.zip` | 原实车部署模型，SHA256 `7a3dad60f76060daf6b311b4bc6db921d3925ef0cdcf11683fe99e282173de11` |
| `models/stop_feedback_20261015/actual95_transfer_bc400_s231016/` | **最新研究候选**；`policy_bc.zip`、`bc_portable/`、示范、优化器、RNG、配置及来源 |
| `models/stop_feedback_20261015/actual95_control_bc400_s231016/` | 相同额外更新次数、旧数据的公平对照 |
| `models/stop_feedback_20261015/early_transfer_bc800_s231015/` | 完整任务 18/18 的诊断父版；仍有额外条件失败 |
| `models/stop_feedback_20261015/w11_recovery_bc800_s231015/` | 参考 BC |
| `models/stop_feedback_20261015/w11_{control,retained}2000_s231015/` | 两组独立 2k PPO；最终 `policy.zip`、优化器、实际 rollout |
| `models/localization_mapping_20261015/ray_consistent_candidate/portable/` | 独立射线一致性定位候选，未混入恢复对照 |
| `models/docking_selected_5000_20261015/` | 位置＋目标朝向停靠模型 |
| `training/neural_navigation/` | 地图、定位、路由、安全与预测、停车反馈、方向目标运行时 |
| `training/experiments/` | BC／PPO、回放审计、配对及闭环评估入口 |
| `training/assets/real_recording_20261014/` | map_01、1,516 条实车决策、1,911 条反馈及精简轮速／IMU 数据 |
| `training/assets/localization_replay_20261015/` | 28 条冻结仿真回放；真值仅供训练监督和评价 |
| `reports/stop_feedback_20261015/` | 正负实验、来源、完整轨迹及各自完成状态 |
| `reports/handoff_continue_20261016/` | 本次快照、已完成 4 项及待评估 14 项清单 |
| `deployment/ros2_recording_adapter/` | ROS 2 记录与接入接口；控制不依赖 Nav2 行为树 |

最新 `policy_bc.zip` SHA256：`4141a52dcb294e9394af893324d77d89d258a5c5bda27962efde0b16d69b42ed`。所有主要模型哈希见快照。

`policy_bc.zip` 为监督初始化；PPO 最终产物为 `policy.zip`。`policy_step_2000.zip` 在最终更新前保存，不能代替最终文件。恢复检查点用 `experiments.stop_recovery_policy.StopRecoveryPPO.load(..., device='cpu')`，运行时严格按配套 manifest 加载。

## 5. 换机后下一步：先补齐回归

1. 最新候选剩余完整任务 **14 项**，不要重跑并重复计数已完成 4 项。待评估清单已经固定，见下方命令。
2. 再检查额外条件 12 项、原恢复历史 2 项和方向停靠 6 项。保留 95/106 秒及 2.4 秒停车对照，按类别报告到达、碰撞、超时、假到达、路程与时间。
3. 106 秒状态先证明存在“相同约束下完整安全脱困”的可行经验；5 种显式动作探针均未成功，合法动作不等于有效脱困。不要给失败片段打成功标签或仅增加 PPO 步数。
4. 额外条件 12 项中含 4 项方向目标，另加原方向探针 2 项，共同组成 6 项停靠回归。回归通过后，再设计小规模、单主要改动的 PPO；最后测试未使用地图和实车退化数据。不能缩小轮廓、放宽掩码、强制前进或无视盲区换取成功率。

激活第 3 节环境并重建包后，在仓库根目录串行执行，输出目录必须不存在：

```bash
nice -n 19 python -m experiments.evaluate_docking_navigation \
  --candidate training/runs/continue_latest_native \
  --candidate-name actual95_transfer_bc400 \
  --cases reports/handoff_continue_20261016/pending_navigation_cases.json \
  --output training/runs/continue_remaining14 --cpu-cores 0.35

# 独立重现 95/106 秒接管；不要与上面的物理评估并行。
nice -n 19 python -m experiments.evaluate_feedback_recovery \
  --bundle training/runs/continue_latest_native \
  --specifications reports/stop_feedback_20261015/actual95_learned_validation_20261016/prefixes.json \
  --label actual95_transfer_bc400 \
  --output training/runs/continue_actual_stalls --cpu-cores 0.35
```

剩余 14 项与 `reports/handoff_continue_20261016/finished_navigation_rows.json` 按 `case_id` 合并，确认恰为 18 个唯一任务。父版配对依据为 `reports/stop_feedback_20261015/early_transfer_regression_20261016/RESULTS.json` 的 `candidate_rows`，配对函数为 `experiments.paired_navigation_gate.compare`。其余任务入口见 [NEXT_STEPS.json](reports/handoff_continue_20261016/NEXT_STEPS.json)。

## 6. 续训入口与注意事项

PPO 入口为 `python -m experiments.train_prefix_recovery --help`；完整旧 2k 协议保留在 `reports/stop_feedback_20261015/retention_pair_20261016/PLAN.json`，最新 2,421 条示范及来源在 `actual95_transfer_bc400_s231016/demonstrations.npz` 和 `reports/stop_feedback_20261015/actual95_transfer_20261016/demonstrations.json`。

已保存网络、优化器、随机状态、课程配置与实际样本。当前训练 CLI **不能直接恢复任意最终检查点的同一半回合物理状态、扫描历史及全部运行记忆**；`--reference-bc` 是初始化一致性检查，不是任意 PPO 检查点续训开关。载入文件用于设计后续实验，不应把重新初始化的独立实验写成无缝追加旧训练步数。历史 `reports/**/run.py` 固定了输出及服务依赖，是可审计实验源码；不要在原目录直接重跑，换机优先使用上面的通用入口。

原机两组 2k 各约 25／26 分钟（0.35 核），峰值 RSS 约 2.4 GiB；最新 400 次监督更新约 58 秒、686 MiB。监督更新、探针、初始化帧和评估均不算 PPO 步数；两组各 2k 也不等于一份模型训练 4k。

## 7. 定位、实车与停靠状态

定位优化主要来自里程计传播、静态支持筛选、弱方向约束、时序融合和自由射线矛盾检查；不是本次恢复 PPO 学出来的。独立冻结回放 22 条、2,855 帧上，时序父版／射线候选的位置 P95 为 **18.51／18.04 cm**，航向 P95 为 **1.526／1.368°**，CPU P95 为 **7.88／14.59 ms**。精度收益有限，耗时增加；详见 [定位算法及验证](docs/localization/README.md)。

实车录包没有独立绝对位姿真值。已知地图、边导航边建图、稀疏开放区域、搬车重定位应分别验收；未证明超过 AMCL、长期无漂移或适配 Jetson 全程延迟。6%～12% 无效束、连续盲区、10/12 米量程、0.1/0.2 秒历史帧、新增障碍未入静态图及指令积分误差仍要依据实测验证。

当前车体约 **39.39×32.95 cm**，4 cm 安全余量；220 束原始扫描、约 219.44° 视野，保留后方盲区。12 个离散 `(vx,wz)` 动作，无横移、无倒车；最高前进 0.20 m/s、角速度 0.50 rad/s。恢复训练 436 维，带方向运行时 439 维，原部署 300 维；模型、特征和合同必须配套。位置＋目标朝向停靠已支持，同位置不同朝向可以转向；最新恢复候选仍需停靠回归。

## 8. 快照范围与原机任务

本分支保留已完成轨迹及成功／失败记录；本次捕获时，原机恢复验证、匹配对照与近场记忆审计均已结束；没有在运行的 PPO 或物理评估。后台进程、systemd 状态和 PID 不会迁移，也不会在新电脑自动恢复。截止时间之后生成的文件不会自动更新 GitHub。

完整 `record.zip`、原始 rosbag／相机图像、虚拟环境、缓存、平台二进制未上传；本轮分析需要的精简实车数据已包含。若要重新提取原始话题，另行复制原机 `/home/ubuntu/rl_learn/record.zip`，源哈希和内容索引见 `training/assets/real_recording_20261014/MANIFEST.json`。

[上一份 09:38 交接说明](docs/handoff/BACKUP_README.md) 保留历史状态，不作为本分支最新进度。换机以本页、当前快照和各报告 `completed` 字段为准。
