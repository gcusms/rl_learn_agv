# AGV 强化学习导航：换机交接

本次新分支：**[`save`](https://github.com/gcusms/rl_learn_agv/tree/save)**。结果固定截至 **2026-10-16 08:40:06，北京时间**。包含当前训练与运行代码、定位内核源码、地图、原部署模型、研究候选、优化器、实际训练样本、精简实车数据，以及成功和失败轨迹。

**当前两组各 2,000 步 PPO 已训练完成，后续评估尚未全部完成。原实车部署模型保留，新候选尚未通过部署验收。** Git 保存的是这个时间点的文件；原电脑的进程及之后生成的结果不会自动随分支迁移。

精确进度见 [SNAPSHOT.json](reports/handoff_save_20261016/SNAPSHOT.json)，固定评估见 [validation](reports/handoff_save_20261016/validation/)，全文件哈希见 [FILES.json](docs/handoff/FILES.json)。上一版 `dev` 的说明、哈希和检查分别归档到 [DEV_README.md](docs/handoff/DEV_README.md)、[DEV_FILES.json](docs/handoff/DEV_FILES.json)、[DEV_VERIFICATION.json](docs/handoff/DEV_VERIFICATION.json)。历史文档中的“当前”指其各自截止时间。

## 1. 当前做到哪里

当前要解决的是：模型能区分停车原因，在有安全可行动作时合理恢复，同时保留避障、路径决策和位置＋方向停靠。结构为冻结基础 actor、localizer、router 和终端方向停靠部分，只训练有启用条件的恢复 residual。追加恢复 PPO 不会直接更新定位器或路由器。

| 模型 | 截止时的闭环结果 | 当前判断 |
|---|---|---|
| 原实车质量恢复 2k | 历史开发集 46/48；与本轮集合不同 | 保留为部署基线 |
| 参考 BC：2,089 标签、800 次监督更新 | 完整 18 项：16 到达、1 碰撞、1 超时；原成功 14/14；6 项方向停靠保留 | 本轮配对参考；新增 PPO 为 0 |
| 旧独立 8k PPO | 完整 18 项：16 到达、2 碰撞；原成功只保留 13/14 | 存在退化，未选作部署升级 |
| 新对照 PPO：2,000 步 | 完整 18 项：17 到达、0 碰撞、1 超时；原成功 13/14 | W11 原成功任务退化为超时，未通过保留要求 |
| 新约束 PPO：2,000 步 | **仅完成 16/18 项**：14 到达、1 碰撞、1 超时 | W11 完整任务恢复到达，但 `motion_profile_002` 超时；不能宣称整体更好 |
| 新诊断 BC：2,355 标签、800 次监督更新 | 专项 **仅完成 5/6 项**，已结束的 5 项均到达 | 未通过原预设全标签保留条件；独立诊断不等于准入或部署验收 |

两组新 PPO 的 BC 初始化、网络、优化器、种子、课程和保护相同，主要变量是示范保留损失系数 `0` / `0.05`；两组的长等待探索损失都关闭。首 250 个实际 rollout 完全一致。两个各 2k 模型不能合称单个模型训练 4k；训练中到达比例也不能当作独立评估成功率。

对照组的固定评估已完成，须分开阅读：

| 类别 | 完成行数 | 安全到达 | 碰撞/接触事件 | 超时 |
|---|---:|---:|---:|---:|
| 完整起点任务 | 18 | 17 | 0 | 1 |
| 原恢复历史 | 2 | 2 | 0 | 0 |
| W11 相关历史 | 3 | 0 | 0 | 3 |
| 额外同地图条件 | 12 | 7 | 4 | 1 |

这 35 行混合完整任务、相关前缀和同图路线，不能合并为独立泛化成功率。约束组的另 2 个原前缀、3 个 W11 历史和 12 条额外条件在本快照中尚未开始。新诊断 BC 的参考组 3 项和 32 秒恢复专项也尚未开始。

W11 从完整起点运行：参考 BC 安全到达约 37 秒、4.62 米；新对照超时 140 秒；新约束安全到达约 44 秒、4.75 米。这说明该完整任务有改善，不能据此认为“已经停很久后也能脱困”。逐项进度以快照为准。

## 2. 已查明的问题与接着做什么

已确认两类等待：有些状态只剩停止合法，需要检查感知、规划和动作可行性；另一些状态有合法运动动作，但策略依然偏向停止。逐动作反馈必须分别检查，不能只看最终执行停车的一个原因。

同输入审计中，近场转向诊断的 170 帧全部有运动可选、恢复分支启用，模型仍全部停止，最低停止概率约 0.9999988。已有示范缺少此前合法等待至少 2 秒后的有效恢复样本；对照 2k 的实际训练也未覆盖该类可训练状态。约束 2k 覆盖了 66 帧，但全部采样停止。仅增加当前课程步数不能保证补齐经验。

有限时段的合法右转物理试探没有碰撞，但仍超时，不把它贴成成功示范。训练碰撞已精确复现 25 帧，记录为原保守接触失败；没有通过修改判定把它重算为成功。详见 [原因诊断及证据](reports/stop_feedback_20261015/stop_support_diagnosis_20261016/README.md)。

换机后的推荐顺序：

1. 完成约束模型固定评估，与参考 BC、对照模型比较原成功任务、方向停靠、碰撞、等待、时间和路程。
2. 完成诊断 BC 与参考组的既定专项；它存在 6 条旧示范动作不一致，局部成功不代替正常导航回归。
3. 执行已固定的 W11 32 秒参考 BC 接管，验证“合法动作是否真正有用”；结果未知，不能提前当成功训练数据。
4. 有完整安全恢复证据后，再补相应历史的示范和实际 PPO 状态覆盖，进行单项对照。长等待探索损失默认关闭，先单独实验，不同时修改保护和奖励来归因。
5. 保留原 14 项成功、6 项方向停靠和合理停车，检查新增碰撞及假到达，再决定是否增大训练量。

## 3. 新电脑安装与检查

建议先在 Ubuntu 22.04、Python 3.10、CPU 环境复现。以下在同一终端执行；先用标准库核对文件，再安装依赖。

```bash
git clone --depth 1 --branch save https://github.com/gcusms/rl_learn_agv.git
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

# 可选完整安装检查：模型加载、输入合同、实现检查；输出目录须不存在。
nice -n 19 python tools/check_handoff.py --output training/runs/save_check
```

仓库文件约 1.1 GiB，环境安装建议另预留至少 8 GB，下载缓存另计。新机先单任务、单线程、`nice -n 19`、软预算 **0.35 核**，保留至少 4 GiB 可用内存。Jetson 使用与 JetPack 匹配的 PyTorch，不直接安装上述桌面 CPU 包；CUDA 不是复现检查的前提。

虚拟环境、缓存和平台 `.so` 不迁移。C++ 定位内核源码已包含，闭环运行前按下一节重新编译；编译不等于重新训练。`check_transfer.py` 只验证文件和完成状态，不启动训练或导航。`check_handoff.py` 会先重建一个定位检查包，再检查 13 份反馈模型、优化器、样本和实现，避免依赖旧机未上传的 `.so`。先前同一训练源码的 127 项实现检查全通过，记录保留在 DEV_VERIFICATION 中；本次换机检查见 [VERIFICATION.json](docs/handoff/VERIFICATION.json)，不能代替实车验收。加 `--build-native` 可进一步重建全部反馈包。

## 4. 模型及代码位置

所有路径相对仓库根目录。权重、manifest、观测语义、定位内核和保护配置需配套使用。

| 路径 | 用途 |
|---|---|
| `models/quality_recovery_20261014/policy_2000.zip` | 原实车部署模型，未替换 |
| `models/stop_feedback_20261015/w11_recovery_bc800_s231015/` | 参考 BC：`policy_bc.zip`、`bc_portable/`、优化器、RNG 和示范 |
| `models/stop_feedback_20261015/w11_control2000_s231015/` | 新对照 2k：最终 `policy.zip`、`portable/`、优化器及全部实际样本 |
| `models/stop_feedback_20261015/w11_retained2000_s231015/` | 新约束 2k，保留系数 0.05；完整训练产物同上 |
| `models/stop_feedback_20261015/early_transfer_bc800_s231015/` | 新诊断 BC，2,355 标签、800 次监督更新、PPO 0 步 |
| `models/stop_feedback_20261015/proactive_augmented_ppo_8000_s231015/` | 旧独立 8k PPO 完整样本、优化器与源码快照 |
| `models/localization_mapping_20261015/ray_consistent_candidate/portable/` | 独立定位算法候选，未合入本轮恢复对照 |
| `models/docking_selected_5000_20261015/` | 历史位置＋朝向停靠模型 |
| `training/neural_navigation/` | 地图、定位、路由、反馈、动作检查和方向目标运行时 |
| `training/experiments/` | 训练、监督初始化、实际样本审计和闭环评估 |
| `training/assets/real_recording_20261014/` | map_01、1,516 条实车决策、1,911 条反馈及轮速/IMU 精简数据 |
| `training/assets/localization_replay_20261015/` | 28 条冻结仿真回放；真值仅供监督和评价 |
| `reports/handoff_save_20261016/` | 本次固定进度、完成轨迹和原机任务状态 |
| `deployment/ros2_recording_adapter/` | ROS 2 记录与模型集成接口 |

原部署 SHA256 为 `7a3dad60f76060daf6b311b4bc6db921d3925ef0cdcf11683fe99e282173de11`。其他模型哈希见 SNAPSHOT。`policy_bc.zip` 是监督初始化；最终 `policy.zip` 才包含完整 PPO 更新。`policy_step_2000.zip` 在最后一次更新前保存，不能代替最终文件。恢复 ZIP 使用 `experiments.stop_recovery_policy.StopRecoveryPPO.load`，运行时按 manifest 的 `required_runtime` 解析。

## 5. 新电脑接着评估

在仓库根目录、已激活环境及第 3 节环境变量下执行。每个输出目录须不存在，各命令串行运行。下面完整重现约束组；不能将重复结果累计为新的独立成功样本。

```bash
model=w11_retained2000_s231015
python tools/prepare_native_bundle.py \
  --source models/stop_feedback_20261015/$model/portable \
  --output training/runs/save_$model

nice -n 19 python -m experiments.evaluate_docking_navigation \
  --candidate training/runs/save_$model --candidate-name $model \
  --cases reports/stop_feedback_20261015/w11_success_augmented_protocol_20261016/gate_cases.json \
  --output training/runs/save_${model}_navigation --cpu-cores 0.35
nice -n 19 python -m experiments.evaluate_feedback_recovery \
  --bundle training/runs/save_$model --label $model \
  --specifications reports/stop_feedback_20261015/frozen_recovery_cases.json \
  --output training/runs/save_${model}_prefixes --cpu-cores 0.35
nice -n 19 python -m experiments.evaluate_feedback_recovery \
  --bundle training/runs/save_$model --label $model \
  --specifications reports/stop_feedback_20261015/proactive_protocol_20261016/w11_prefixes.json \
  --output training/runs/save_${model}_w11 --cpu-cores 0.35
nice -n 19 python -m experiments.evaluate_docking_navigation \
  --candidate training/runs/save_$model --candidate-name $model \
  --cases reports/stop_feedback_20261015/w11_success_augmented_protocol_20261016/fresh_cases.json \
  --output training/runs/save_${model}_fresh --cpu-cores 0.35
```

本次 16 项已完成轨迹在固定 validation 中。评估入口没有自动跳过已完成行；若只运行余下任务，应从原 cases 按同一 id 筛选到新文件，保持参数不变，合并统计时按模型、任务和协议去重。要重现对照组，将 `model` 改为 `w11_control2000_s231015`。参考 BC 使用其 `bc_portable/`，完整配对示例见 [历史详细指南](docs/handoff/WORK_README.md)。

32 秒接管诊断可以脱离原机 systemd 队列直接运行；它是固定历史后的恢复测试，不是全程自主任务：

```bash
python tools/prepare_native_bundle.py \
  --source models/stop_feedback_20261015/w11_recovery_bc800_s231015/bc_portable \
  --output training/runs/save_reference_bc
nice -n 19 python -m experiments.evaluate_feedback_recovery \
  --bundle training/runs/save_reference_bc --label reference_bc_control_wait_t32 \
  --specifications reports/stop_feedback_20261015/reference_wait_t32_protocol_20261016/prefix.json \
  --output training/runs/save_reference_wait_t32 --cpu-cores 0.35
```

## 6. 训练入口与续训边界

入口为 `python -m experiments.train_prefix_recovery`。本轮完整 2k 配置见 [PLAN.json](reports/stop_feedback_20261015/retention_pair_20261016/PLAN.json)，可复制的相对路径训练命令见 [历史指南第 5 节](docs/handoff/WORK_README.md)。训练前先用 `tools/prepare_native_bundle.py` 重建父包，输出到新的 `training/runs/save_*` 目录。示范保留系数 0.05 与关闭组 0 分开对照；长等待探索系数本轮均为 0。

网络、优化器、RNG、课程配置、示范及实际 rollout 均保留，可加载分析和用于新实验设计。但现 CLI **不支持精确续跑同一半回合的物理状态、雷达历史和全部内部记忆**；不能将一次新初始化实验直接写成旧模型无缝追加步数。`--reference-bc` 是核对 BC 初始化，并非任意最终 PPO 检查点的续训开关。原机 `run.py`、`after_*.py` 的绝对路径、PID 与依赖关系只供审计；新机使用上述直接入口。

每组 2k 训练原机约 25.0 / 26.4 分钟，软预算 0.35 核，峰值 RSS 约 2.35 / 2.39 GiB。新机耗时应实测；评估与初始化不计入 PPO 步数。先完成已计划的验证，再根据失败来源决定更新恢复分支、常态 actor、定位器或路由器。

## 7. 定位、停车与实车条件

定位改善来自里程计传播、静态支持筛选、弱方向约束、时序融合和自由射线矛盾检查；本轮恢复 PPO 没有训练定位网络。独立定位对照在 22 条冻结轨迹、2,855 帧中，时序父版 / 射线候选位置 P95 为 18.51 / 18.04 cm，航向 P95 为 1.526 / 1.368°，CPU P95 为 7.88 / 14.59 ms；收益有限、耗时增加，尚未合入恢复对照。源码与验证见 [定位说明](docs/localization/README.md)。

实车录包没有独立绝对位姿真值，当前没有验证超过 AMCL、长期无漂移、任意搬车重定位或 Jetson Orin Nano 时延。已知地图、边导航边建图、开放稀疏场景仍需分别验收。

硬件合同保持：约 39.39×32.95 cm 车体、4 cm 余量；12 个离散动作，当前无横移、无倒车；最高前进 0.20 m/s、转向 0.50 rad/s。220 束原始雷达、约 219.44°，保留后方盲区。恢复训练 436 维、方向运行时 439 维，区别于原部署 300 维。位置＋朝向目标已接入，同位置不同朝向可转向，6 项方向停靠在固定回归集中。

当前闭环主要输入带噪声运动反馈；实车若仍为指令积分里程计，不能认为已经匹配。实车约 6%～12% 无效束、连续缺失、0.1/0.2 秒历史时间、10/12 米量程差异、新障碍未入静态图及低层超时停车还需实测。不能通过缩小车体或取消必要停车来提升表面成功率。

## 8. 迁移范围

当前研究源文件与必要产物已包含；历史工作区差异另有归档，不要求复刻本机的隐藏 worktree 目录。完整 `record.zip`、原始 rosbag/相机图像、虚拟环境、缓存和平台二进制未上传。当前停车分析与回放使用的精简数据已包含；若后续需重新提取原始话题，请另复制原机 `/home/ubuntu/rl_learn/record.zip`，源清单见 [MANIFEST.json](training/assets/real_recording_20261014/MANIFEST.json)。

仓库中的训练示范和本轮“额外条件”主要来自已用地图，不等于未见地图泛化。源记录中的旧绝对路径保留作审计，运行以本 README 的相对路径为准。原机后台评估仍可继续；本分支固定截止，结果更新需再次提交推送。
