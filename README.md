# AGV 定位优化方法与验证方案（2026-10-15）

定位专项说明见 **[定位优化 README：方法、实测证据与验证步骤](docs/localization/README.md)**。

本分支 `docs/localization-validation-20261015` 在已发布的 2k 模型归档上增加方法文档、配置快照和精简评估证据。最新定位改善来自里程计预测、静态地图几何匹配和短时扫描融合，不是定位网络重新训练。本分支不是最新实验代码或候选模型的发布包；实车绝对精度、任意位置重定位和在线 SLAM 尚未通过验收。

下文保留 2026-10-14 模型的历史归档说明，历史成绩与最新定位回放分开阅读。

## AGV 学习导航：历史较优模型归档（2026-10-14）

本分支 `models/best-2026-10-14` 保存本地已查到的历史完整开发回归中到达表现最好的候选：`quality_recovery_4000` 试验的 **2,000 步中间检查点**。它继承已有预训练，不是从零只训练 2,000 步。本次是归档交付，没有新增训练。

**历史固定开发集：46/48 安全到达（95.8%）、1 次碰撞、1 次超时。没有通过部署验收，也没有证明最短路径、窄道或实车能力。** 这 48 项不是封存的 200 项最终验收集。

## 模型与配套文件

| 内容 | 路径 |
|---|---|
| 原始 PPO 检查点，按文件保留权重和优化器 | [policy_2000.zip](models/quality_recovery_20261014/policy_2000.zip) |
| 新导出的推理包、学习定位和路由组件 | [portable/](models/quality_recovery_20261014/portable) |
| 原 4,000 步试验配置；本分支选择中间 2k | [training_run_config.json](models/quality_recovery_20261014/training_run_config.json) |
| 发布信息及 SHA256 | [RELEASE_MANIFEST.json](models/quality_recovery_20261014/RELEASE_MANIFEST.json) |
| 历史报告、固定任务与逐任务结果 | [reports/](models/quality_recovery_20261014/reports) |
| 9,924 帧实际历史策略输入和动作，压缩保存 | [parity_inputs.npz](models/quality_recovery_20261014/reports/parity_inputs.npz) |
| 训练失败模板和唯一回放记录 | [training_inputs/](models/quality_recovery_20261014/training_inputs) |

检查点 SHA256：`7a3dad60f76060daf6b311b4bc6db921d3925ef0cdcf11683fe99e282173de11`。

本分支恢复了 229 个记录过哈希的历史源码文件，匹配该模型的评估版本，见 [source_restore.json](models/quality_recovery_20261014/reports/source_restore.json)。原工作区的窄道改动保留在原分支，本分支使用归档运行时。源码相符不等于随机训练能逐位复现。

## 成绩与比较范围

| 同一历史 48 场景开发集 | 安全到达 | 碰撞 | 超时 |
|---|---:|---:|---:|
| 原 masked4k 基线 | 44/48 | 3 | 1 |
| 所选质量恢复 2k 检查点 | **46/48** | **1** | **1** |

3 个旧失败转为安全，但 `independent_dynamic_013` 从旧成功退化为超时。43 个共同安全任务的用时差中位数为 0 秒，路程差中位数约 −0.00446 米，不能据此声称明显更快、更短。[配对证据](models/quality_recovery_20261014/reports/MATCHED_BASELINE_COMPARISON.json)

已推送的追加 10k 恢复模型使用另一套协议，普通回放 9/9、专项 2/5，不能与 46/48 混算。今天新增 20k 的窄道试验在另一套 25 场景中训练前后都是 7/25，完整窄道均 0/13，未作为本分支模型。原始 100k 在单个静态绕障演示中曾更快、更短，同样不能当作综合排名。

本分支的“较优”仅指已有完整开发回归记录，不表示任何环境下最优。

## 方法与硬件约束

已知二维地图、初始粗位姿、激光、里程计/速度输入；学习定位网络估计位置、神经路由网络输出路线、PPO 选择局部速度。运行时没有 Nav2、行为树或 A* 接管，但地图/激光融合、动作掩码和紧急停车仍由程序处理。

- **300 维观测、12 个离散速度动作**，需要 `QualitySensorNavigationRuntime`；不能接普通 282 维或新窄道 318 维入口。
- 前进上限 0.20 m/s、角速度上限 0.50 rad/s，无横移；这 12 个策略动作不含倒车。
- 车体约 39.39 × 32.95 cm；220 束雷达，视野约 219.44°，保留后方盲区。详见 [hardware.json](models/quality_recovery_20261014/hardware.json)。
- 此恢复阶段冻结定位/路由权重，使用 1/60 奖励缩放、质量状态观测、恢复片段回放和参考分布约束。原试验配置里 `completed_steps=4000` 指整个试验，归档模型实际取自 2,000 步。
- 动力学、噪声、延迟未实车标定，Jetson Orin Nano 耗时未测；仍存在动态碰撞和无可用动作后的停车超时。

## 安装与校验

推荐 Python 3.10，在仓库根目录执行 CPU 示例：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r training/requirements.txt

PYTHONPATH=training:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python training/verify_best_model_release.py
```

校验包内文件、历史源码和地图/传感器配置哈希，在 9,924 帧实际历史观测上比对 ZIP、TorchScript、动作掩码和记录中的执行命令。**该校验没有重新运行 48 个闭环场景。** 本次交付检查见 [release_checks.json](models/quality_recovery_20261014/reports/release_checks.json)。

## MuJoCo 查看

需要图形桌面和 OpenGL，纯模型不需要 ROS 2：

```bash
bash scripts/preview_best_model.bash
bash scripts/preview_best_model.bash --case independent_dynamic_006
```

空格暂停，R 重置。B 两边指向同一归档模型，不能用来作版本对比。本入口按固定任务目标导航，使用历史查看器，不包含后来版本的手动目标点和柱子搬运功能。仿真真值只用于显示/评分。

## 重跑固定开发测试

单个案例或全部 48 案例，单线程顺序运行，输出目录必须是新的：

```bash
PYTHONPATH=training:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python training/evaluate_best_model_release.py \
  --case independent_dynamic_006 --output training/runs/best_20261014_one

# 不传 --case 就运行全部 48 项，耗时较长
PYTHONPATH=training:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python training/evaluate_best_model_release.py \
  --output training/runs/best_20261014_full48
```

这些入口限制计算线程，不自动替其他机器设置系统 CPU/内存配额。大型日志、视频和虚拟环境不上传；本包保留压缩的演员输入、任务和结果。

## 从此模型续训

示例追加 2,000 步并输出为新试验，不自动替换归档模型：

```bash
PYTHONPATH=training:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m experiments.train_quality_navigation \
  --checkpoint models/quality_recovery_20261014/policy_2000.zip \
  --failure-pool models/quality_recovery_20261014/training_inputs/failure_pool.json \
  --training-log models/quality_recovery_20261014/training_inputs/replay_episode.jsonl \
  --replay-step 10458 --resume-quality --scheduled-recovery \
  --steps 2000 --seed 92521 --output training/runs/best_20261014_continuation
```

这不是重现原检查点。后续先处理预测失真、安全死锁和窄道可执行性，再增加训练规模。
