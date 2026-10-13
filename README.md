
# AGV 学习导航与 MuJoCo 仿真

> **模型交接：** [核心思想、训练过程、能力边界与后续优化建议](docs/MODEL_HANDOFF/README.md)。包含三个学习模块的分工、地图/激光实时融合、现有评估结果，以及独立学习导航的优化方向、传统方法对比和实车迁移建议。

已知地图、部分视野激光与动态实体障碍下的四轮差速机器人导航。当前系统由学习定位、神经路线规划和 PPO 速度决策三个模型配合；地图/激光融合、路线切换确认和运动约束仍由程序处理。机器人导航运行时不使用 AMCL/A*；传统对照另使用实际 Nav2 AMCL、NavFn 与 TEB。

## 2026-10-13：恢复专项模型与交互测试

本分支从首次克隆版本 **`c63c1b7ddf1045c2a3b0c381eb4171bab0b8f57a`** 整理后续训练程序、独立候选模型和评估证据。原始模型保持保留，没有替换默认启动模型。**最新模型未通过专项验收，不能据此部署到实车。**

### 本次训练方案

- 从保守恢复对照模型继续训练 **10,000 步 Maskable PPO**；父模型此前专项追加过 8,000 步，双方还继承更早的预训练基础，因此 10,000 是本轮新增步数，不是从零训练总量。
- 学习定位器与神经路由器固定，普通导航 actor 基底固定；训练恢复分支和价值网络，保留父检查点权重及 Adam 状态。
- 观测为 **300 维**：三帧激光历史、子目标/终点、速度、上一动作、定位质量、停车原因、实际执行命令、合法动作与动态风险。雷达 **220 束、219.44° 视野**；策略不接收障碍物或机器人真实位姿。
- 策略在 **12 个速度组合**中选择动作；动作屏蔽来自传感器安全检查，采样、PPO 概率和导出策略使用同一套规则。当前动作集合不包含倒车。
- 本轮扩展恢复分支的学习条件：此前接受了原地转向、路径及定位可信且尚未充分对齐时，恢复分支继续参与决策。没有固定转向方向、时长或接管速度的替代控制器。
- 训练交替使用历史训练失败与一般课程，评估不作为训练标签；维持原奖励结构，训练奖励缩放 **1/60**，附加 alignment cost 为 **0**。
- 参数：单环境，rollout/batch 250，4 epochs，γ=0.995，GAE λ=0.95，entropy coefficient=0.003，恢复分支 LR=3e-4，critic LR=3e-5。
- 本机 CPU 实验采样及学习 **50 分 44 秒**，约 **3.29 步/秒**；后台组限速约半个 CPU 核心。此限速不是下述脚本在其他机器上的默认保证。

发布目录：[models/recovery_20261012](models/recovery_20261012)。包括 final/initial 检查点、可续训父模型、推理包、训练输入、原始训练源码快照和评估摘要，总大小约 6 MB；不上传虚拟环境、重复中间检查点或全部回放轨迹。文件 SHA256 见 [MODEL_MANIFEST.json](models/recovery_20261012/MODEL_MANIFEST.json)。

### 与首次克隆结果比较

首次克隆保留的统一限速四目标评估：原 `policy_100000.zip` **4/4 到达、0 碰撞、208.6 秒、26.66 米**；同报告 TEB 为 **4/4、0 碰撞、122.8 秒、24.18 米**。这是历史记录，本分支没有重新运行该旧协议。另一个视频调参报告的 PPO 100k 为 150.4 秒，控制参数不同，不能混用。原始报告已从克隆提交提取到 [docs/results/clone_baseline_c63c1b7](docs/results/clone_baseline_c63c1b7)。

| 对比项 | 首次克隆版本 | 本轮候选模型 |
|---|---|---|
| PPO 控制 | 235 维连续动作策略 | 300 维、12 动作的 Maskable PPO 恢复分支 |
| 停车信息 | 没有本轮完整反馈观测 | 停车原因、实际执行动作、合法动作、等待及风险信息 |
| 历史四目标协议 | 4/4 到达、0 碰撞 | **未跑同协议，不能宣称优于首次克隆模型** |
| 本轮 9 组配对 | 不属于首次克隆的评估 | 初始与最终均 9/9 到达、0 碰撞、0 超时 |
| 本轮 5 组专项 | 不属于首次克隆的评估 | **2/5 到达、1 碰撞、2 超时** |
| 实车验证 | 无 | 无 |

以下是相同 5 个专项场景下的后续模型对比，**“旧质量模型”不是首次克隆的 100k 模型**：

| 版本 | 安全到达 | 碰撞 | 超时 |
|---|---:|---:|---:|
| 旧质量模型＋时间一致传感器配置 | 3/5 | 1 | 1 |
| 8k 保守对照父模型 | 2/5 | 1 | 2 |
| 8k 探索增强模型 | 3/5 | 1 | 1 |
| 本轮新增 10k 持续恢复模型 | **2/5** | **1** | **2** |

本轮 023 场景仍有 **646 帧存在合法非零动作却等待**；021 仍超时；006 到达时间从父模型的 75.4 秒增至 105 秒。说明反馈接口和普通安全回归已工作，但**自主脱困能力没有得到验证，不能把更多训练步数当成必然改善**。详见 [专项结果](models/recovery_20261012/reports/specialist_five.json) 与 [9 组配对结果](models/recovery_20261012/reports/paired_nine.json)。这些是重复使用的开发场景，不是独立部署验收集。

### 训练后的路线效率修改（与权重训练分开）

可选 `--efficient-routes`：从神经路由分数有界解码最多四条不同起始方向的候选，对完整路线做地图和激光占据检查，优先剩余总长度；长度接近时小幅偏好朝目标、少掉头。有效旧路线切换仍需稳定确认。未增加 A*/Nav2，速度命令仍由 actor 给出；这属于程序中的候选路线比较，**不能说路线选择本身全部由新训练学得**。

同一最新 actor、同一随机种子与前方静止柱子场景：

| 指标 | 原路线逻辑 | 开启路线比较 |
|---|---:|---:|
| 到达时间 | 43.0 秒 | **30.6 秒** |
| 实际行驶距离 | 4.38 米 | **3.54 米** |
| 碰撞／物理保护触发 | 0／0 | 0／0 |

仅此案例耗时下降约 28.8%、路程下降约 19.2%。**不是模型训练收益，也不是跨场景最短路径或动态最优证明**；完整回归尚未完成。本机控制延迟 P95 约 109 ms，未测 Jetson Orin Nano。[对照数据](models/recovery_20261012/reports/route_efficiency_closed_loop.json)

### 使用发布模型

完成下文安装后，从仓库根目录执行：

```bash
# 验证发布文件及原生检查点/导出 actor 的输出一致性
.venv/bin/python scripts/verify_recovery_release.py --inference

# 最新模型：目标点输入窗口＋静止柱子搬运；初始暂停
bash scripts/preview_recovery_model.bash

# 单独启用实验性路线效率比较
bash scripts/preview_recovery_model.bash --efficient-routes

# 从打包的父模型重新进行本轮专项训练，不自动替换默认模型
bash scripts/train_recovery_model.bash training/runs/recovery_reproduction_10000 10000
```

图形测试需 Ubuntu 桌面及 `python3-tk`。输入 X/Y（米），点击发送目标；双击柱子取走并隐藏，右键地面放置；默认柱子不自动移动，放置时拒绝与墙、小车或其他柱子重叠。R 重置，B 切换打包的父模型并重置。手动测试不是固定协议评估，移走障碍物后仍可能等待传感器记忆被真实自由空间扫描清除。

续训入口保留初始/最终和每 1,000 步的中间检查点，训练后自动跑 9 组配对；本次发布只收录初始/最终及父模型。至少 250 MiB 空闲磁盘才允许启动，小于 100 MiB 时保存低磁盘检查点并停止。打包日志仅包含 replay step 10458 所需的一条历史记录；原始训练源码和来源哈希保留在 `training_source_snapshot/` 与 `TRAINING_SOURCE.json`。当前源码另含训练后的交互/路线修改，**不承诺逐位复现原训练随机序列**。

后续优先诊断 021/023 的动作概率、实际探索与奖励结果，验证自主恢复和碰撞回归，再增加训练量。本候选及原模型均没有通过真机部署验收。


## 录制演示

https://github.com/user-attachments/assets/07f6c48b-e696-44cd-a4b2-b898b000c18c

**[观看 / 下载完整视频](docs/media/navigation-four-window.mp4?raw=true)** · 约 8.7 MB · 82 秒 · 1600 × 1254 · 25 fps · H.264 MP4

左上为 Navigation2 + TEB，右上为追加 PPO 1 万步，左下为 5 万步，右下为 10 万步。画面包含规划路线、实际轨迹、部分视野激光扫描和实体柱子干预；统一 5 倍速，片尾汇总用时与路程，并介绍两种导航方式的优劣势。

本视频展示不同限速与控制参数下的表现；10 万步组完成更快，TEB 路程更短。模型权重未因录制而改变，实时测试命令仍运行原始模型配置。

## 安装（Ubuntu 22.04 / Python 3.10）

```bash
sudo apt-get install python3-venv python3-tk build-essential libglfw3-dev libgl1 libosmesa6
bash scripts/setup_cpu.bash
source training/activate.bash
```

纯模型 MuJoCo 测试不要求 ROS。GPU 训练请按服务器 CUDA 环境安装相容的 PyTorch 构建，再安装 training/requirements.txt；CPU 安装脚本不会配置 CUDA。

## 原始 10 万步模型实时测试

所有命令从本仓库根目录执行。

```bash
source training/activate.bash
MUJOCO_GL=glfw python training/preview_live_navigation.py \
  --policy training/runs/remote_room_100k/policy_100000.zip \
  --components training/runs/remote_room_100k --max-leg-steps 1800
```

双击柱子后按住右键拖动，松开取消选中；左键旋转、滚轮缩放；空格暂停，R 重新开始。使用原始策略，无视频演示额外调参。激光范围 −108.38° 至 +111.06°（219.44°），220 束；两根实体柱子、四个目标点、机械臂离线归位姿态与原案例一致。

## 模型和训练

- training/runs/remote_room_100k/localizer.pt：学习定位网络。
- training/runs/remote_room_100k/router.pt：神经路线网络。
- policy_10000.zip / policy_50000.zip / policy_100000.zip：有预训练基础的追加 PPO 检查点，权重按原文件保留。
- training/runs/full_model_navigation_v3/：原续训父模型、定位/规划网络和离线动作样本，供复现原并行续训。

定位与路线网络的训练入口：training/train_neural_components.py；PPO 并行续训入口：training/remote/train_parallel.py。原 10 万步实验固定定位与规划权重，只续训 PPO，随机起点、朝向和目标顺序，主要使用既有房间布局。

```bash
# 生成学习定位/规划的数据地图，再独立训练网络
python training/generate_navigation_dataset.py
python training/train_neural_components.py --output training/runs/new_components

# 从保留的 v3 父模型开始新的 PPO 续训，CPU 示例
python training/remote/train_parallel.py --device cpu --envs 4 \
  --steps 100000 --output training/runs/new_continued_ppo
```

历史脚本默认引用其他旧检查点时，需要自行训练或从原目录另取，整理版没有打包所有历史结果。

## 自动干预评估与参数演示

```bash
# 原统一限速的模型评估
python training/navigation_comparison.py \
  --policy training/runs/remote_room_100k/policy_100000.zip \
  --output training/runs/new_fair_100k --max-leg-steps 1800

# 视频调参配置：与原始模型测试分开
python training/navigation_parameter_demo.py --preset ppo_100k \
  --output training/runs/new_parameter_100k
```

实体柱子按靠近目标触发、停留约 8 秒后离开的规则干预。录制脚本及压缩后的四窗口演示视频已保留；原录制轨迹未打包，重新录制需先生成其所需的四组轨迹。不同参数演示不能用来证明训练步数或算法的公平优劣。保留的实测数据见 docs/results/。

## 可选传统 Nav2 + TEB 对照

需要 ROS 2 Humble、Nav2、colcon、g2o 与 costmap_converter/TEB 构建依赖；第三方源码保留在 training/teb_ws/src/，构建产物需在目标机器重新生成。原桥接方式与配置见 docs/NATIVE_NAVIGATION_COMPARISON.md。这部分不属于纯模型启动依赖，未在没有 ROS 的机器上验证。

## 适用范围

目前是仿真导航原型，底盘采用平面力驱动跟踪速度，不是已标定的完整轮胎/电机模型。没有实车验证，也未证明跨任意地图的成功率。旧版历史说明见 docs/LEGACY_TRAINING_README.md，以本 README 的当前入口为准。

整理清单见 docs/EXPORT_MANIFEST.json；验证结果见 docs/VALIDATION.json。上传前检查第三方许可说明 THIRD_PARTY_NOTICES.md；本副本没有替作者新增开源授权。
