# 2026-10-16 受控实车验证包

本轮按用户要求收尾，停止追加训练与扩大回归。交付独立候选、匹配的运行源码、ROS2 适配修改、原 2k 回退包及已完成的仿真记录。原发布模型未覆盖。新候选尚未通过完整验收。

## 已完成与限制

@@RESULT_TABLE@@

12 个闭环仿真条件合计 **11 到达、0 碰撞、1 超时**。最后一个新噪声条件仍有 636 帧在存在运动选项时等待，140 秒超时。因此不能宣称“有可行动作却不尝试”已全部解决。7 个保留条件的时间和路程通过相对直接父版 1.2 倍以内的预设门槛。这些条件仍来自开发地图；4 个新噪声种子没有进入拟合，不能等同于未见地图泛化。

旧失败条件 `early_yield_sensor_holdout_05_0`：直接父版 140 秒超时、620 帧有运动选项时等待；本候选 **36.8 秒到达、40 帧等待**。改善仅针对已比较条件。相同旧 8 条件中，直接父版 7/8、本候选 8/8；不能据此断言全面优于原实车 2k 模型。完整常规/动态、95/90/106 秒恢复和方向停靠回归没有对新候选完成。

本轮在已有 RL 基础 actor 上训练可观测上下文分支：1790 条完整安全来源、**实际 50 次监督更新，新增 PPO 0 步**。目录中的 bc2000 是更新上限，不是实际步数。localizer、router、方向停靠和基础 actor 冻结；本轮不能证明实车定位精度、未知场景定位或停车方向进一步改善。失败或部分轨迹没有作为正示范。

新增上下文包含当前雷达距离、有效命中/无回波比例、目标/子目标方向、测量速度、实际发布动作和停车反馈。12 个原动作、0.2 m/s 线速度上限、0.5 rad/s 角速度上限、矩形车体、4 cm 保护余量、动作掩码与盲区记忆保留，无横移、无倒车动作、无强制运动或部署教师。

主机限资源验证平均约 0.35 个 CPU 核、峰值约 2.13 GiB；不是 Jetson Orin Nano 实测。部分单步推理超过现有 ROS 250 ms 截止时间，可能触发保护停车；请一并记录端到端延迟。极窄位置脱困、全新无地图边导航边建图和搬车后自动全局重定位不在本包已验证范围。

## 文件与版本

- `models/stop_feedback_20261015/release_context_bc2000_s231016/portable/`：本轮完整 625 维推理包。检查点和可续训状态位于它的上一级。
- `models/quality_recovery_20261014/`：原实车 300 维质量恢复 2k 模型与 portable 回退包。
- `training/`：本轮配套 Python 源码与 C++ 定位核源码。
- `ros2_patch/rl_navigation/`：需要合并到当前实车工程的四个适配文件。
- `reference_ros/rl_navigation/`：录制快照的 Python 依赖加上述修改，仅供导入检查和对照；不是完整 ROS 安装包，不覆盖当前传感器/底盘程序。
- `maps/recorded_map_01/`：录制时地图的拷贝，作为历史对照；实车启动优先使用现在的地图和匹配位姿。
- `reports/`：训练、12 项闭环结果、完整成功与失败轨迹、适配检查及取消任务记录。
- `SOURCE.json` 与 `SHA256SUMS.json`：归档身份及逐文件散列；`verify.py` 检查归档完整性。

新 checkpoint SHA256：`a77684103768dbdaa5213246118e05b74fc108cc93f9ed25978dd48da2b0cfe3`。

原实车 policy SHA256：`7a3dad60f76060daf6b311b4bc6db921d3925ef0cdcf11683fe99e282173de11`。

本包是本地收尾快照，包含此前 GitHub `handoff3` 截止时间之后的修改；本轮没有自动推送远程。GitHub handoff3 仍固定在 `6d443f31bb3b20d2a5d31539a6490be3c2bddf05`，14:06:06 北京时间。生成配置中的 `branch_commit` 使用源码散列身份，不能当作已提交的 Git commit。

## 在另一台电脑准备

解压后进入本目录，使用目标 ROS 环境可访问的 Python。仅推理依赖 numpy、torch、scipy、PyYAML、Pillow；沿用实车已经工作的环境。主机版本见 `ENVIRONMENT.json`，Jetson 使用匹配 JetPack 的 PyTorch，不能直接套用主机 CUDA 轮子。本包没有 venv、原始 bag、大量历史运行目录和 MuJoCo 安装。

```bash
python3 verify.py
python3 configure.py --kind candidate
python3 configure.py --kind baseline
```

分别生成 `generated/model_candidate.json` 和 `generated/model_baseline.json`，含本机绝对路径、对应 manifest 散列和运行时合同。移动目录后重新生成。配置生成不会覆盖归档权重。

625 维候选必须使用 `neural_navigation.release_runtime.ReleaseRuntime`；300 维回退包使用 `experiments.quality_sensor_runtime.QualitySensorNavigationRuntime`。不能只替换 actor，也不能混用旧合同和新特征生成器。

### 在 Jetson/不同 CPU 架构上

归档 native 定位核为 x86_64。在 Orin 的 ARM64 上先用匹配源码生成独立目录，需要 g++ 和 Eigen3 头文件。以下命令只编译定位核，不训练 actor：

```bash
python3 tools/prepare_native_bundle.py \
  --source models/stop_feedback_20261015/release_context_bc2000_s231016/portable \
  --output generated/candidate_native
python3 configure.py --kind candidate --bundle generated/candidate_native
```

工具校验源码及组件散列，更新新 native 文件的 manifest；归档包与 actor 权重保留。原 300 维回退包没有这个 native 内核。没有在本机验证 ARM 编译或目标板实时性。

## 合并与启动

先备份实车当前 `rl_navigation` 包、配置及模型。把 `ros2_patch/rl_navigation/` 中四个文件合并到当前 Python 包；沿用现有 package.xml、setup.py 和控制接口。在 ROS 工作空间按原流程重新安装这个包，例如：

```bash
colcon build --packages-select rl_navigation --symlink-install
source install/setup.bash
```

其他工作目录中的控制、定位节点按原工程启动。停止旧 model_node，避免两个模型竞争命令。新节点明确指定本包目录与生成配置；`REALTEST_ROOT` 是解压目录，`ROBOT_MAP` 是当前地图 YAML 的绝对路径：

```bash
REALTEST_ROOT=/absolute/path/to/realtest_20261016
ROBOT_MAP=/absolute/path/to/map_01.yaml
ros2 run rl_navigation model_node --ros-args \
  -p model_source:="$REALTEST_ROOT" \
  -p model_config:="$REALTEST_ROOT/generated/model_candidate.json" \
  -p map:="$ROBOT_MAP" -p mode:=shadow -p auto_initial_pose:=false
```

上述先检查加载、合同、TF、里程计和候选动作，不驱动车辆。使用 RViz 的 **2D Pose Estimate** 给出实际位置与当前朝向，再用带箭头的导航目标指定位置和期望朝向。当前实现仍需要可信地图初始位姿，不能用目标方向代替机器人姿态。

沿用现有底盘/模型参数；录制快照的轮径 0.10 m、轴距 0.20 m、轮距 0.22 m 仅作对照，若现在不同需要传入当前值。雷达外参、反馈速度来源、控制仲裁和原超时停车不由本包改写。launch 文件可能覆盖节点参数，应检查启动后的实际参数。

开始受控实车路线时，沿用原工程的执行与急停机制，将上述 `mode:=shadow` 改为 `mode:=execute`。manifest 仍是 `deployment_approved=false`，表示完整发布验收未完成。原工程底盘的 enable_control 仍按用户已有流程处理；本包不修改固件和控制使能。

雷达无效束门槛默认 8%，而这次录制运行时曾设为 30%。这会影响是否停车和对比公平性；本包不自动取其中更宽松的值。先检查实际参数与记录，区分扫描质量保护、推理超时、只有停止可选和策略主动等待，避免把停车全部归咎于模型。

## 建议这一轮实车做什么

在同一地图、起终点、初始朝向和同一保护设置下，先旧版、再候选各重复 3 次，记录人工重定位及接管次数。先做熟悉空旷路线，再做前方新增静态障碍、慢速横穿后离开、短时遮挡后恢复；窄门放在最后。不要直接以最窄通道或故意雷达断流作为第一条路线。

重点观察：新增障碍未入图时地图定位是否被拉偏；障碍离开后多久恢复；有可行动作时是否持续等待；是否发生非必要停车、碰撞、人工接管；到达位置和朝向是否满足需求。原地只改目标方向也单独试一次，此轮没有新增其闭环验收证据。

录制原有 record_node，保留 metadata、decision 和实际命令；或在完整轨迹开始前订阅以下主题。topic 名称以实车实际发布为准：

```bash
ros2 bag record -o realtest_candidate \
  /scan /odom /rl_navigation/odom_raw /imu /tf /tf_static \
  /initialpose /goal_pose /rl_navigation/model_status \
  /rl_navigation/decision /rl_navigation/session_metadata \
  /rl_navigation/model_pose /rl_navigation/cmd_vel_candidate \
  /rl_navigation/cmd_vel
```

同时记录底盘最终接收的速度与实际轮速/IMU（已有则录，无须为本轮增加硬件）、地图 YAML/图像、实际参数、视频和人为干预时刻。录制程序需要在发目标前订阅 decision，才能保留完整 625 维观测。本包对应元数据主题是 `/rl_navigation/session_metadata`；传感器和底盘最终命令主题以实车为准。

位置误差要用独立参考点、测量或可信定位对照，不能以模型自身估计证明自身准确。静止、直行、转弯、闭环回到标记点分别检查；尚无独立真值时，只能报告现象和一致性。

## 回退

取消当前任务并确认零速：

```bash
ros2 service call /rl_navigation/cancel std_srvs/srv/Trigger '{}'
```

停止新 model_node，用同样地图、参数和 model_source，改加载 `generated/model_baseline.json` 后启动原 300 维 2k 包；需要完全还原旧系统时，恢复备份的 ROS 包和原配置。候选的方向停靠/新定位核不能与旧 actor 混搭。旧版目标只处理位置，不应当成同等朝向验收。

## 本地完成的检查

新观测/冻结与掩码检查 3/3；ROS 适配无 ROS 检查 4/4；保存及导出 1790 条来源动作一致；12 项闭环结果完整保留。完整 ROS 节点、真实控制链、Orin 速度与新地图仍需实车验证。

下一轮优化以实车录制证据为依据：先判断未入图障碍的定位偏移、发布延迟和无效等待分别来自哪一层，再选择定位或策略实验。本轮不继续堆训练步数。
