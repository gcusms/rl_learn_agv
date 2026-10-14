# 本轮优化与复现

工作分支 `fix/localization-mapping-20261015`，基线代码 `277c4bf`。所有新产物单独保存，实车原模型未覆盖。实验需要现有 Python 虚拟环境；边建图实验还需要本机已安装的 ROS 2 Humble / SLAM Toolbox，不需要 Nav2 控制器或行为树。

## 运行环境与限制

```bash
cd /home/ubuntu/rl_learn/.real-adaptation-20261014
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=training
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

Python：`/home/ubuntu/rl_learn/rl_learn_agv/.venv/bin/python`。每个主要 Python 实验 `nice -n 19`、单线程，软平均 CPU 配额 0.25～0.35 核，RSS 上限 4 GiB；RSS 上限不是所有并发子进程之和。SLAM Toolbox 的 Ceres 子进程固定在一个逻辑 CPU，避免已安装 Humble 插件开启最多 50 个工作线程。没有修改系统 ROS 安装。私有 ROS domain 182～199，实验桥仅发布模拟扫描、时钟与 TF，不发布实车电机控制指令。

## 停车训练

本轮做了两个实验：原 5k 方向停靠模型追加 6k PPO；原 5k 模型增加临界角度示范预热，再追加 3k PPO。仅终端 actor 与 critic 更新，正常导航 actor、localizer/router 网络权重冻结。完整命令选项见 `python -m experiments.train_docking_navigation --help`。

```bash
nice -n 19 /home/ubuntu/rl_learn/rl_learn_agv/.venv/bin/python \
  -m experiments.train_docking_navigation \
  --checkpoint models/docking_selected_5000_20261015/policy.zip \
  --components models/localization_mapping_20261015/known_map_v5/portable \
  --demonstration-data models/localization_mapping_20261015/boundary_assisted_terminal/demonstrations.npz \
  --output /tmp/rl_boundary_training_repeat_20261015 \
  --steps 8000 --seed 116120 --cpu-cores .3 --bc-updates 600 \
  --resume-terminal --precision-curriculum --terminal-warmup
```

训练配置、种子、父模型 SHA、示范来源、完整优化器 checkpoint 与阶段记录保存在 `models/localization_mapping_20261015/{precision_terminal,boundary_assisted_terminal}`。这不是从头重训全部导航网络，也不是新增一轮百万步训练。示范角度增广只能用于监督预热，不能算作真实闭环成功；后续 PPO 和评估采用实际运动闭环。预热后重置终端 actor 的 Adam 动量，环境 RNG 未做逐位连续恢复，配置中明确记录。

## 已有地图与原模型回归

```bash
nice -n 19 /home/ubuntu/rl_learn/rl_learn_agv/.venv/bin/python \
  -m experiments.evaluate_docking_navigation \
  --candidate models/localization_mapping_20261015/boundary_assisted_terminal/portable_8000 \
  --candidate-name boundary_ppo_8000 \
  --cases reports/localization_mapping_20261015/navigation_validation_cases.json \
  --output /tmp/rl_known_map_repeat_20261015 --cpu-cores .25
```

48 个新增任务属于验证集，原阶段验证地图重复使用；地图上的路线、种子和部分初始朝向为新任务。旧 28 项属于开发回归集，不能与历史 46/48 或 9/9 直接合并为成功率。真实部署 2k 模型是 300 维位置模型，公平对比只用位置目标任务，不拿方向停靠任务扣它的分。

## 边导航边建图

```bash
source /opt/ros/humble/setup.bash
export PYTHONPATH=training:$PYTHONPATH
nice -n 19 /home/ubuntu/rl_learn/rl_learn_agv/.venv/bin/python \
  -m experiments.evaluate_online_mapping \
  --bundle models/localization_mapping_20261015/boundary_assisted_terminal/portable_8000 \
  --cases 12 --seed-base 116710 --exploration --dynamic --domain 187 \
  --output /tmp/rl_online_map_repeat_20261015 --cpu-cores .25
```

模拟静态地图只供物理场景与离线任务抽样/可达性检查，实时 SLAM、路由器、actor、保护层均看不到它。实时输入为同一份退化扫描、带误差的里程计、实际 SLAM 地图和估计位姿。车体与动作合同保留；雷达为实车配置的 220 束、约 219.44° 视野，扫描有效量程上限 10 m；旧模型特征归一化上限仍为 12 m。

初始化是明确记录的监督扫描旋转，不是已经训练出来的自主冷启动。全局未知区域采用已观测连通区域中的目标导向子目标选择器，它是工程逻辑；速度与转向由 RL actor 决定。没有宣称全局探索完全由强化学习接管。未知格保持阻塞，图像补齐不制造雷达可见墙体。保存图像使用 127 灰度保持未知语义，避免 205 灰度在原 free_thresh 下变成空闲。

地图 `.data` 与 `.posegraph` 支持持久化。恢复测试只覆盖保存图的原起点，不包含任意位置重定位。SLAM Toolbox 已开启回环配置，但本轮没有独立量化长距离回环纠偏效果。部分旧实验大文件进行了无损 gzip；还原 `gzip -dk persistent_map.data.gz persistent_map.posegraph.gz`，原始 SHA 和压缩路径见 `compressed_maps.json`。大图文件保存在本地，不加入 Git。

## 可选 RF2O 研究试验

RF2O 没有安装到系统。官方 ROS 2 分支源码通过 `https://codeload.github.com/MAPIRlab/rf2o_laser_odometry/tar.gz/refs/heads/ros2` 下载到 `.research`，单线程编译。研究版修正了零初始四元数和 `laser_oldpose_` 自赋值；扫描虚拟坐标系只对齐 RF2O 的对称视野假设。没有向 RF2O 发布仿真真值初始化。实际 6 项对照为 4/6，并出现两次假到达，因此不是本轮默认方案。仅作为被拒绝的研究对照保留。

## 功能检查与产物

```bash
/home/ubuntu/rl_learn/rl_learn_agv/.venv/bin/python -m unittest \
  experiments.test_mapping_runtime experiments.test_docking_navigation experiments.test_laser_increment_odometry -q
/home/ubuntu/rl_learn/rl_learn_agv/.venv/bin/python -m experiments.build_localization_review \
  --root reports/localization_mapping_20261015
PYTHONPATH=training python3 -m experiments.verify_localization_replay \
  --root reports/localization_mapping_20261015
```

回放交互校验用本机系统 Python 中已有的 `websocket-client` 与临时 Chrome 配置，不依赖训练虚拟环境中的该包，也不操作已有浏览器会话。ROS 回调与驻留状态通过函数级检查；归档适配器不是完整的实车 ROS 运行包，本轮未执行完整 ROS 适配器启动或真实电机控制。

真实录制回放缺少独立参考位姿，只能分析不合理的跳变、漂移与恢复，不是厘米级实车定位精度验证。录制实际配置为指令来源的平移反馈加 EKF 融合 IMU 转向，不能称为“轮子实测走过的距离”。录制中已有轮反馈和轮里程计，但本轮未擅自更换真实底盘的里程计来源，没有要求增加编码器或修改固件。上线前必须保持扫描时间、雷达安装变换、地图坐标和方向目标一致；便携运行时、ROS 目标回调和 worker 需要一起更新，不能只换 actor 文件。


## 指令积分输入挑战与新增地图

`command_route_challenge_cases.json`、`command_position_pair_cases.json`、`online_command_challenge_cases.json` 与 `unseen_command_map_cases.json` 显式设置 `input_odometry: command_integrated`。输入只来自该时刻实际发布的速度及控制时间积分；没有调用实际运动反馈或仿真真值。正常 48 项集合使用带噪声运动反馈，不能把其成绩当成指令积分输入成绩。48 项中仅部分任务额外加入 8% 丢点，并非全部任务都有该比例。

```bash
nice -n 19 /home/ubuntu/rl_learn/rl_learn_agv/.venv/bin/python \
  -m experiments.evaluate_docking_navigation \
  --candidate models/localization_mapping_20261015/selected_known_map/portable \
  --candidate-name selected_8k \
  --baseline models/quality_recovery_20261014/portable \
  --baseline-name real_deployed_2k \
  --cases reports/localization_mapping_20261015/command_position_pair_cases.json \
  --output /tmp/rl_command_position_repeat_20261015 --cpu-cores .25
```

新增几何地图未参加本轮 RL 训练，但曾用于本轮定位诊断，属于验证集；不是未查看的最终密封测试集。物理射线场景没有自动增加规划边界墙体，命令输入没有实际车轮反馈，参考真值仅用于奖励／评分。

## 被保留的可选失败修复试验

默认均不启用。`--laser-increments` 启用本项目相邻扫描点到线增量匹配；`--ray-free-layer` 仅清除实际有限射线经过的未知格；`--coarse-subgoals` 让子目标连通性与神经路由器实际 `[2::4,2::4]` 采样一致。原始集合和每任务结果分别保存在 `laser_increment_repair_3`、`online_laser_fresh_validation_8`、`ray_free_repair_3`、`coarse_subgoal_repair_3`。这些实验仍有失败，不能直接作为新的部署默认配置。

`--known-free-dynamics` 是另一项默认关闭的在线建图对照：未知格已由静态图保守阻塞，仅把已观测空闲区域中的新回波写入动态路由层。actor 与停车保护仍读取完整原始扫描，原已有地图运行方式不启用该过滤。配对集合是 `imu_online_cases.json`，结果为 `imu_online_4` 与 `imu_known_free_dynamics_4`，每次只改此开关。

本项目匹配器只借鉴 [Censi 2008 点到线 ICP](https://censi.science/pub/research/2008-icra-plicp.pdf) 的度量思路，使用数值迭代、残差截尾、弱轴检测和里程计回退；没有声称完整复现闭式 CSM 或 RF2O 论文。实测结果用于决定是否保留，论文存在不等于本项目已经解决长距离漂移。


## 原导航保留与转向精修对照

`build_preserved_navigation` 把原部署 2k 的 300 维 actor 与本轮 303 维方向 actor 组合导出，两个部分均为学习策略。位置目标使用原导航 actor；方向目标仅在距终点 30 cm 以内切换到原有方向接近／停靠 actor。输入前缀和动作合同严格核对，1024 个观测核验分支输出一致。它只是组合导出，额外 PPO 步数为 0，两个完整父 checkpoint 保存于 `training_parents`，不是新的统一可续训 PPO 优化器。

`preserved_navigation_candidate` 保留 v5 定位；`preserved_turn_candidate` 为 v6 定位，仅增加一个纯航向精修候选并保留所有纠正上限和静态支持门限。v6 固定指令回放改善一例，也恶化其他几例，必须以闭环任务判断。两种模型均默认未通过部署验收。原导航权重不变不能保证输入分布变化后的行为相同，因此继续单独评估指令积分与历史避障任务。

## 录制实际输入审计与 IMU 近似对照

`REAL_SENSOR_INPUT_AUDIT.json` 由只读解析真实 rosbag2 SQLite 得到，原始文件未修改。配置快照明确：底盘 `odom_source=command`；`ekf.yaml` 同时融合 `/rl_navigation/odom_raw` 和 `/imu` 的转向角速度。记录中的 `/imu` 约 100 Hz，且已有新鲜有效 `/wheel/feedback` 与 `/wheel/odom`。这些反馈并不构成独立绝对真值，也没有证明轮径、极性、速度比例已经标定。

纯 `command_integrated` 是不使用 IMU 的退化对照，不能代表实际 EKF。新增 `command_linear_imu_yaw` 输入仅接受发布线速度和独立含噪陀螺仪角增量／角速度，不读取实际平移或参考位姿。0.8 融合权重和 0.003 rad/s 偏置是实验假设，不是从实车辨识出的完整 EKF 参数。固定集合包括 `imu_command_position_cases.json`、`imu_directional_cases.json`、`imu_unseen_position_cases.json` 和 `imu_online_cases.json`。

```bash
nice -n 19 /home/ubuntu/rl_learn/rl_learn_agv/.venv/bin/python \
  -m experiments.evaluate_docking_navigation \
  --candidate models/localization_mapping_20261015/preserved_turn_candidate/portable \
  --candidate-name preserved_turn_v6 \
  --baseline models/quality_recovery_20261014/portable \
  --baseline-name real_deployed_2k \
  --cases reports/localization_mapping_20261015/imu_unseen_position_cases.json \
  --output /tmp/rl_imu_new_map_repeat_20261015 --cpu-cores .25
```

原 2k actor 不支持方向任务，此处只比较位置目标。方向移动配对使用两个均支持 303 维方向输入的模型。各类条件分别统计，不将传感器来源不同、地图不同或重复开发任务混合为总成功率。

## 约 12 m 在线建图定位测量

`long_online_slam_2` 与 `long_online_slam_distance_prior_2` 使用两条相同的脚本传感器轨迹，未知初始地图、10 m 量程、8% 丢点、指令平移＋IMU 转向、+8% 平移尺度误差。后者只显式传入 `--slam-distance-prior-variance .0025`；不传参数时保留系统 YAML 的 0.5 默认值。这个实验不修改系统安装或实车配置，不算 RL 自主导航成功，也不能隔离回环的因果贡献。

```bash
source /opt/ros/humble/setup.bash
export PYTHONPATH=training:$PYTHONPATH
nice -n 19 /home/ubuntu/rl_learn/rl_learn_agv/.venv/bin/python \
  -m experiments.probe_long_online_slam \
  --bundle models/localization_mapping_20261015/selected_known_map/portable \
  --output /tmp/rl_long_slam_prior_repeat_20261015 --cpu-cores .25 --domain 193 \
  --slam-distance-prior-variance .0025
```

官方 [Karto 实现](https://github.com/SteveMacenski/slam_toolbox/blob/humble/lib/karto_sdk/src/Mapper.cpp#L622) 用这个值控制离开里程计预测搜索中心的惩罚；本轮改变它是针对几何退化的待验证对照。绝对位置不可观测时，强化里程计先验仍不能凭空消除里程计漂移。
