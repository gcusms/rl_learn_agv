> 最新多地图实现与使用入口见 [MULTIMAP_NAVIGATION.md](MULTIMAP_NAVIGATION.md)。历史 solid_* 场景的透明行人包络曾未被激光射线扫描，原有到达记录不能验证动态障碍激光感知；该问题已修复，后续结果使用 multimap_visible_*。

# 已知地图中的差速机器人强化学习导航

已实现有限视场激光仿真、全局地图规划、激光定位、PPO 训练与评估，以及 ROS 推理适配器。
底盘为用户确认的普通四轮差速底盘；雷达为用户确认的 SCLIDAR。
原有 agv/base_controller 仓库源码未修改。源码架构梳理见 SOURCE_REVIEW.md。

## 激光范围与模型

直接读取 agv/peripherals/config/lidar_filters_config_sclidar.yaml 的后方清除扇区，
有效角度为 −1.89159265 到 +1.93840735 rad，即 −108.38° 到 +111.06°，总计约 219.44°。
仿真没有后方测距；后方障碍不会出现在策略观测中。第一版禁用倒车。
URDF 固定关节链解析得到雷达相对 base_footprint 的位置为
(x,y,z)=(0.118389,0,0.116682537) m，yaw=0。仿真和 ROS TF 使用相同外参。
之前场景的雷达可视标记高度已修正。

training/config/lidar.yaml 设置 220 根仿真射线、0.10–12 m 量程、测距噪声与掉点率。
视场和安装位置来自源码；量程/噪声是可配置的保守仿真参数，尚未用实车测量标定。
摄像头和机械臂不进入第一版策略观测。

## 激光定位

默认后端 ros-amcl 使用实际安装的 Nav2 AMCL C++ 节点，并非读取仿真真值。
每个环境自动启动独立命名空间中的 map_server 和 AMCL，用生命周期服务配置/激活，关闭环境时回收进程。
输入是 YAML+PGM 已知地图、有限视场 LaserScan、带噪声/尺度误差的差速里程计以及 TF。
AMCL 输出 map→odom；策略位姿由该变换与里程计合成，并使用 AMCL 协方差。
传感器桥不发布 map→base 的真值变换。没有定位输出或变换过期时停止回合并报错，不退回真值。

使用 DifferentialMotionModel 和 likelihood_field_prob，并启用 beam skipping，减少动态障碍对地图匹配的干扰。
回合初始化使用预先给定的起点先验，仿真实际出生位姿另外扰动；不是每步喂入真实位置。
这是已知地图定位，采用现有 AMCL 链路；不重复运行建图 SLAM，也不是无初始位置的全局重定位。
AMCL 与已有 SLAM 一样依赖激光和地图匹配，但这里没有声称实现 slam_toolbox 算法。

particle 后端是轻量的似然场粒子定位，供快速 CPU 实验，明确区别于真正的 Nav2 AMCL。
它也只接收先验、里程计和激光，不接收当前真值位置。默认训练/评估使用 ros-amcl；本次正式模型用 particle 加速训练，再用真实 ros-amcl 独立验证。

## 导航训练接口

10×8 m 自定义地图包含房间、交叉通道和静态障碍；两个移动障碍不写入静态地图。
已知地图经 0.30 m 规划半径膨胀，用 A* 求全局路径；超覆盖栅格检查防止路径切穿转角。
这是训练环境内的全局规划器，实车仍可接收已有 Nav2 的 /plan。
局部目标点只选与估计位置之间在膨胀地图上可通行的前方路径点。

策略观测为 228 维：3 帧×72 个激光扇区距离 + 局部路径点 + 目标相对位置 +
里程计速度 + 上次动作 + 定位标准差 + 路径偏离距离。
所有目标/路径坐标都由激光定位估计位姿转换到机器人坐标系。
动作是两个归一化值，映射为 0–0.2 m/s 的前进速度和 ±0.5 rad/s 的转向角速度。
负向油门按等待处理，不能驶入后方盲区。机器人有速度响应延迟、加速度限幅和速度尺度随机化。

奖励使用估计位置沿路径的进度、到达目标、碰撞惩罚、近障碍惩罚、平滑和时间成本。
真值仅用于生成物理传感器、几何碰撞及独立评估指标，不进入策略观测。
动态障碍初始相位、速度、传感器误差和初始位姿按种子随机化；评估使用独立种子。

MuJoCo 中的机器人是按 URDF 外廓建立的平面 mocap 简化体；并非完整轮胎接触和电机 PID 动力学。
固定/mocap 物体接触可能被物理引擎省略，因此碰撞判定使用精确二维旋转矩形与墙体/圆形障碍的几何检测，
并在每个动作内做 10 次小步碰撞检查，避免穿墙。不能据此宣称已完成动力学标定或仿真到实机迁移。

## 运行

在 /home/ubuntu/rl_learn：

```bash
source training/activate.bash
python training/test_navigation.py

# 默认：真实 Nav2 AMCL 定位 + PPO 训练，100000 步
python training/train_navigation.py --curriculum

# 较快的粒子定位训练（不是 Nav2 AMCL），保存至独立目录
python training/train_navigation.py --localizer particle --curriculum \
  --steps 100000 --output training/runs/ppo_particle

# AMCL + 动态障碍评估训练好的策略
MUJOCO_GL=osmesa python training/evaluate_navigation.py \
  --localizer ros-amcl --policy training/runs/ppo_particle/policy.zip \
  --episodes 3 --output training/runs/ppo_evaluation

# 续训，使用完整路线（不再指定 curriculum）
python training/train_navigation.py --localizer particle --steps 100000 \
  --resume training/runs/ppo_initial/policy.zip --output training/runs/ppo_full_routes

# 交互场景。无 policy 参数时展示路径跟随基准控制器
MUJOCO_GL=glfw python training/run_navigation.py --localizer ros-amcl --static-only
# PPO 模型交互运行
MUJOCO_GL=glfw python training/run_navigation.py \
  --policy training/runs/ppo_initial/policy.zip

tensorboard --logdir training/runs
```

训练支持 --static-only、--curriculum、--resume、--seed 和 --max-episode-steps。
curriculum 使用全局路径上较短的起终点任务；它不是完整路线测试的成功保证。
模型旁的 config.json 保存扫描外参/角度/量程/观测配置；扫描配置不一致时评估/续训拒绝加载。
初始模型 policy.zip 不等于成熟可用的导航策略，评估结果见各 runs 子目录 metrics.json。
PPO 按整段 rollout 采样，实际步数可能略大于 --steps；config.json 记录实际步数。

## 以到达最终目标为准的当前版本

用户允许碰撞：碰撞不结束回合，记录接触次数和步数，阻挡平移，只有足迹允许时才继续转向；不允许穿墙。
最终成功率要求回合在 1000 步（200 秒仿真时间）内结束，且实际终点距目标 ≤0.30 m。
策略使用激光定位估计距目标 <0.12 m 来停车，为定位误差留余量；实际距离仅用于独立验证。

当前模型：runs/goal_ppo/policy.zip。先用 18 条已知地图路径跟随示范预训练 actor（60 轮），
再进行 20224 步 PPO 更新。示范只用于训练；正式评估中动作全部来自 PPO 网络，没有基准控制器接管。
训练是完整路线、有未知动态障碍的任务。正式测试用同一张地图的三条完整路线、独立随机种子，
随机化障碍相位/速度、出生位姿和传感器/运动误差；这不代表任意新地图或任意起终点的成功率。

本次真实 Nav2 AMCL 测试成功 12/12（100%），快速粒子定位测试成功 26/30（86.7%）。
两组回合均发生接触；这些数字是允许碰撞的最终到达率，非无碰撞成功率。
AMCL 测试种子 1000–1011，粒子测试种子 3000–3029；18 条示范种子 100–117。

成功率及碰撞统计：goal-navigation-validation.json；原始逐回合记录：
runs/goal_amcl_evaluation/metrics.json 和 runs/goal_particle_evaluation/metrics.json。
MuJoCo 实际成功轨迹的视频、可拖动进度条的地图回放和终点截图保存在 runs/success_demo/。
视频重放真实 ros-amcl 评估中记录的状态，不是手工设定成功路线。

```bash
source training/activate.bash
# 查看成功视频，也可以直接打开 runs/success_demo/replay.html
# 在 MuJoCo 中重新运行训练好的策略（动态障碍、真正 AMCL 激光定位）
MUJOCO_GL=glfw python training/run_navigation.py \
  --policy training/runs/goal_ppo/policy.zip --localizer ros-amcl --task 0 --seed 1000 --episodes 1

# 复现正式 AMCL 测试
python training/evaluate_navigation.py --policy training/runs/goal_ppo/policy.zip \
  --localizer ros-amcl --episodes 12 --seed 1000 --max-episode-steps 1000 \
  --output training/runs/goal_amcl_repeat --record-success

# 从示范到强化学习复现训练
python training/warm_start_navigation.py --episodes 18 --epochs 60 \
  --output training/runs/goal_navigation
python training/train_navigation.py --localizer particle --steps 20000 \
  --resume training/runs/goal_navigation/warm_start.zip \
  --output training/runs/goal_ppo_repeat --max-episode-steps 1000
```

当前版本 11 项行为检查通过，包括碰撞继续但不能穿墙、有限雷达视场和观测不读取真值。
目前仍存在顶墙超时失败；这是一版决策层导航模型和简化平面仿真，暂不驱动实车。

## ROS 接口

ros_policy_node.py 使用 /scan、/odom、/plan、/amcl_pose 和 map→base_footprint TF 构造同样的观测。
实机 0–2π 扫描在跨零度的有效扇区重新采样；缺失扇区不补成 360° 感知。
历史激光按策略 5 Hz 更新，避免与训练帧间隔不同。
需要现有 AMCL、全局规划和正确坐标系已经运行；地图参数应指向实际场景地图。

```bash
python training/ros_policy_node.py \
  --policy training/runs/ppo_initial/policy.zip \
  --map /absolute/path/to/real_map.yaml
```

默认输出 /rl/cmd_vel，后续通过显式速度仲裁器接入底盘；不能让原有 TEB/DWB 与策略同时控制同一速度话题。
代码包含激光/里程计/TF 过期停车、定位不确定度过大停车、已知地图足迹及观测障碍的短时预测检查。
该适配器的实车控制闭环尚未验证，初始训练策略仍需完成验证；本次没有连接或驱动实车。
底盘通信超时参数仍需在真正接入时启用，现有默认 cmd_vel_timeout=0。

## 早期流程验证（历史结果）

- 有限视场、前方可见/后方不可见障碍、碰撞检测、真值观测隔离、路径通行性、等待动作和回合超时测试。
- 实际 Nav2 AMCL 后端完成 256 步、多回合、动态障碍 PPO 功能训练，输出 runs/amcl_smoke/policy.zip。
- 轻量粒子定位后端完成约 2 万步初始 PPO 训练，输出 runs/ppo_initial/policy.zip。
- 实际 AMCL 下静态地图路径跟随基准抵达目标，行驶约 9.69 m，
  单次测试平均定位误差约 0.019 m、最大约 0.052 m。这是基准控制器结果，不是 PPO 成功率。
- PPO 独立动态障碍评估见 runs/ppo_particle_evaluation 和 runs/ppo_amcl_evaluation；
  evaluation.png/trajectories.json 展示估计位姿与仅用于诊断的真值轨迹。
- 依赖版本见 requirements.lock.txt；系统 Nav2 和 ROS 消息/描述包的基础验证见 environment-report.json。

实际部署前还需要完整路线/不同障碍轨迹的大规模评估和真实里程计、量程、动作延迟标定。

本次初始 PPO 评估：粒子定位 3/3 回合碰撞，实际 AMCL 1/1 回合碰撞，成功率均为 0。
实际 AMCL 下该 PPO 轨迹平均定位误差约 0.029 m；目前失败主要表现为策略导航失败，
不能用良好的定位误差掩盖碰撞结果。该模型仅作为训练流程起点，尚不能部署。
早期汇总报告见 navigation-validation.json；最新结果以 goal-navigation-validation.json 为准。

## 路径、雷达与 URDF 模型展示

MuJoCo 场景按 agv_description/urdf/agv_chassis.urdf 导入 20 个视觉网格，保留各 mesh 的比例、安装位姿和材质。
包含底盘、四轮、SCLIDAR、屏幕、麦克风、机械臂、摄像头和夹爪；机械臂采用 URDF 零关节姿态。
视觉模型不参与激光射线或碰撞计算，导航仍使用原有决策层平面运动和几何足迹。
青色线为已知地图的 A* 全局规划路径，橙色线为实际行驶轨迹，绿色标记为目标。
绿色射线和红色返回点显示本次真实扫描数据：只显示有效的 219.44° 视场，不补全后方盲区。
规划路径由全局规划器产生，路径附近的速度和转向由 PPO 决策。
最新带模型近景的展示见 runs/visual_navigation_demo/successful_navigation.mp4 和 replay.html。

```bash
source training/activate.bash
MUJOCO_GL=glfw python training/run_navigation.py --policy training/runs/goal_ppo/policy.zip \
  --localizer ros-amcl --task 0 --seed 1000 --episodes 1
```

## 纠正示范与后续 PPO 训练

最新候选模型 runs/navigation_refined/policy.zip 使用原策略采集 36 回合纠正示范（种子 300–335，
25% 动作由旧策略执行，其余为示范动作；所有记录的训练标签均为示范动作），重新训练 actor，
再进行 20224 步 PPO 更新，学习率 3e-5，并用示范损失约束 actor，保留等待/转向行为。
正式评估仍只调用模型，不调用基准控制器。
奖励改为连续路径投影的剩余距离，移除最近离散路径点切换和裁剪引入的绕圈奖励问题；
碰撞成本从每步 0.5 提高到 1.0，碰撞仍不终止回合。

视频及窗口显示策略要求的前进速度、实际速度和停车来源。policy_stop 表示策略要求
前进速度 ≤0.01 m/s；collision_blocked 表示策略仍要求前进但发生接触且实际前进速度 <0.01 m/s。
stop_near_obstacle 统计前方距离 <0.8 m 且未接触时的策略停车动作；可能同时有转向，
不等同于证明识别到了动态障碍，也不等同于所有障碍都能避开。
正式测试与原始逐回合结果见 runs/refined_particle_evaluation 和 runs/refined_amcl_evaluation。

```bash
source training/activate.bash
MUJOCO_GL=glfw python training/run_navigation.py --policy training/runs/navigation_refined/policy.zip \
  --localizer ros-amcl --task 0 --seed 5000 --episodes 1
```

本轮正式评估：旧模型粒子定位 11/12，新模型粒子定位 12/12；新模型真实 AMCL 12/12。
均为允许碰撞的最终到达率，不能解释成无碰撞率或任意场景保证。
新演示保存在 runs/refined_demo；视频约 28.4 秒时策略和实际前进速度均为 0，累计接触数为 0。
汇总见 refined-navigation-validation.json，速度对比见 runs/refined_demo/control_diagnostics.png。

## 实体场景与行人让行（当前默认）

训练、评估和交互入口现在默认使用 assets/physical_scene.xml，由 MuJoCo mj_step
进行真实动态刚体接触计算，不再每步强制更新障碍物位置。历史平面版可用 --legacy-planar 复现。
机器人有 x/y 平移和 yaw 转动动态关节，外力伺服跟随策略速度；保持在水平地面上。
URDF 视觉网格的凸碰撞包络及底盘盒参与实体接触。行人有 70 kg 模拟质量、0.25 m
半径的实体包络和人体外观，以 0.5–0.85 m/s 向随机可达目标移动，每个平移轴的速度变化限制为 1.8 m/s²。
墙、行人与机器人之间由 MuJoCo 处理接触，并以 5 ms 步长检查运动；额外回退保护防止可见重叠。
行人接近机器人时先减速、让行，等待后可侧移给小车留空间；道路恢复通行后继续移动。
走路、让行、侧移、受阻和短暂停留状态分别记录，不把墙边受阻当成向小车让行。
行人控制器知道模拟场景物体位置；PPO 策略仍只使用有限视场激光、里程计和激光定位，
不会获得行人真值位置。AMCL 仍匹配已有静态地图。

这是用于导航的受平面约束动态刚体模型，不包含已标定轮胎摩擦、电机 PID 或人体步态动力学。
行人圆柱是保守的整体碰撞/激光包络，外观为人体，未模拟激光穿过双腿间隙。
旧环境的 12/12 结果不能作为该实体环境的成功率；新结果需查看 solid_*_evaluation。

```bash
source training/activate.bash
python training/generate_physical_scene.py
python training/test_physical_navigation.py
MUJOCO_GL=glfw python training/run_navigation.py --policy training/runs/navigation_refined/policy.zip \
  --localizer ros-amcl --task 0 --seed 6300 --episodes 1
```

物理验证覆盖：行人遇车停车、车不能穿过停止行人、行人之间不能穿透、原生墙体接触阻挡
强制运动、随机出生不重叠和原生时间积分。数值容差 0.1 mm，避免把软接触求解的微小误差
说成绝对数学零穿透。最终实体视频和完整验证报告将在 runs/solid_demo 中保存。

本轮实体场景复测：快速粒子定位 9/9、真实 AMCL 3/3 到达目标，未触发穿透回退保护。
既有策略在新场景重新验证，尚未在该新场景重新训练；旧成功率未沿用。
实体演示视频、让行停车截图与验证报告见 runs/solid_demo/。
