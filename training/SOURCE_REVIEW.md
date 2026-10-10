# AGV 强化学习训练环境与源码梳理

本目录用于 CPU 强化学习实验。已通过 MuJoCo 步进/动态障碍射线/离屏渲染、PPO 256 步短训练、ROS 自定义消息与 cv_bridge 导入检查，pip check 无依赖冲突。
仿真步进、PPO 短训练通过验证不代表导航策略已经训练成功。
原有 agv、base_controller 仓库源码未修改。软件版本与 ROS 包状态见 environment-report.json。

## 使用

在 /home/ubuntu/rl_learn 执行：

```bash
source training/activate.bash
python training/verify_environment.py
# 额外检查离屏渲染（需要系统 OSMesa 库）：
MUJOCO_GL=osmesa python training/verify_environment.py --render
# 查看训练日志时：
tensorboard --logdir training/runs
```

环境为 Python 3.10 的 .venv，独立于系统 Python 包；激活脚本通过 ROS setup 加载 Humble 与本地消息包，训练依赖安装在 .venv 中。
NumPy 固定在 1.26.4，避免将 ROS 的既有二进制扩展直接切换到 NumPy 2 ABI。
安装 CPU torch 的索引单独使用，不给其他依赖混用索引：

```bash
.venv/bin/pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r training/requirements.txt
```

## 现有源码与接口

- base_controller：STM32F407 固件、FreeRTOS、电机编码器/PID、串口协议和 Python SDK。
  SDK：base_controller/sdk/ros_robot_controller_sdk.py；默认 /dev/rrc、1000000 baud。
  协议起始字节 AA 55，包含长度、功能、数据与 CRC8；电机功能号 3。
  ROS 的 MotorsState 中每个电机的 rps 转为串口指令。
  motor_safety.c 包含 K1/K2、PD3 联锁与释放延时；get_motor_status 查询的是联锁/计数，非轮编码器里程计。
- agv/driver/ros_robot_controller：底板 ROS 节点及 SDK；它与 base_controller 中有同类副本，
  后续修改协议要明确部署哪份节点，避免版本不一致。
- agv/driver/controller：Twist 到电机转速；支持 Mecanum、Differential、Acker 等分支。
  无命名空间时订阅 /cmd_vel、/controller/cmd_vel、/app/cmd_vel；最后一项还有阿克曼特有转换。
  输出 /ros_robot_controller/set_motor，发布 /odom_raw，由 EKF 融合后输出 /odom。
  /cmd_vel 路径限速 x/y ±0.2 m/s，角速度 ±0.5 rad/s；/controller/cmd_vel 并没有相同限幅。
- agv/peripherals：激光扫描通常为 /scan_raw → 过滤 → /scan，坐标 lidar_frame；具体启动依赖 LIDAR_TYPE。
  摄像头由 DEPTH_CAMERA_TYPE 决定；ascamera 的 RGB/depth 话题重映射到 depth_cam/rgb0/image_raw、
  depth_cam/depth0/image_raw 和 depth_cam/depth0/points 等。设备型号及实际话题要实机确认。
- agv/navigation：map_server + AMCL + Navfn 全局规划 + TEB/DWB 局部控制 + velocity_smoother。
  navigation_base.launch.py 将局部速度重映射为 cmd_vel_nav，再平滑到 cmd_vel。
  Navfn use_astar=false，当前配置采用 Dijkstra；全局/局部代价地图使用激光观测。
- agv/slam/maps：map_01、map_02、map_03、test 的 YAML+PGM；解析时必须使用 resolution/origin/
  negate/occupied_thresh/free_thresh，PGM 的图像行方向与地图 y 轴需要转换，未知区域需明确处理。
- agv/simulations/agv_description：agv_chassis.urdf、底盘/四轮/雷达/机械臂/相机 mesh 以及显示节点。
  base_footprint → base_link → lidar_link → lidar_frame；相机在机械臂 link4 上，安装姿态会随机械臂变化。

## 针对已知地图、未知动态障碍的建议架构

保留已有地图定位和全局规划，把强化学习范围限定为局部路径跟随、绕行、减速、等待。
已知地图通过全局路径/局部地图窗口进入策略，动态障碍通过实时传感器进入策略。
不要仅训练激光到最终目标，否则缺少全局路径信息时在 U 形墙、死路等环境容易陷入局部最优。

```text
静态地图 + AMCL/TF 定位 → 全局路径规划 → 前方局部路径点
实时激光历史 + 自身速度 + 局部路径点/局部地图 → PPO 局部策略
PPO 速度指令 → 限速/碰撞检查/失联停车 → 底盘 Twist → 电机协议
```

已确认实车为普通四轮差速底盘，动作固定为 (vx,wz)，vy=0。
用户授权自定义地图，新增 assets/decision_map.yaml + PGM（10×8 m、0.05 m/像素）、
对应 decision_scene.xml 以及 scene.json 中的起终点和动态障碍轨迹。
动态障碍不会写入静态地图。robot 为按 URDF 外廓建立的 mocap 简化体，需要后续 Gym 环境
按差速运动学更新；目前场景演示没有策略、轮胎动力学或自动导航。

```bash
python training/generate_scene.py
MUJOCO_GL=osmesa python training/preview_scene.py
# 桌面交互窗口（关闭窗口结束）：
MUJOCO_GL=glfw python training/preview_scene.py --viewer
```

第一版策略：MLP [64,64] 或 [128,128]；4 个并行 CPU 环境作为起点，按实际吞吐再调整。
观测包含经清洗和降采样的激光距离、3–4 帧历史、自身速度、前方路径点在机器人坐标系的位置；
若需要显式地图推理，再加局部占据栅格。输入历史用于判断障碍物接近/远离，不只观察单帧距离。
差速动作 (vx,wz)，麦轮可选 (vx,vy,wz)，阿克曼动作需要遵守不能原地转向等约束。
奖励使用沿路径的进度、到达目标、碰撞惩罚、近障碍惩罚、动作平滑和时间成本；
动态障碍阻路时允许停等，不要求机器人无条件持续前进。

MuJoCo 场景由地图静态碰撞几何、简化底盘模型与动态障碍构成；用 ray casting 生成激光观测。
可以先用平面运动学底盘/速度响应模型训练决策，之后验证轮接触动力学。
前者不是已校准的轮胎动力学或电机 PID 模型，不能据此宣称仿真到实机已经一致。
随机化障碍初始位置、速度、出现时刻、传感器噪声、动作延迟和速度响应。
训练/评估分开障碍轨迹与随机种子；记录成功率、碰撞率、超时率、时间、距离和最小间距。
策略只能观察传感器可见的数据；仿真中隐藏的动态障碍真值仅用于碰撞判断/评分，不直接喂给策略。
第一版无需训练 RGB 感知；若激光高度覆盖不到低矮/悬空物，再考虑深度相机观测融合。

## 已发现的对接问题

1. odom_publisher_node.py 的 cal_odom_fun 用命令速度积分位置，不读实际编码器反馈。
   普通模式没有把 linear_y 积分进 x/y，仅 multi_control_only 模式有；麦轮横移会影响定位一致性。
   固件内部编码器闭环不代表 ROS 已有编码器里程计。
2. URDF 底盘 box 约 0.394×0.329 m，平面半对角线约 0.257 m；Nav2 robot_radius=0.15 m
   不能包住此矩形。部署前应按真实外廓设置 footprint；机械臂突出部分也需要计入。
3. URDF 底盘和轮子缺少 inertial，mesh 使用 package:// 路径；不能直接视为现成 MuJoCo 动力学模型。
   应解析资源路径、补充质量/惯量、执行器与轮接触；决策实验可另建简化 MJCF。
4. 标准控制节点 cmd_vel_timeout 默认 0，现有固件联锁也不等价于通信超时停车。
   真机策略对接时应启用超时停车、传感器/TF 过期停车，并确保只有一个速度控制源。
5. 部分 launch 路径写死 /home/ubuntu/ros2_ws/src，还有外部 third_party 驱动。
   need_compile=True 依赖安装后的包路径；MACHINE_TYPE、LIDAR_TYPE、DEPTH_CAMERA_TYPE、HOST、MASTER
   等环境变量需要确定，不应仅根据 URDF 假定实车类型。
6. 现有 sim=true 分支主要传仿真时间/速度参数，仍包含串口、雷达、相机等实机启动项；
   未找到现成 MuJoCo 场景或 ROS-MuJoCo 桥。不能直接以 sim=true 启动完整离线训练。
7. ROS 导航包是否完成系统安装以 environment-report.json 为准；源码尚未全量构建，
   摄像头/雷达厂商驱动、硬件话题和定位闭环未实机验证。

底盘与初始测试地图已确定，下一阶段实现 map→MuJoCo 场景、Gymnasium reset/step/奖励，
用固定测试集对比现有 TEB/DWB，再对接 ROS 局部控制接口。当前已实现导航训练接口并保存初始策略，实际收敛与评估结果见 README.md 和 runs 下的报告。

ROS 本地基础构建选择 ros_robot_controller_msgs、robot_sensor_msgs、agv_description，
产物位于 training/ros_install；激活脚本会加载它们。完整 AGV 应用和外部传感器驱动尚未构建。

安装结果：Nav2 navigation2/nav2_bringup、robot_localization、laser_filters、joint_state_publisher、xacro、OSMesa/GLFW 已就绪。
requirements.lock.txt 保存本次环境的完整版本；torch 带 +cpu 后缀，重建时仍需使用 CPU wheel 索引。
自定义场景 1000 步运行与离屏渲染通过；按 0.27 m 保守半径膨胀地图后，3 组起终点均在同一可达区域。
这只验证静态通行性，不代表动态避障策略已通过评估。

额外注意：navigation/setup.py 当前只打包 launch 顶层文件，没有打包 launch/include 子目录，
后续使用安装后的导航 launch 之前应修正资源安装规则，否则可能找不到 include 文件。
map_02.yaml 与 map_03.yaml 当前都引用 map_01.pgm，只改变了地图 origin；应确认这是否为预期。
