# 轮速与 IMU 输入候选

本次录包已有轮反馈，底盘运行参数却是 `odom_source=command`。先保留原配置作对照，再显式选择编码器输入；没有修改固件、轮径、轮距、轴距或速度上限。

## 复用现有 ROS2 EKF

实车录制快照的 `navigation.launch.py`、`base.launch.py` 已支持 `odom_source:=encoder`。在原来的完整启动命令中传入这个参数，`/rl_navigation/odom_raw` 就由底盘选择编码器速度。原有 EKF 可以继续融合这个话题的速度与 `/imu` 的 Z 角速度。启动后要从运行参数确认选择生效，不能只改 YAML 又被 launch 默认值覆盖。

另一个明确的输入方式是把现有唯一 EKF 的参数替换为 [ekf_encoder_imu.yaml](ekf_encoder_imu.yaml)，直接从 `/wheel/odom` 取速度。两种方式选其一即可；不要另启动第二个 EKF 或将同一编码器生成的位置、速度当成两份独立测量。麦轮实际测得的 `vy` 保留，控制动作仍不增加横移。

[chassis_encoder_override.yaml](chassis_encoder_override.yaml) 是底盘参数覆盖示例。录包几何为轮径 0.10m、轴距 0.20m、轮距 0.22m，本次保留。轮速换算与消息数值一致不等于这些尺寸已经用实际路程标定。

## 不使用 ROS 滤波节点的纯 Python 方式

`neural_navigation.measured_odometry.MeasuredOdometry` 接收车体坐标系的实测 `(vx, vy, wz)` 与 IMU `gyro_z`，按采集时间因果融合，不接受速度命令作为运动反馈。IMU 应先经过已知外参转换；本次录包 IMU 外参仅绕 Z 旋转，平面 Z 角速度方向与底盘一致。

```python
from neural_navigation.measured_odometry import MeasuredOdometry, MeasuredNavigationAdapter

motion = MeasuredOdometry()
navigation = MeasuredNavigationAdapter(existing_runtime, motion)
# 按同一采集时钟顺序喂入测量；不要把 scan 后的测量用于 scan 时刻。
motion.update(t, measured_vx_vy_wz, wheel_stamp, body_gyro_z, imu_stamp,
              wheel_valid=feedback_fresh_and_valid, imu_valid=imu_valid)
result = navigation.step(scan, scan_stamp, received_timestamp=received_stamp)
```

`existing_runtime` 可以是原位置目标或方向目标运行时。融合与扫描时间需按顺序协调；这里没有自动 ROS 订阅、TF 发布或串口发送。调用方仍须发布返回命令、反馈实际发布历史并保留原看门狗。不要同时叠加该融合与另一个融合里程计而将它们当作独立测量。

缺失/过期/未来数据、非有限值、超过 100ms 的运动积分空档会使输出不可用；空档后需要显式恢复连续运动参考，不能继续假定缺失期间没有移动。适配器在输入不可用时不调用策略，返回停止。新鲜输出的协方差也是未标定估计，不能据此宣称绝对定位可信。

## 短测验证

同一起始位置分别使用原输入与编码器输入，记录固定距离直行、左右转、停车和受阻减速片段。用地面测量点/已测朝向作为独立参考，检查轮序、比例、时间对齐与制动响应；不要用指令积分或该融合器自身当真值。保留原录包的话题、原始扫描、反馈质量和实际命令发布记录。

之后在已知地图上，临时放入未入图障碍物：检查物理车不动时 `map→odom` 是否异常移动、移走障碍后能否恢复静态匹配。再进行方向目标和原地改朝向测试。新射线定位合同与输入适配都属于实验候选，尚未完成实车闭环验收。

方法依据：[robot_localization 官方传感器融合说明](https://github.com/cra-ros-pkg/robot_localization/blob/rolling-devel/doc/configuring_robot_localization.rst)。本项目的轻量融合器不是该 EKF 的完整移植，也不增加绝对位置来源。
