# 实车定位回放输入

来自用户 2026-10-14 的 map_01 录制。包含 1516 条决策、22 个任务代次、1911 条轮反馈以及原传感器数组和地图。没有独立参考位姿，人工 2D Pose 是初始化/重置，不能当作定位真值。

`sensors.npz` 和准备后的 PGM/YAML 按字节复制；`decisions.jsonl`、`feedback.json`、`session.json` 只保留当前定位及全图恢复回放所需字段，数值、时间、顺序和任务边界不变。完整原始文件及紧凑输入的 SHA256 见 [MANIFEST.json](MANIFEST.json)。完整 rosbag、TF 录制、相机数据和策略全部输出没有在此重建。

`analysis/sensors.npz` 中里程计行顺序为 `[接收相对时间, 采集相对时间, x, y, yaw, vx, vy, wz]`，IMU 为 `[接收相对时间, 采集相对时间, gyro_z, ax, ay]`。决策中的 `scan_stamp_ns/decision_stamp_ns` 属 ROS 纪元，`result.timestamp/received_timestamp` 属运行时间轴，不直接跨时间轴相减。

导出入口为 `tools/create_recording_fixture.py --record <解压后的单次记录目录> --analysis <已提取传感器目录> --output <新目录>`。回放命令见[根 README](../../../README.md)。这些片段已用于开发，不称为未见测试。
