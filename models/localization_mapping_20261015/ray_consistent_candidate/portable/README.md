# 射线一致性定位候选

实验候选，`deployment_approved=false`。由 `temporal_native_candidate/portable` 的原 actor/router/方向停车合同生成，本轮没有重新训练策略权重。

- 新定位合同：`ray_consistent_static_tracking_v1`，Python 实现，不要求新增编译。
- 保留 303 维观测、12 个动作、最高前进 0.2m/s 与角速度 0.5rad/s、方向目标；不增加横移控制。
- 需要已知地图与可靠初始地图位姿，不提供全图自动恢复或在线 SLAM。
- 轮速＋IMU 适配为显式调用选项；本包不会自动改变 ROS/底盘里程计来源。

加载前使用 `neural_navigation.goal_pose_runtime.check_files` 核验 manifest。运行时可用 `resolve_runtime(manifest['required_runtime'])`，与父版本方向目标入口一致，需要配套的本工作树新定位源代码。源文件哈希记录于 `MODEL_CARD.json`。

[完整结果与限制](../../../../reports/localization_mapping_20261015/real_input_repair_20261015/REPORT.md) · [输入接入说明](../../../../deployment/ros2_recording_adapter/examples/measured_motion/README.md)

22 条固定仿真验证轨迹的位置误差 P95 仅从 18.51cm 小幅降至 18.04cm；射线检查增加定位耗时，真实录包可接受的静态更新减少，可能带来更多等待。必须进行参考位姿与闭环导航回归后再决定采用，不能只把该包替换到实车视为已解决定位。
