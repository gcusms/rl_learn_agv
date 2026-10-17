# 实车录制版本的 ROS2 适配修改

本目录是可合并到现有 `rl_navigation` 的修改文件，沿用实车工程的 package.xml、setup.py、传感器和底盘程序。没有向车辆写入文件，也没有启动车辆上的 ROS 节点。

合并以下四个文件，并一起使用对应的 training 源码：

- `model_node.py`：保留目标 yaw；父进程记录实际发布命令；校验前加入明确选择的 model_source。
- `inference_worker.py`：持久推理子进程，下一帧前恢复父进程发布历史。
- `execution_feedback.py`：同步上一次实际发布动作、停车原因及等待状态，保留每帧只记录一次的约束。
- `model_bundle.py`：校验运行时、维度、动作、全部文件散列和配套观测合同。

2026-10-17 补充：适配器现明确支持 `neural_navigation.aligned_release_runtime.AlignedReleaseRuntime`。新联合保留候选 `full_navigation_joint_linear_s231017/portable` 使用该版本，仍为625维，但风险历史包含同一帧短时防碰撞证据；旧625维 `ReleaseRuntime` 的特征语义不同，不能互换。配置应使用新bundle的固定manifest SHA256，并指定：

```json
{
  "sensor_core": "learned-terminal-goal-pose",
  "required_runtime": "neural_navigation.aligned_release_runtime.AlignedReleaseRuntime",
  "observation_dimensions": 625,
  "action_choices": 12
}
```

现有配置和权重保持；上述配置还需由现有configure工具补全bundle路径及版本身份。新候选尚未通过完整物理验收，默认shadow。本次6项无ROS检查通过，包括真实新bundle准入、旧625/300维兼容、错误运行时拒绝，以及重新计算哈希后仍拒绝新旧625维语义混用。检查仅读取合同与哈希，没有加载Torch、MuJoCo或ROS，没有在车上发布控制命令。实际发布历史的纠正仍不等同于收到编码器或下位机执行确认。

## 2026-10-16 实车验证候选

`models/stop_feedback_20261015/release_context_bc2000_s231016/portable` 必须使用：

```json
{
  "sensor_core": "learned-terminal-goal-pose",
  "required_runtime": "neural_navigation.release_runtime.ReleaseRuntime",
  "observation_dimensions": 625,
  "action_choices": 12,
  "route_mode": "shortcut"
}
```

另需填写 bundle 绝对路径、manifest_sha256 和版本身份字段；交付包的 configure.py 会自动生成。625 维包括旧风险历史与当前扫描质量合同，不能把新 actor 放入旧 300/303/481 维 bundle。新模型 `deployment_approved=false`，用于受控验证，默认 shadow。沿用现有 `mode=execute` 执行入口时，需要保留原工程的速度仲裁、取消和超时停车。

本轮 4 项无 ROS 回归检查通过：625 维候选与 300 维回退包校验、错误合同拒绝、外部停车历史纠正、下一帧记录不重复。没有在目标车和 ROS 节点内完成端到端验证。

## 兼容旧版本

2026-10-14 原质量恢复 2k 包采用 `native`、`experiments.quality_sensor_runtime.QualitySensorNavigationRuntime`、300 维；必须固定 manifest SHA256。原方向目标试验采用 `odom-goal-pose`、`experiments.goal_pose_navigation.GoalPoseRuntime`、303 维，仅 shadow。终点停靠包采用 `learned-terminal-goal-pose`、`experiments.docking_navigation.DockingRuntime`、303 维，v2 在估计目标距离 0.10 m 内启用学习到的停靠 actor。

## 实车记录语义

建议 `auto_initial_pose=false`，手动设置可信的 2D Pose；目标箭头表示目标姿态。发布历史指节点实际发出的速度命令，不能代替编码器测量的真实车速。外部停车细分原因暂归入 `other`；等待时间从下一次传感器处理时刻修正，尚未实现完整的逐次发布事件时间线。此修改不能替代端到端延迟与动作保持周期的测量。

雷达缺失阈值、车体几何、速度上限、轮速源和固件没有自动修改。真实扫描缺失超过当前 8% 门槛时仍会停车，应在记录中区分 `INVALID_SCAN_COVERAGE`、推理超时和策略主动等待。不能仅放宽门槛来获得前进效果。全新无地图导航、自动绑架重定位及陌生地图泛化并未由本轮验证。
