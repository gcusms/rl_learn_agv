# 对实车录制版本的 ROS2 适配修改

这里是录制快照的修改文件，不是完整的 ROS2 安装包。原始 package.xml、setup.py 和传感器/底盘程序仍使用实车工程。文件没有写入真实车辆，也未在 ROS 节点上执行。

在独立实车测试分支上合并这四个文件到 `rl_navigation` Python 包：

- `model_node.py`：按新合同保留目标 yaw、记录父进程实际发布命令，新候选只允许 shadow。
- `inference_worker.py`：下一帧前恢复父进程实际执行历史。
- `execution_feedback.py`：新增的实际命令历史恢复方法。
- `model_bundle.py`：校验新 303 维观测、两份特征合同和文件散列。

需要同时使用本分支的 training 源码，并将 `model_source` 指向对应目录。新的 model.json 需要包括 bundle 的绝对路径、分支及 commit、`sensor_core: odom-goal-pose`、`required_runtime: experiments.goal_pose_navigation.GoalPoseRuntime`、`observation_dimensions: 303`、`action_choices: 12`、`route_mode: shortcut` 和 manifest 的 SHA256。不要用新 actor 覆盖旧 bundle 中的 actor，也不要复制旧 300 维合同。

新的 bundle 默认 `deployment_approved=false`，这里只做 shadow 验证。先手动设置可信初始位姿，建议 `auto_initial_pose=false`；目标方向是停靠要求，初始位姿方向才是当前姿态。旧 bundle 仍按旧的位置目标处理。

发布历史指上游实际发出的速度命令，不代表真实车速。策略观测中的速度仍由所选里程计/反馈提供，需要保留录制并分别检查。动作历史修复也没有消除全部 ROS 推理间隙：必须再测端到端耗时和控制保持周期。

雷达缺失阈值、底盘几何、速度上限、轮速源和固件未在此自动修改。已有编码器反馈可用于后续标定对照，不要求这次加装新硬件。
