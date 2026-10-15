# 换机时的冻结进度

`SNAPSHOT.json` 记录归档时点和实际状态；`paired/RESULTS.json` 只含已经结束的导航条目，并附逐条哈希对应的完整 NPZ。归档时后台评估尚未完成，不能把部分条目当作完整配对结论。

4,000 步训练已完成；最终网络与优化器见 `models/stop_feedback_20261015/ppo_4000_s221015/policy.zip`。训练回合统计不是验证成功率。原、新版本在 `motion_profile_015` 均发生碰撞，目前没有通过验收。

`pipeline_status.json`、日志和 `original_machine_service.txt` 是原电脑历史记录，其中绝对路径、PID 和 systemd 单位不会自动迁移。原机仍在评估，后续结果不会自动更新此 Git 快照。

新电脑按根 README 第 7 节重新编译原生包并完整运行相同 12 例、24 条配对任务；不必重复训练，也不要将缺少的条目补写进冻结结果。整个实验不自动替换原部署模型。
