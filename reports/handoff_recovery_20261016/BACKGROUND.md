# 旧电脑后台进度快照

这是换机时的一致副本，未完成的物理对照不算模型成绩。

原服务：`rl-training-stall-probes-20261016.service`；状态：`active/running`。

完整旧命令先运行 `experiments.probe_recovery_prefix`，完成后才运行三个有效 holdout 前缀的候选对照。Python 环境与 systemd 服务不随 Git 迁移；新电脑按根 README 的命令、使用新输出目录执行。

`partial_stall_probes/RESULTS.json` 与所列完整单例轨迹已复制并核对哈希；仍在写入的源目录没有直接上传。
