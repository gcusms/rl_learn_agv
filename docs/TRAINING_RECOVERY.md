# 当前专项训练的重启恢复

此服务仅管理已授权的clearance_motion_300k训练及其开发评估，不发送实车指令、不更换默认模型。

- 用户服务：agv-clearance-training-recovery.service。当前用户Linger已开启，服务启用到default.target。
- 状态入口：training/runs/optimization_stage3_20261011/pilot_clearance_motion_300k/recovery.json。current字段指向最新恢复分支；base_steps加该分支progress.json.steps是当前采样进度。
- 原中断记录与恢复分支分目录保存；只有完整ZIP且内部步数匹配的检查点可继续。恢复优化器/网络，但丢弃未保存rollout并重建环境、换独立种子；不能声称逐位复现。
- 恢复前检查已有奖励、物理和推理源码哈希；改变则拒绝自动继续。只有编排脚本明确允许升级。
- 训练完成后评估中断，可继续缺失评估并校验已有产物模型/场景哈希；部分产物改名保留。所有筛选门槛仍然要求原始接触为0并安全到达。
- 服务异常失败最多尝试3次/180秒；正常完成后服务退出，是否通过导航门槛见最新分支status.json，服务成功退出不等于模型通过验收。

查看：`systemctl --user status agv-clearance-training-recovery.service`；日志：原目录recovery.log及各分支training.log。

用户要求暂停时，应执行`systemctl --user disable --now agv-clearance-training-recovery.service`，保留检查点，并将会话目标暂停；否则只停进程而保持开机启用会在重启后恢复。用户要求继续后再启用启动。

目前做过损坏检查点回退、源码改变拒绝恢复和旧/部分评估不复用的3项检查，并从真实中断的6万步检查点实际恢复。尚未通过人为重启主机/断电破坏文件做端到端恢复测试。没有硬件部署验收。

## 2026-10-11 conservative resource profile

User reported desktop freezing. Applied persistent service Nice=15, CPUAffinity=2-5, CPUQuota=400%, CPUWeight=20, IOWeight=20, one BLAS/OpenMP thread, and AGV_TRAINING_ENVS=4 (previously 12). Affinity enforces a four-logical-core limit even if CPU quota delegation is unavailable. No hard RAM cap is claimed: reducing worker count reduces memory consumption; monitor growth. CPU device leaves GPU unused by this PPO job.

Restarted from verified recovery_2 policy_10008.zip into recovery_3, retaining 120012 total checkpointed steps with 179988 requested steps remaining. Unsaved rollout samples were discarded. Optimizer/policy restored; RNG/rollout not exactly resumed. Worker reduction changes PPO rollout batch size, so evaluation remains required. Supervisor-only prior source hash explicitly authorized via AGV_PREVIOUS_RECOVERY_SHA256; reward, physics and inference source verification remain active. Three recovery tests passed. Resource limits persist on boot and also apply to child evaluation processes.
