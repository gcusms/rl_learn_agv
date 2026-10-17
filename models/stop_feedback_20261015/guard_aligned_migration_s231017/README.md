# 风险输入v2迁移候选

该模型用于把实际短时保护的接触风险纳入长期风险输入。**它已通过固定28项闭环保留，尚未证明导航整体改善，也没有批准替换实车模型。**

来源为 `normal_suite_wait_pair_v2_s231016/wait_continuation`，新增540次全批监督更新、新增PPO 0步。只更新现有恢复头；父导航actor、定位器、路由器及方向停靠组件冻结。3,799条成功来源标签全部拟合正确，其中557条停车标签全部保留；不包含失败前缀或新种子评估轨迹的正标签。

输入仍为625维，但439:481的三帧风险历史改变语义：同一帧短时保护已预测的接触风险作为旧长期风险的下界。原439维前缀与481:625后缀语义保持；必须使用manifest指定的独立v2运行时，不能给旧模型直接换入新输入。原硬件、12动作、动作掩码规则、记忆参数和4厘米碰撞余量保留。

严格运行时为 `neural_navigation.aligned_release_runtime.AlignedReleaseRuntime`。完整新bundle构造检查和导出动作一致性已通过。优化器、随机数状态及来源哈希保存在 [checkpoint.pt](checkpoint.pt)，统计在 [RESULTS.json](RESULTS.json)。训练耗时约25.9秒，CPU约9.1秒，峰值RSS约786 MiB；这些是本机训练阶段测量，不是Orin或树莓派性能。

闭环保留验证： [进度](../../../reports/stop_feedback_20261015/guard_forecast_migration_validation_20261017/STATUS.json)、[已产生结果](../../../reports/stop_feedback_20261015/guard_forecast_migration_validation_20261017/RESULTS.json)。验证最多28个重复使用的开发任务，不能当作陌生地图泛化。另有条件排队的混合状态PPO及独立物理验证，产物会另存目录，不覆盖本候选。

2026-10-17 03:54更新：28/28全部安全、0碰撞、0假到达、0物理保护。逐项与BC4500的物理轨迹和执行一致；917帧风险历史改变，5,028帧实际反馈核对通过。这是输入迁移保留，不是新增导航收益。见[完整核对](../../../reports/stop_feedback_20261015/guard_forecast_full_audit_20261017/README.md)。
