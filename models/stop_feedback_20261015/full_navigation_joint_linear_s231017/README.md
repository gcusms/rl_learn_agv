# 联合约束导航候选

已保留sensor01安全到达37秒，并恢复动态任务02安全到达49.2秒、可运动却停车569→1帧。两项均为已知训练/开发任务，完整原验收未完成，不替换部署模型。

截至2026-10-17 08:28，当前权重已完成原固定28项及6项已知完整任务，34回合全部安全到达。独立核对5,028＋1,326帧，原安全与效率门槛通过；这是已知能力保留，不是陌生地图泛化。原106秒困难状态正在验证，97.6秒、96秒保留和条件冻结8组仍须完成。见[固定28项](../../../reports/stop_feedback_20261015/full_navigation_joint_linear_20261017/FIXED28_AUDIT.json)、[6项完整任务](../../../reports/stop_feedback_20261015/full_navigation_joint_linear_20261017/KNOWN6_PHYSICAL_AUDIT.json)。

5,608条监督参考、692条旧停车标签全部保留；冻结父模型与特征层，最后一层最小L1约束拟合1次更新，来源2,500次辅助监督，新增PPO0步。原625维输入、12动作、车体4厘米余量、未知/盲区、记忆、定位/路由与方向停靠保留。

[方法与验证](../../../reports/stop_feedback_20261015/full_navigation_joint_linear_20261017/README.md) · [完整任务审计](../../../reports/stop_feedback_20261015/full_navigation_joint_linear_20261017/PRIORITY_PHYSICAL_AUDIT.json) · [实时验收](../../../reports/stop_feedback_20261015/full_navigation_joint_linear_validation_20261017/STATUS.json)。
