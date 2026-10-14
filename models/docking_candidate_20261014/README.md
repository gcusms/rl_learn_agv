# 10k 终点 PPO 候选

独立候选，未通过部署验收。结果和边界见 [本轮报告](../../reports/docking_training_20261014/REPORT.md)。

`portable` 与 `portable_v4_localization` 使用同一 PPO 演员、不同定位合同；均需 `experiments.docking_navigation.DockingRuntime`、303 维观测、12 动作。`policy.zip` 含终点演员、冻结导航演员、critic 与 Adam；`LINEAGE.json` 记录保留链。训练源码封存在 source_snapshot；预热数据不是实车示范。
