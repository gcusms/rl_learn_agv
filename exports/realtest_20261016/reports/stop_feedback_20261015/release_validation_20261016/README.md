# 2026-10-16 收尾候选：12 项闭环结果

完成核对时间：2026-10-16T15:34:28.342977+08:00。按用户要求结束本轮优化，等待受控实车验证。原模型未替换；当前无训练或物理评估运行。

本候选 625 维上下文分支，1790 条完整安全来源、实际 50 次监督更新、新增 PPO 0 步。localizer、router、方向停靠与基础 actor 冻结。

**12 条件：11 到达、0 碰撞、1 超时；新噪声门槛未通过。** 旧失败条件改善，但新噪声条件仍有持续等待。不能写成全部修复或发布验收通过。

| 条件 | 分组 | 到达 | 碰撞 | 超时 | 时间/秒 | 路程/米 | 有运动选项时等待/帧 |
|---|---|---:|---:|---:|---:|---:|---:|
| early_yield_sensor_holdout_05_0 | repaired_wait_condition | 1 | 0 | 0 | 36.8 | 4.948 | 40 |
| proactive_fresh_condition_01 | retained_development | 1 | 0 | 0 | 36.6 | 4.865 | 46 |
| proactive_fresh_condition_02 | retained_development | 1 | 0 | 0 | 37.2 | 5.112 | 55 |
| proactive_fresh_condition_04 | retained_development | 1 | 0 | 0 | 35.0 | 5.290 | 22 |
| proactive_fresh_condition_05 | retained_development | 1 | 0 | 0 | 35.6 | 4.815 | 48 |
| early_yield_sensor_holdout_02_0 | retained_development | 1 | 0 | 0 | 33.4 | 4.935 | 38 |
| early_yield_sensor_holdout_02_1 | retained_development | 1 | 0 | 0 | 34.8 | 5.099 | 31 |
| early_yield_sensor_holdout_05_1 | retained_development | 1 | 0 | 0 | 34.0 | 4.933 | 32 |
| long_risk_sensor_02_0 | frozen_new_noise | 1 | 0 | 0 | 33.8 | 5.026 | 33 |
| long_risk_sensor_02_1 | frozen_new_noise | 1 | 0 | 0 | 35.6 | 5.149 | 29 |
| long_risk_sensor_05_0 | frozen_new_noise | 1 | 0 | 0 | 37.6 | 5.119 | 44 |
| long_risk_sensor_05_1 | frozen_new_noise | 0 | 0 | 1 | 140.0 | 2.199 | 636 |

旧 8c3 父版在相同旧 8 条件为 7/8，本候选为 8/8；其中 7 个保留条件沿用拟合来源，不能称为独立泛化。新增 4 个冻结噪声条件为 3/4，仍属于开发地图。原实车 2k 版历史 46/48 来自不同集合，不能与本轮 11/12 直接排名。

旧失败 early_yield_sensor_holdout_05_0 从父版 140 秒超时/620 帧等待改善到 36.8 秒到达/40 帧等待；新的 long_risk_sensor_05_1 在 140 秒超时，636 帧有合法运动选项却等待。必要停车、原保护规则与 4 cm 余量保留，未强迫前进。

最大单步主机推理约 308.9 ms；这不是 Orin 测试。ROS 250 ms 截止机制可能导致额外停车，需在实车记录中单独分析。定位误差来自仿真评分，不是实车测量；本轮未更新定位器。

完整常规/动态、旧恢复前缀、95/90/106 秒条件、朝向与额外等待回归未对本候选执行。排队的 broad 和 decision audit 在开始工作前按用户收尾意向取消，相关目录有 CANCELLATION.json，不能记作检查通过。极窄位置脱困仍未解决。

模型：models/stop_feedback_20261015/release_context_bc2000_s231016/checkpoint.pt

SHA256：a77684103768dbdaa5213246118e05b74fc108cc93f9ed25978dd48da2b0cfe3。目录 bc2000 是最大更新上限，实际更新为 50。

交付目录：exports/realtest_20261016；独立归档：/home/ubuntu/rl_learn/realtest_20261016.tar.gz。实车部署、记录与回退见交付 README。

全部物理输入/输出/扫描与失败轨迹保留在 physical/；没有挑选成功样本替换失败记录。GitHub handoff3 仍固定于旧截止时间，本轮新研究未自动推送。
