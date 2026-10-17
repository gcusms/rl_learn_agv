# 3,000 次候选：本次验证已结束，效率门槛未通过

主服务已正常结束，共 21 次物理回合，476 项固定输入哈希核验通过。候选 `sensor_retention_pair_s231016/retained500`，本阶段 BC 3,000 次更新、PPO 0 步，部署未批准。

| 组 | 完成 | 安全到达 |
|---|---:|---:|
| sensor5 | 5 | 5 |
| normal_repaired2 | 2 | 2 |
| source_recovery3 | 3 | 3 |
| heading2 | 2 | 2 |
| fresh_noise_reference2 | 2 | 1 |
| fresh_noise_candidate2 | 2 | 2 |
| normal_other16 | 5 | 5 |

其中 fresh_noise_reference2 是旧参考，不能计入候选成功数。normal_repaired2 与 normal_other16 共测 7 项完整常规任务，仍有 11 项未测。

停止原因为 `train_failure_w11_e7`：旧参考 37 秒/4.616 m/等待 1 帧，候选 58.6 秒/4.751 m/等待 104 帧；耗时增加 58.4%，超过单项 +20% 上限。三个晚恢复和剩余常规任务未执行，不能称完整验收通过。已测候选无碰撞、假到达或物理保护干预。

已知噪声 05_0 恢复至 33.8 秒/31 帧等待；新噪声 0 从旧参考超时改善为 35 秒到达；新噪声 1 则比旧参考慢约 5.5%、路程长约 5%。专项改善与退化均保留，不据此替换实车模型。

实际平均 CPU 0.35 核，峰值 RSS 2,429.4 MiB；内存硬上限 3 GiB，CPU 为软件平均预算。定位器、路由器和方向停靠冻结，原实车模型及交付包保留。

[最终结果](RESULTS.json) · [固定协议](PLAN.json) · [后续首次分叉](../full_wait_regression_replay_20261016/README.md) · [最新对比](../training_comparison_20261016/README.md)
