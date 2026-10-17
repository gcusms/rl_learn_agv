# AGV 激光导航

更新：2026-10-17 · 分支：`recovery-work`。包含最新模型、配套运行代码、训练记录和评估结果。

**训练已暂停；最新模型有局部改善，尚未通过完整部署验收。原实车模型保留。**

## 模型与结果

| 版本 | 模型位置 | 状态 |
|---|---|---|
| 最新联合约束候选 | [full_navigation_joint_linear_s231017](models/stop_feedback_20261015/full_navigation_joint_linear_s231017/) | 625维观测、12个动作；用于下一轮受控对照 |
| 原实车质量恢复2k | [quality_recovery_20261014](models/quality_recovery_20261014/) | 300维观测；保留回退，历史48项为46到达、1碰撞、1超时 |

同一37项开发/恢复检查的对比：

| 版本 | 安全到达 | 碰撞 | 超时 |
|---|---:|---:|---:|
| 此前625维等待修复版 | 33 | 1 | 3 |
| 最新联合约束候选 | **35** | **0** | **2** |

这些检查包含训练参考和相关恢复片段，不代表独立泛化成功率；原实车版48项来自另一套任务，不能直接排名。

- **改善**：动态00由超时变为61.6秒到达；退化雷达任务01由碰撞变为37秒到达。相对直接上一候选，动态02由超时恢复为49.2秒到达。
- **路径**：普通18项仍为459.4秒、67.136米；没有普遍缩短路径，也没有最短路径保证。
- **定位与停车**：定位器、路由器和方向停靠部分冻结；没有本轮新增定位精度结论，固定2项方向停靠检查保留。
- **剩余问题**：97.6秒、106秒晚期恢复仍超时；新8组配对和Orin Nano/树莓派实测尚未完成。
- **训练性质**：继承1993步PPO；之后为辅助拟合和约束更新，最后一步新增PPO为0。

详见[六版模型对比](reports/model_comparison_20261017/README.md)、[正式35项](reports/stop_feedback_20261015/full_navigation_joint_linear_validation_20261017/RESULTS.json)、[补验2项](reports/stop_feedback_20261015/deployment_acceptance_supplement_20261017/README.md)。

## 获取与验证

```bash
git clone --depth 1 --branch recovery-work https://github.com/gcusms/rl_learn_agv.git
cd rl_learn_agv
python3 tools/check_github_snapshot.py
```

校验仅使用Python标准库，检查本分支文件与模型哈希，不启动训练或车辆。依赖见[training/requirements.txt](training/requirements.txt)；建议Python 3.10虚拟环境。Jetson需安装与JetPack匹配的PyTorch。

定位原生内核需在新电脑重建，`*.so`不随Git迁移。安装`g++`、`libeigen3-dev`后执行：

```bash
python3 tools/prepare_native_bundle.py \
  --source models/stop_feedback_20261015/full_navigation_joint_linear_s231017/portable \
  --output generated/native_candidate
```

模型、特征合同和运行时必须配套。最新625维候选使用 `neural_navigation.aligned_release_runtime.AlignedReleaseRuntime`；不能只替换旧模型的actor。实车接入见[ROS2适配说明](deployment/ros2_recording_adapter/README.md)，先使用shadow模式核对传感器、时间与动作记录。

完整证据见[模型对比数据](reports/model_comparison_20261017/COMPARISON.json)、[当前快照](docs/releases/20261017/SNAPSHOT.json)；较早过程见[历史README](docs/handoff/README_history_20261017.md)。
