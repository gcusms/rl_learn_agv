# Orin Nano / 树莓派：2026-10-16 CPU 优化与板端验证

按用户确认，优先面向 **Jetson Orin Nano 8GB**，同时保留 **Raspberry Pi 5** 的 CPU 路线；树莓派内存容量尚未确定。本轮已优化动态障碍预测的重复计算，主机配对回放通过。尚无目标板测时，不能保证两块板均达到实时控制要求。

## 已测到的收益

主机：Intel Xeon E5-2696 v3，x86_64，Python 3.10.12、NumPy 1.26.4、PyTorch 2.8.0。推理单线程，实验平均预算约 0.35 核，不使用 GPU。3 个已保存条件各回放前 48 帧；两个独立运行时交替测量，共 144 个传感器帧、288 次调用。每条件前 8 帧预热后，每实现有 120 个计时样本。两组都安装实车原有的 `fast_guard` 地图检查加速，差别仅为方向风险计算实现。

| CPU 单步耗时 | 原实现 | 批量实现 | 减少 |
|---|---:|---:|---:|
| P50 | 77.74 ms | 54.49 ms | 29.9% |
| P95 | 172.00 ms | 102.22 ms | 40.6% |
| P99 | 187.59 ms | 110.28 ms | 41.2% |
| 最大值 | 190.46 ms | 113.74 ms | 40.3% |

证据：[配对原始结果](paired_host/RESULTS.json)。计时包含传感器核心、定位、路由、模型及保护检查；不包含 ROS 传输、扫描对齐、进程通信、其他节点和功耗。实验节流在步骤之间执行，单步耗时不能当作完整控制周期。地图是开发用仿真地图，并非实车 map_01 的板端测量；地图更大时路由成本也可能变化。

全部 144 帧的 **625 维观测、动作掩码、策略选择、执行命令及估计位姿**与优化前保存记录一致；配对的停车及动作否决原因一致。另有 [25 项功能检查](UNIT_CHECKS.json)通过，覆盖有限响应、各向异性误差、长轮廓分块、接触阈值、掩码和实际发布动作历史。有限回放与单元检查不能替代完整闭环回归或实车验证。

## 具体改了什么

热点在方向风险预测，而非已经证实需要压缩的网络。把“逐动作、逐预测时刻、逐障碍点”的 Python 循环改成 NumPy 批量计算，每 256 个轮廓点分块以限制临时内存；没有有效运动轨迹时跳过重复几何计算。接近原接触阈值或出现非有限中间值时回退原实现。

源码：`training/neural_navigation/directional_risk_batch.py`；原标量实现保存在 `directional_risk.py` 的 `directional_body_risk_reference`。观测语义、模型权重、12 个动作、矩形车体、4 cm 余量、预测时域和保护门槛均未调整。这一轮新增训练 **0 步**。

## 内存、功耗和能力边界

单模型剖析进程峰值约 **498 MiB**，优化前后基本相同；配对进程同时持有两个运行时，峰值 **508 MiB**。这不是整台机器人内存预算，不含 ROS2、SLAM、摄像头、桌面和系统缓存。之前含训练/物理仿真的约 2.13 GiB 也不能用来代表纯部署内存。本轮主要节省 CPU，尚未证明节省内存或降低整机功耗。

625 维候选的权重与上一交付包相同，SHA256 为 `a77684103768dbdaa5213246118e05b74fc108cc93f9ed25978dd48da2b0cfe3`。历史 12 项闭环为 11 到达、0 碰撞、1 超时，来自优化前；本轮没有重新运行闭环仿真。长时间等待、极窄脱困、未知地图定位及完整方向停靠回归仍未解决或完成，`deployment_approved=false`。不能把 CPU 加速描述为新增导航成功或定位精度改善。

## 在目标板复现

使用独立 `realtest_edge_20261016` 包。沿用目标板已能工作的 ROS/Python 环境及 numpy、torch、scipy、PyYAML、Pillow；Jetson 选择与本机 JetPack 匹配的 PyTorch。仅推理不需要 MuJoCo、PPO、训练并行环境或迁移桌面虚拟环境。

进入解压目录，先验证原文件，再重建目标架构定位库。包内原 `.so` 为 x86_64，不能直接用于 ARM64。需要 g++ 和 Eigen3 头文件；重建只生成新目录，不训练网络。

```bash
python3 verify.py
python3 tools/prepare_native_bundle.py \
  --source models/stop_feedback_20261015/release_context_bc2000_s231016/portable \
  --output generated/candidate_native
python3 configure.py --kind candidate --bundle generated/candidate_native
python3 configure.py --kind baseline
```

先在没有控制车辆的终端进行单线程回放。`board_bench_001` 必须是新目录；脚本既不会训练，也不会启动 ROS 节点或发送速度。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  nice -n 19 python3 -B tools/benchmark_embedded_runtime.py \
    --bundle generated/candidate_native \
    --output board_bench_001 --frames-per-case 64 --cpu-budget 1
```

这里 `--cpu-budget 1` 允许至多一个 CPU 核的平均实验预算，避免人为低配额掩盖板端速度；仍是单线程。电脑需要更保守时使用 `0.35`。工具记录本机版本、内核/模型散列、耗时分位数、峰值 RSS，并逐帧核对记录；若出现不一致将报错，不能继续引用主机的输出一致结论。可加 `--compare-reference` 在板端公平比较两种实现，但会同时持有两个模型。

启动计时只覆盖运行时对象构造，排除 Python/PyTorch 导入和 ROS 启动，不能用作完整冷启动时间。暂未针对完整冷启动优化。

可在另一个终端观察实际功耗模式、温度与内存：Jetson 使用 `nvpmodel -q`、`tegrastats`；树莓派在工具已安装时使用 `vcgencmd measure_temp` 和 `vcgencmd get_throttled`。本包不自动提高功耗、锁频或修改系统服务。

## 板端通过标准与下一步

以下是建议验收预算，不是已达到的结果：在原 0.2 s 控制周期下，单模型处理 **P99 ≤100 ms**，为传输和调度留出余量；进程峰值 RSS 暂以 **≤800 MiB** 作检查线，系统还要给 ROS、定位/建图和摄像头单独留预算。当前主机 P99 约 110 ms，尚未达到该建议线。

回放后在真实 ROS 链上记录扫描时间戳、推理起止、最终发布、保护原因、CPU/RSS 和温度，检查端到端 P95/P99、最大值及超时率。保持相同控制周期和雷达历史时间语义；降低雷达更新或放宽保护不能算加速通过。若某帧过期，沿用原超时停车流程。

优先在 Orin Nano 上做已有地图、相同路线的旧版/候选配对实测，再评估树莓派。若板端仍超预算，先按新剖析结果处理几何/地图检查的重复计算；仅在网络确为瓶颈时再评估 ONNX/量化/蒸馏，并重新检查动作边界与闭环结果。量化与低精度没有包含在本版。

本版为本地独立性能交付，未自动推送 GitHub；原 `realtest_20261016.tar.gz` 和原实车 2k 回退模型保留。完整实车配置、录制主题与回退步骤见包根目录 README。

## 平台参考

- [NVIDIA Jetson 模组规格](https://developer.nvidia.com/embedded/jetson-modules/)：具体功耗/内存选项随型号而异，TOPS 不能直接换算当前 CPU 导航链的延迟。
- [NVIDIA Orin 功耗与性能管理](https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)：实际性能受功耗模式、频率及温度影响。
- [Raspberry Pi 5 官方说明](https://www.raspberrypi.com/products/raspberry-pi-5/)：具体内存配置和散热条件需要一并记录。
