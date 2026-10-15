# 定位算法调研与下一轮优化方案 · 2026-10-15

调研基于代码 `a17929bfe9e86c3872536f66efa80214dffa2b46`、本地历史实验和实车录制 `20261014_193644_map_01test_953a9959`。本文件是算法选择、输入审计与实验设计；没有把文献结论记为本项目已实现的精度提升。

建议继续研究“连续里程计预测＋静态结构匹配＋按方向处理退化＋经过验证的回环”这条路线。已知地图先用 AMCL 做独立定位对照；边建图先定位现有 SLAM Toolbox 的误差来源，再决定是否比较 Cartographer 2D。定位网络优先学习观测可靠性与运动误差，随后再微调导航策略。

**这次核对得到的新证据**

新增只读工具：[`audit_recorded_scan_contract.py`](../../../training/experiments/audit_recorded_scan_contract.py)。完整检查实车包中的 **2,041 帧 `/scan`**，没有启动 ROS 节点或发布控制指令。结果见 [`SCAN_CONTRACT_AUDIT.json`](SCAN_CONTRACT_AUDIT.json)。

| 项目 | 录制实际值 | 对优化的影响 |
|---|---|---|
| 原生扫描 | 每帧 400 束，消息角度约 0～359.1° | 不能把模型重采样的 220 束称为原生雷达束数 |
| 模型输入 | 220 束，约 219.4°扇区；归一化上限 12 m | 扇区、重采样与模型合同仍需保持一致 |
| 实车量程 | 0.1～10 m | 评估与训练必须体现此量程，不能生成 10 m 外的定位线索 |
| 原生扫描中模型扇区内的缺失 | 按原生束计，NaN 约 **5.97%** | 这是全包平均；不是模型重采样后缺失率，也不代表连续缺失分布 |
| 模型扇区外回波 | 有限有效回波约 **17.37%**，NaN 约 82.63% | 不能当作可靠完整后方视野；可以另做定位输入覆盖范围对照 |
| 相邻扫描时间戳间隔 | 中位 **99.63 ms** | 保留真实时间，不只按历史帧数拼接 |
| 一帧首末束跨度 | `399 × time_increment` ≈ **105.00 ms** | 原始扫描并不是同一时刻采集 |
| 接收时间减 header | 中位 **0.452 ms** | 需核对 driver 的 header 究竟表示首束、末束还是发布时刻 |
| 若直接假设 header 是首束 | 末束落在接收之后，中位 **104.55 ms**；2,041 帧全部如此 | 这是时间语义疑点，不能直接把 header 加 105 ms 后就认为去畸变正确 |

ROS 的 LaserScan 合同规定 header 为首束采集时间，`time_increment` 为束间隔。上面的记录说明应先检查部署驱动、扫描排序和时钟；接收时间本身并不是独立采集真值，尚不能断言具体是哪项错误。[官方消息定义](https://raw.githubusercontent.com/ros2/common_interfaces/humble/sensor_msgs/msg/LaserScan.msg)

例如以 0.5 rad/s 转弯，105 ms 内航向变化约 3°；5 m 处的端点几何变化可达约 26 cm 量级。这只是运动学量级估算，不是本次实测定位误差。

仓库已有 [`NativeScanAdapter`](../../../training/neural_navigation/scan_alignment.py)，按首束时间补偿到末束；而录制的 ROS 包装器和当前 `deployment/ros2_recording_adapter/model_node.py` 仍按 header 对齐里程计并直接重采样。**已有可选工具不等于实车链路已启用或已验证。** 应先做时间合同对照，再决定怎样接入，避免补偿反而引入误差。

[`REAL_SENSOR_INPUT_AUDIT.json`](../REAL_SENSOR_INPUT_AUDIT.json) 还确认：实车 `/odom` 已融合指令来源的平移速度和 IMU 角速度；已有有效轮反馈，但未选为底盘里程计来源。这里不需要先购买编码器，也不能把“指令速度积分”解释为轮子实际走过的路。轮反馈的尺度、方向和滑移仍需独立验证。

**可借鉴的算法及适用范围**

| 方法与一手资料 | 值得学习的逻辑 | 在本项目中的用途与限制 |
|---|---|---|
| [AMCL 官方文档](https://docs.nav2.org/rolling/configuration_and_development/configuration_guide/others/configuring_amcl/)；[本机对应 1.1.19 源码](https://github.com/ros-navigation/navigation2/blob/1.1.19/nav2_amcl/src/sensors/laser/likelihood_field_model_prob.cpp) | 里程计传播多个位姿假设，再用静态地图更新；`likelihood_field_prob` 可跳过大多数粒子不支持的回波 | 优先作为已有 map_01 的独立基线，针对新增未入图障碍物；不承担未知环境建图 |
| Zhang、Kaess、Singh，ICRA 2016，[On Degeneracy of Optimization-based State Estimation Problems](https://www.cs.cmu.edu/~kaess/pub/Zhang16icra.pdf) | 分析加权几何约束的特征值和方向，只在可约束方向接受匹配更新，弱方向保留预测 | 最贴近长墙、走廊漂移；论文实验为视觉与三维激光组合，需把原理适配到本车平面三个自由度，不能照搬实验精度 |
| [Cartographer 算法说明](https://google-cartographer-ros.readthedocs.io/en/latest/algo_walkthrough.html)；[二维 Ceres 匹配源码](https://github.com/cartographer-project/cartographer/blob/master/cartographer/mapping/internal/2d/scan_matching/ceres_scan_matcher_2d.cc) | 运动预测为局部子图匹配提供初值与约束；后台通过经过筛选的约束连接子图 | 适合借鉴局部与全局分工，可作为建图备选对照；不是换上就能解决无几何信息的问题 |
| [SLAM Toolbox 官方项目](https://github.com/SteveMacenski/slam_toolbox)；[本机对应 2.6.10 Karto 源码](https://github.com/SteveMacenski/slam_toolbox/blob/2.6.10/lib/karto_sdk/src/Mapper.cpp) | 保存位姿图与观测；通过滚动扫描和图约束支持定位、续建与回环 | 项目已接入，优先诊断它；PGM＋YAML 不等于完整位姿图，图定位需要兼容的序列化图数据 |
| [RF2O 作者开源实现](https://github.com/MAPIRlab/rf2o_laser_odometry) | 从相邻二维扫描的距离变化估计运动，用鲁棒误差降低异常约束影响 | 可作为局部激光里程计对照；本身不是绝对地图定位或完整回环 SLAM，无特征方向仍可能退化 |
| Jonschkowski 等，RSS 2018，[Differentiable Particle Filters](https://roboticsproceedings.org/rss14/p01.pdf)；[作者代码](https://github.com/tu-rbo/differentiable-particle-filters) | 保留递归预测、测量更新和多假设结构，学习运动与观测模型 | 适合后续学习定位可靠性和重复结构中的多解；原实验包含视觉任务，不是能直接替换本项目的二维雷达预训练模型 |

AMCL 的跳束并非无条件过滤：对应版本源码只在滤波收敛后启用；若过多束不支持地图，还会退回完整观测更新，防止把错误定位锁死。这个保护思路值得保留。官方默认 `laser_model_type=likelihood_field`、`do_beamskip=false`，只打开布尔参数而不核对模型类型并不能完成试验。上述参数需要针对当前雷达验证，不照抄默认距离阈值。

Cartographer 官方仓库明确说明已不再积极维护，ROS 分支维护也有限。因此推荐学习其预测、子图与约束设计，先完成受控对照再考虑迁移；当前不把整套迁移作为第一项任务。[维护说明](https://github.com/cartographer-project/cartographer)

**现有代码已经有的能力与缺口**

[`odom_anchored_localizer.py`](../../../training/neural_navigation/odom_anchored_localizer.py) 已有静态端点筛选、Huber 精修、几何方向投影和连续里程计预测。下一轮不能把相同功能再包装为新算法。

| 现有处理 | 尚未解决的问题 | 建议改进 |
|---|---|---|
| 在预测位姿下检查扫描端点是否靠近静态墙体 | 初始位姿较错时可能把真实静态回波一起拒绝；新物体靠近墙也可能被接纳 | 区分“位姿已可信时的异常回波筛选”与“重新定位时的多假设搜索”；结合预计射线距离、多个角区和连续帧支持 |
| 根据距离场法向构造方向投影 | 阈值、航向尺度、观测噪声尚为经验值；局部几何良好也不能排除重复柱列中的错误匹配 | 以噪声加权约束、分方向误差和独立验证集校准；另保留不同位置假设及时间一致性 |
| 保存三个标准差，接收观测后经验收缩 | 不等于经过验证的完整协方差；相关性、偏置与多峰分布表达不足 | 传播位置与航向相关性；检验误差覆盖率、错误高置信度、创新一致性，不能只把标准差压小 |
| 在线建图接收 SLAM 位姿和协方差 | 新鲜位姿也可能错误；输出协方差并不自动证明可靠 | 记录前端增量、图优化前后位姿、约束种类和几何退化，再决定是否接纳；位姿拒绝还要检查内部地图是否已被错误构建 |

沿一面长墙，通常能约束到墙的距离和航向，却难以约束沿墙位移。应该按弱方向降低雷达影响；不应让整个雷达定位一律失效，也不应一律增大里程计权重。完全没有可辨识结构时，任何上述方法都无法仅靠训练创造绝对位置约束。

麦轮不发横移指令不代表实际横向速度严格为零。滤波应保留侧滑及其不确定性；不能照搬差速底盘的高置信度 `vy=0` 约束。使用已经含有 IMU 的 `/odom` 作先验时，也不能再把同源 IMU 当成独立重复证据，造成虚假的高置信度。[robot_localization 融合说明](https://raw.githubusercontent.com/cra-ros-pkg/robot_localization/rolling-devel/doc/configuring_robot_localization.rst)

**为什么暂不继续只调“更相信里程计”**

已完成的长距离未知地图对照采用相同物理轨迹、雷达退化和输入近似：12.06 m、92.8 s、指令平移＋带噪模拟 IMU，另加 8% 平移比例误差、0.003 rad/s 陀螺偏置。这些是实验假设，不是精确复现真实 EKF。

| 场景 | 单独里程计位置 P95 | 当前 SLAM P95 | 减小 `distance_variance_penalty` 后 P95 |
|---|---:|---:|---:|
| 单面长墙 | 0.806 m | 8.489 m | 10.336 m |
| 从有特征区进入空旷区 | 0.806 m | 1.336 m | 2.998 m |

数据分别来自 [`long_online_slam_2`](../long_online_slam_2/RESULTS.json) 和 [`long_online_slam_distance_prior_2`](../long_online_slam_distance_prior_2/RESULTS.json)。这是人工给定轨迹的传感器专项，不是自主导航成绩、真实精度或多数场景统计。坏结果已经保留，参数改动未采纳。

本机安装 SLAM Toolbox 2.6.10。核对同标签源码可见：

- 里程计距离项是扫描相关性得分的乘法惩罚，并设有最低惩罚值，不是硬性运动约束。
- `LinkNearChains()` 调用匹配时传入 `doPenalize=false`；显式回环匹配也关闭该项惩罚。
- 因而只调整顺序匹配的这一参数，无法约束所有扫描链和图链接。**这解释了为何它不足以保证稳定；尚未证明某一类链接就是当前大漂移的根因。**

来源：[2.6.10 Mapper.cpp](https://github.com/SteveMacenski/slam_toolbox/blob/2.6.10/lib/karto_sdk/src/Mapper.cpp)。后续需记录具体约束和位姿变化，再做因果对照，不能凭源码直接认定“都是回环出错”。

**建议按这个顺序推进**

| 顺序 | 工作 | 是否训练 | 验证与下一步条件 |
|---|---|---|---|
| 1 | 核对雷达 header、束时间和扫描排序；比较未补偿与经过验证的补偿；分清有效命中、无回波、无效束及断流 | 暂不训练 actor | 静止不引入偏差；左右转时匹配改善；无未来时间查询或跨回合历史混合 |
| 2 | 已知地图：当前 v5/v6 与 AMCL 的普通/跳束模式做同输入回放 | 暂不训练 actor | 对新增障碍、转弯、人工重定位、静止漂移分别报告；原包没有独立绝对真值，不宣称厘米级准确 |
| 3 | 边建图：固定长墙、特征到空旷两条轨迹，隔离顺序匹配、邻近链链接和显式回环 | 暂不训练 actor | 找到大漂移首次发生的环节；同时检查地图一致性与分方向误差，不能只在输出端遮住坏位姿 |
| 4 | 在已验证估计框架中训练静态支持/异常权重、增量误差模型或不确定性；必要时增加少量位姿假设 | 定位专项监督/时序训练 | 使用仿真独立真值或独立实测参考；不能把当前 SLAM 输出直接当自身精度标签 |
| 5 | 把改进定位接入闭环，检查已有避障、正常路线、新地图和方向停车 | 先回归，再决定 PPO 微调 | 同时改善弱特征失败且正常能力不明显退化；局部定位变化仍可能改变原 actor 输入 |

第 3 项至少保留：现有配置、关闭显式回环、关闭邻近链但保留显式回环、仅顺序匹配，以及单独里程计对照。关闭显式回环并不等于关闭所有邻近链链接；后者需在隔离构建中增加诊断开关，不修改系统或实车安装。每次保持扫描、里程计、地图初值和种子一致；先确认同配置重复运行的波动。

必要时才加入 Cartographer 2D 对照，使用相同可用传感器、时间合同和退化数据。纯定位需要与其兼容的 `.pbstream`；不能把现有 PGM 直接当成等价的冻结子图状态。[纯定位说明](https://google-cartographer-ros.readthedocs.io/en/latest/tuning.html)

**模型具体学习什么**

定位训练建议学习三个量：哪些雷达回波得到可靠静态支持；当前几何能约束哪些方向；当前运动与匹配误差的合理范围。训练片段覆盖左右转、静止、短时受阻、侧滑、连续缺失、未入图物体、重复柱列、长墙和返回有特征区。保留正常任务，按原始地图和采集路线划分数据，避免同一轨迹泄漏。

多假设定位用于不同位置都能产生相似扫描的场景；不要把两个相距较远的高概率位置求平均后当作可信位姿。借鉴可微粒子滤波的递归结构不要求使用旧项目的视觉模型或依赖。第一版先与普通粒子滤波比较必要性和耗时，不直接重写全部模型。

策略仍由强化学习输出速度和转向；定位模块提供位姿与可信度。定位专项监督学习和之后的 PPO 控制训练职责不同。当前恢复分支训练冻结定位网络，继续增加相同 PPO 步数无法直接更新定位器。

连续局部运动与全局地图纠偏要分别维护：`odom→base` 保持连续，地图定位或回环更新 `map→odom`；纠偏后按坐标变换同步路线和障碍记忆，防止把地图坐标变化误判成物体运动。这是可脱离 ROS 实现的数学分工，不要求接管为 Nav2 行为树。[REP-105](https://raw.githubusercontent.com/ros-infrastructure/rep/master/rep-0105.rst)

**验收与资源范围**

先报告绝对/相对位姿误差、沿墙与垂墙误差、航向误差、错误高置信度、重定位时间，以及地图与路线是否完整；随后才看闭环到达、碰撞、假到达、实际用时和方向停车。原 12 项历史避障、48 项方向与导航验证、8 项新图挑战都要保留，再扩展固定任务与多种子。没有独立实车参考的回放只检验一致性与异常纠正，不能充当绝对精度验收。

目标精度要由通行余量确定。车宽约 0.329 m；通道仅比车宽 10～15 cm 时，单侧空间只有 5～7.5 cm，还需分配给安全余量、横向定位、地图边界和航向误差。不能拿“20 cm 到达容差”证明窄道定位足够。粗规划图连通性与约 0.30 m 规划膨胀是独立限制，也不能只靠定位训练修复。

保持单线程、小批次、实验顺序执行，避免重新占满电脑；对 SLAM 子进程也计入资源预算。本机已有 AMCL、Cartographer 与 SLAM Toolbox，可以先做对照，无需先下载大模型。Jetson Orin Nano 的端到端 P95/P99 和常驻内存仍需目标机实测；论文或本机平均耗时不替代该验收。

复现本次新增输入审计：在仓库根目录用 Bash 执行（只读 bag，写入报告文件）：

```bash
source /opt/ros/humble/setup.bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="training:$PYTHONPATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
nice -n 19 /usr/bin/python3 -m experiments.audit_recorded_scan_contract \
  --recording /home/ubuntu/rl_learn/record_analysis_20261014/raw/20261014_193644_map_01test_953a9959 \
  --hardware models/quality_recovery_20261014/portable/hardware.json \
  --limit 10000 \
  --output reports/localization_mapping_20261015/localization_research/SCAN_CONTRACT_AUDIT.json
```
