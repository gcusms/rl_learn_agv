# 2026-10-17 更新前的 README 历史记录

本文件保留旧进度说明；当前状态以[根 README](../../README.md)和最新评估为准。

# AGV 强化学习导航：停车反馈与恢复优化

<!-- bounded-acceptance-20261017 -->
**2026-10-17 10:13（北京时间）：两项部署前补验与独立核对已完成，仍未通过完整部署验收。** 原96秒成功保留：132.4秒安全到达、10.394米；原97.6秒仍在140秒超时，原106秒失败也保留。两项补验零碰撞，补验及核对约9.3分钟；加原35项共37次混合记录检查、35安全到达、2超时，不能作为独立泛化成功率。模型权重和原实车部署不变，训练/优化仍暂停，无活动RL服务或定时器。[补验结果与原始证据](../../reports/stop_feedback_20261015/deployment_acceptance_supplement_20261017/README.md)。
<!-- /bounded-acceptance-20261017 -->


**2026-10-17 09:55（北京时间）：按用户要求暂停本轮优化，后台仿真与自动重放等待已停止，无活动RL服务或定时器。** 最新研究候选、训练与验证记录均保留，原实车部署模型未替换。正式结果仍为35次仿真34安全到达、1超时、零碰撞；原固定28项和6项已知任务通过，但困难停车恢复未解决，原97.6/96检查及新8组尚未完成，不能视为完整验收通过。

[收尾说明与停止核对](../../reports/stop_feedback_20261015/user_pause_20261017/README.md)。以下为各自截止时间的历史记录，其中“进行中”不表示当前仍在运行。


**2026-10-17 08:50（北京时间）核对：本轮正式验证35个回合，34次安全到达、1次超时、0碰撞；有局部改善，整体部署验收未通过。** 相比直接上一候选，动态避障任务02由140秒超时、569帧可运动却停止，改善为49.2秒安全到达、1帧主动等待；sensor01仍保持37秒安全到达。原固定28项及6项已知完整任务均通过独立核对，常规18项总耗时459.4秒、路程67.136米与直接上一版持平。

失败为原106秒困难恢复：接管后170帧有合法运动但全部停止，34秒后按原140秒时限超时。当前低内存恢复示范诊断仍在进行，不算模型成功或新增PPO训练。原97.6/96及条件冻结8组尚未完成。定位、路由及方向停靠冻结，本轮没有新增定位精度或Orin/树莓派实测结论；原部署模型保留。

本轮保留5,608条监督参考、692条旧停车标签，来源2,500次辅助更新后增加1次受约束末层更新，新增PPO采样0步。当前原因输入会影响概率及部分动作，但困难状态尚未有效恢复。

后续诊断更新至2026-10-17 09:37：新空闲证据记忆从初始化准入通过，但两种完整老师试探均在原140秒超时，未采用该规则或生成正向标签。258项几何核对一致，后期部分状态连当前扫描本身也不允许运动。现已在原模型/记忆/保护下启动[逐帧恢复搜索](../../reports/stop_feedback_20261015/late106_framewise_protocol_20261017/README.md)，覆盖短转向窗口，最多保留5个方向候选；一个主仿真进程、0.35核软件预算、3GiB硬内存。启动不代表恢复成功，模型权重和35回合结果未改变。

[本轮最终同条件比较](../../reports/stop_feedback_20261015/training_comparison_20261016/COMPARISON_20261017_JOINT_LINEAR_TERMINAL.md) · [正式验证结果](../../reports/stop_feedback_20261015/full_navigation_joint_linear_validation_20261017/RESULTS.json) · [失败原因核对](../../reports/stop_feedback_20261015/full_navigation_joint_linear_20261017/CURRENT106_FAILURE_AUDIT.json)。

以下保留较早时间截面。

**2026-10-17 06:16（北京时间）最终核对：有局部优化，整体安全验收未通过。** 最新候选保留固定28项安全及效率门槛；动态00保持61.6秒到达，动态01相对直接上一轮由53.4秒缩短为34.6秒、路程减少5.2%，但两例已用于训练开发。完整验收已结束：34个实际物理回合中33次安全到达、1次碰撞，失败为动态障碍与雷达退化组合任务；困难3项及预留新种子8组未执行。定位、路由、方向停靠冻结，没有本轮定位精度提升证据。模型另存，原部署保留。

[最终逐项比较与证据](../../reports/stop_feedback_20261015/training_comparison_20261016/COMPARISON_20261017_TERMINAL.md)。本次追加119次辅助监督更新，新增PPO0步；较早PPO1993采样另计。

以下为较早时间截面。

**2026-10-17 05:07（北京时间）最终结果：有局部改善，整体升级未通过，评估已结束。** v2 PPO完成1,993次有效采样；固定28项回归全部安全到达，常规18项用时459.4→460.0秒、路程67.136→67.111米，基本持平。动态00由140秒超时改善为61.6秒到达；动态01也到达，但相对稳定候选BC4500耗时34.6→53.4秒（+54.3%），超过原单项20%上限，扩展测试因此停止。共30个物理回合，余下已知4项、困难3项及新种子8组未执行，106秒问题未豁免。定位器/路由器冻结，暂无本轮定位改善证据。原部署模型保留，当前两项服务均已结束。

以下保留较早时间截面。

**2026-10-17 04:46（北京时间）：v2 PPO完成，独立验收进行中。** 新模型追加1,993次有效策略采样、106/128次actor/critic更新，另存为`models/stop_feedback_20261015/guard_aligned_mixed_ppo_resume_s231017`。固定回归已完成19/28项，全部安全；常规18项全部到达，耗时459.4→460.0秒、路程67.136→67.111米，原效率门槛通过，能力基本保留。其余仍在运行，尚无整体升级结论。

完整1,993次采样已独立核对：当前拒绝原因、实际动作历史、风险时间尺度及整批优势归一化一致；445次可选运动却停车全部进入可训练范围，408次得到负优势。634次只有停车可选不分配actor梯度。训练中的减少等待尚未解决97.6/106秒困难状态，不能代替自主到达。557条参考停车标签保留；定位、路由、方向停靠冻结。

首批后RSS退出的完整状态已恢复，第二批在新进程完成；未完整保存的失败片段不作PPO样本。仅一个主物理任务，平均0.35核软预算、Nice19、3 GiB硬内存。原最多53回合验收、安全/效率及106秒恢复要求保留。原实车模型保留，目标尚未完成。[续训与验收](../../reports/stop_feedback_20261015/guard_aligned_resume_protocol_20261017/README.md)、[学习信号与旧碰撞核对](../../reports/stop_feedback_20261015/guard_aligned_credit_audit_20261017/README.md)、[当前证据](../../reports/stop_feedback_20261015/native625_learning_diagnostic_20261017/CURRENT_RESULTS.md)。以下保留历史截面。

**2026-10-16 22:20 最终验证：局部修复有效，整体升级未通过。** 最新 3,500 次监督候选完成 19 个开发/训练回归物理回合，均安全到达，零碰撞、零假到达；这些回合包含初始化恢复，不能合称陌生地图导航成功率。e7 从上一 3k 候选的 58.6 秒、104 帧主动等待恢复为 37.0 秒、1 帧；但另一项动态任务相对较早成功参考由 51.4 秒变为 79.2 秒（+54.1%），主动等待 0→194 帧，超过预定单项耗时 +20% 门槛，验证因此结束。完整常规只测 9/18 项，三个晚恢复未执行。

本阶段新增 PPO 仍为 0；定位器、路由器冻结，原实车模型保留。下一轮原生 625 维 PPO 的 8 项核心检查与 4 项采集衔接检查通过，但因本候选正常导航保留未通过，训练服务未启动。当前没有该轮后台训练或物理验证。详见[最终对比](../../reports/stop_feedback_20261015/training_comparison_20261016/README.md)、[最终闭环](../../reports/stop_feedback_20261015/full_mission_retention_validation_20261016/README.md)。以下为历史时间截面。

**2026-10-16 21:55 更新：新 3,500 次监督候选已修复一项完整导航退化，较广回归继续运行。** e7 从上一 3k 候选的 58.6 秒、104 帧主动等待恢复为 **37 秒、1 帧等待**，路程 4.751→4.616 m；两项关键噪声分别保持 33.8/34.0 秒，均安全到达。这些是训练/开发回归，不能称完整验收或陌生地图泛化。新候选保留全部 4,949 条标签，配对原数据对照仍有 23 条差异；本阶段累计 BC 3,500 次更新，新增 PPO 0 步。

旧 106 秒记录输入仍有 169/170 帧合法运动却等待，晚恢复未解决。此前三种原许可转向均无碰撞但未到达，多数后缀只剩停车合法；未使用失败轨迹作为成功标签。当前仅一个物理主进程，软件平均 CPU 预算 0.35 核、3 GiB 内存硬限制。本机 CPU 硬配额未生效。原实车模型与封存包哈希未变，定位器/路由器未更新，无本轮定位精度或目标板性能提升结论。

上一 3k 候选的 21 回合验证因 e7 单任务耗时 +58.4% 结束，完整常规只测 7/18 项，未替换实车模型。当前新候选优先重测该任务后继续原传感器、常规、恢复与方向停靠检查，最多 31 回合；结果见[最新对比](../../reports/stop_feedback_20261015/training_comparison_20261016/README.md)、[新物理验证](../../reports/stop_feedback_20261015/full_mission_retention_validation_20261016/README.md)、[模型与配对训练](../../models/stop_feedback_20261015/full_mission_retention_pair_s231016/README.md)。以下均为历史截面。

**2026-10-16 20:36 更新：专项停滞修复，整体升级仍未通过，验证进程已结束。** 新候选累计 2,500 次监督更新、PPO 0 步，3,728 条来源标签全部保留。两项完整常规任务、三项历史恢复、两项方向停靠及四项传感器任务均安全到达，无碰撞；这些集合不能合并为独立泛化成功率。一项噪声任务相对固定参考由 33.8 秒变为 41.8 秒（+23.7%），可运动却等待由 31 增至 77 帧，超过单任务耗时 +20% 的预设门槛，因此停止后续 20 项。完整常规 18 项和晚恢复均未完成本候选验收，定位器/路由器冻结，原实车模型保留。实际平均 CPU 0.35 核、峰值 RSS 约 2.36 GiB；本机 CPU 硬配额未生效，3 GiB 内存硬限制已核验。见[最新对比](../../reports/stop_feedback_20261015/training_comparison_20261016/README.md)、[已结束的验证及证据](../../reports/stop_feedback_20261015/normal_retention_validation_20261016/README.md)。

**2026-10-16 19:32 更新：原 2k 候选常规 16/18，未通过能力保留。** 同条件较早参考为 18/18；两例超时、零碰撞，方向停靠 2/2 和传感器 5/5 保留，晚恢复仍为 1/3。28 项诊断结束，停止扩大。新增“当前动作拒绝原因”输入的单因素研究候选完成 2,000 次监督更新、PPO 0 步，拟合 3,250/3,252 未准入；离线确实改变了 106 秒状态的主动等待，现仅运行一例隔离物理验证，不作为部署升级。定位/路由冻结，原实车模型保留。见[最新对比](../../reports/stop_feedback_20261015/training_comparison_20261016/README.md)、[原因输入实验](../../reports/stop_feedback_20261015/guard_reason_ablation_20261016/README.md)、[单例状态](../../reports/stop_feedback_20261015/guard_reason_physics_20261016/README.md)。

**2026-10-16 19:04 更新：有专项修复，整体升级未成立。** 新 2,000 次监督候选恢复了上一候选失败的原 95/90 秒历史导航；更晚的困难组仍为 1 到达、2 超时，均无碰撞。两项方向停靠保留，较广诊断继续运行。106 秒状态仍连续 170 帧有运动可选却停止，覆盖审计发现其合法动作组合没有运动训练标签，但尚无该状态完整安全恢复证据。定位器和路由器冻结，新增 PPO 0 步，部署准入未通过。原实车模型保留。见[最新对比](../../reports/stop_feedback_20261015/training_comparison_20261016/README.md)、[困难组与实时进度](../../reports/stop_feedback_20261015/context_broad_diagnostic_20261016/README.md)。

**2026-10-16 18:51 更新：三个恢复状态的物理诊断全部安全到达，较广诊断已启动。** 修正候选在原 95/90 秒恢复中由上一候选超时改善为接管后 19.8/20.4 秒到达，可选运动却等待 129→2、225→0 帧；额外等待状态 41.4 秒到达，0 帧主动等待。三个状态均无碰撞或保护干预，但均属已参与开发训练的来源回归，不能宣称整体或泛化胜出。

当前最多 42 项受限物理诊断先覆盖此前未解决的晚状态，再查方向、传感器和常规/额外导航。原拟合 3,251/3,252 门槛仍失败，仅使用仿真诊断包；原实车模型保留，PPO 新增 0 步，定位器没有更新。详见[三项证据与限制](../../reports/stop_feedback_20261015/context_physics_diagnostic_20261016/README.md)、[后续诊断进度](../../reports/stop_feedback_20261015/context_broad_diagnostic_20261016/README.md)。以下为历史记录。

2026-10-16 17:29，修正候选 `sensor_release_retained400_s231016` 实际完成 **350 次监督更新、PPO 0 步**。独立闭环 **7/7 安全到达、0 碰撞、0 超时**，另有 2 次上一版配对对照。已知长等待由上一阶段 90.6 → **34.8 秒**，主动等待 323 → 25 帧；原成功 36.8 秒、噪声保留 33.0 秒及方向停靠保留。两组未拟合新噪声从 35.0 → 33.8 秒、37.6 → 34.0 秒，路程也略短，但均为同一开发地图。长等待任务自身路程增加约 8.1%，不能称所有路径最短或整体已通过部署。

修正前的 300 次候选在另一个噪声条件退化为超时，已如实保留并拒绝替换；本轮补充完整安全保留轨迹后，才通过上述专项检查。定位器和路由器冻结，本轮不声称定位改善。其后完整回归在首两项失败后停止扩大，资源为单进程 0.35 核软预算、40% 单核/3 GiB 硬限制。原实车模型、封存包及远程快照保留。见[同条件数据、训练方法和边界](../../reports/stop_feedback_20261015/retained_sensor_release_validation_20261016/README.md)、[完整回归结果](../../reports/stop_feedback_20261015/retained_sensor_release_broad_20261016/RESULTS.json)。

此前 16:44 的阶段权重候选 `release_phase300_s231016` 实际更新 100 次、PPO 0 步，将旧候选的 140 秒超时改善为 90.6 秒安全到达，但最长连续等待仍为 58.6 秒。其三项有界检查见[历史结果](../../reports/stop_feedback_20261015/release_phase_validation_20261016/README.md)，不覆盖上面的新进度。

此前低资源诊断复现了旧候选的 700 帧失败记录，发现 636 帧由模型主动选择等待，且等待至少 5 秒后恢复只有 5 条不同训练输入。诊断与可选阶段加权入口见[说明](../../reports/stop_feedback_20261015/release_wait_context_20261016/README.md)。以下 CPU 交付仍封存旧候选，未包含上述新阶段权重模型。

2026-10-16，按 Orin Nano 8GB 与树莓派 5 的目标完成第一轮 CPU 优化。主机单线程配对回放中，CPU P50 **77.74 → 54.49 ms**，P95 **172.00 → 102.22 ms**；144 帧观测、掩码、动作和执行命令一致，25 项功能检查通过。独立包又完成导入、原 2k 回退配置、实际无 ROS worker 和 48 帧回放检查。

独立交付包为 `/home/ubuntu/rl_learn/realtest_edge_20261016.tar.gz`，约 20.3 MiB；535 个归档文件及 55 个未改动模型文件已校验。见 [板端验证包](../../exports/realtest_edge_20261016/README.md)、[优化测量与板端步骤](../../reports/embedded_runtime_20261016/README.md)。原交付包与原实车模型保留；本轮新增训练 0 步，尚无 Orin/树莓派实测或整机内存/功耗结论，本地新产物未自动推送远程。

以下是此前导航候选的收尾结果；其 11/12 闭环成绩来自 CPU 优化前。本轮性能回放不能替代完整闭环与实车验收。

---

# AGV 强化学习导航：当前导航候选收尾状态

2026-10-16 本轮已收尾，后台训练与仿真均结束。最新 625 维候选完成 12 个仿真条件：**11 到达、0 碰撞、1 超时**；旧等待失败条件改善，新噪声条件仍有长时间等待，未通过完整验收。原实车 2k 模型保留。

本轮新增 50 次监督更新、PPO 0 步；定位器、路由器和方向停靠保持冻结。按用户要求，先交付可回退候选进行实车验证，不再扩大训练。见[实车验证包 README](../../exports/realtest_20261016/README.md)、[完整结果](../../reports/stop_feedback_20261015/release_validation_20261016/README.md)。本地归档为 `/home/ubuntu/rl_learn/realtest_20261016.tar.gz`。GitHub handoff3 仍固定于此前 14:06 快照，本轮收尾产物未自动推送。

以下保留较早换机快照的历史说明；其中“当前”“运行中”仅指各自截止时间。

---

# AGV 强化学习导航：换机交接

本次新分支：**[`save`](https://github.com/gcusms/rl_learn_agv/tree/save)**。结果固定截至 **2026-10-16 08:40:06，北京时间**。包含当前训练与运行代码、定位内核源码、地图、原部署模型、研究候选、优化器、实际训练样本、精简实车数据，以及成功和失败轨迹。

**当前两组各 2,000 步 PPO 已训练完成，后续评估尚未全部完成。原实车部署模型保留，新候选尚未通过部署验收。** Git 保存的是这个时间点的文件；原电脑的进程及之后生成的结果不会自动随分支迁移。

精确进度见 [SNAPSHOT.json](../../reports/handoff_save_20261016/SNAPSHOT.json)，固定评估见 [validation](../../reports/handoff_save_20261016/validation/)，全文件哈希见 [FILES.json](../../docs/handoff/FILES.json)。上一版 `dev` 的说明、哈希和检查分别归档到 [DEV_README.md](../../docs/handoff/DEV_README.md)、[DEV_FILES.json](../../docs/handoff/DEV_FILES.json)、[DEV_VERIFICATION.json](../../docs/handoff/DEV_VERIFICATION.json)。历史文档中的“当前”指其各自截止时间。

## 1. 当前做到哪里

当前要解决的是：模型能区分停车原因，在有安全可行动作时合理恢复，同时保留避障、路径决策和位置＋方向停靠。结构为冻结基础 actor、localizer、router 和终端方向停靠部分，只训练有启用条件的恢复 residual。追加恢复 PPO 不会直接更新定位器或路由器。

| 模型 | 截止时的闭环结果 | 当前判断 |
|---|---|---|
| 原实车质量恢复 2k | 历史开发集 46/48；与本轮集合不同 | 保留为部署基线 |
| 参考 BC：2,089 标签、800 次监督更新 | 完整 18 项：16 到达、1 碰撞、1 超时；原成功 14/14；6 项方向停靠保留 | 本轮配对参考；新增 PPO 为 0 |
| 旧独立 8k PPO | 完整 18 项：16 到达、2 碰撞；原成功只保留 13/14 | 存在退化，未选作部署升级 |
| 新对照 PPO：2,000 步 | 完整 18 项：17 到达、0 碰撞、1 超时；原成功 13/14 | W11 原成功任务退化为超时，未通过保留要求 |
| 新约束 PPO：2,000 步 | **仅完成 16/18 项**：14 到达、1 碰撞、1 超时 | W11 完整任务恢复到达，但 `motion_profile_002` 超时；不能宣称整体更好 |
| 新诊断 BC：2,355 标签、800 次监督更新 | 专项 **仅完成 5/6 项**，已结束的 5 项均到达 | 未通过原预设全标签保留条件；独立诊断不等于准入或部署验收 |

两组新 PPO 的 BC 初始化、网络、优化器、种子、课程和保护相同，主要变量是示范保留损失系数 `0` / `0.05`；两组的长等待探索损失都关闭。首 250 个实际 rollout 完全一致。两个各 2k 模型不能合称单个模型训练 4k；训练中到达比例也不能当作独立评估成功率。

对照组的固定评估已完成，须分开阅读：

| 类别 | 完成行数 | 安全到达 | 碰撞/接触事件 | 超时 |
|---|---:|---:|---:|---:|
| 完整起点任务 | 18 | 17 | 0 | 1 |
| 原恢复历史 | 2 | 2 | 0 | 0 |
| W11 相关历史 | 3 | 0 | 0 | 3 |
| 额外同地图条件 | 12 | 7 | 4 | 1 |

这 35 行混合完整任务、相关前缀和同图路线，不能合并为独立泛化成功率。约束组的另 2 个原前缀、3 个 W11 历史和 12 条额外条件在本快照中尚未开始。新诊断 BC 的参考组 3 项和 32 秒恢复专项也尚未开始。

W11 从完整起点运行：参考 BC 安全到达约 37 秒、4.62 米；新对照超时 140 秒；新约束安全到达约 44 秒、4.75 米。这说明该完整任务有改善，不能据此认为“已经停很久后也能脱困”。逐项进度以快照为准。

## 2. 已查明的问题与接着做什么

已确认两类等待：有些状态只剩停止合法，需要检查感知、规划和动作可行性；另一些状态有合法运动动作，但策略依然偏向停止。逐动作反馈必须分别检查，不能只看最终执行停车的一个原因。

同输入审计中，近场转向诊断的 170 帧全部有运动可选、恢复分支启用，模型仍全部停止，最低停止概率约 0.9999988。已有示范缺少此前合法等待至少 2 秒后的有效恢复样本；对照 2k 的实际训练也未覆盖该类可训练状态。约束 2k 覆盖了 66 帧，但全部采样停止。仅增加当前课程步数不能保证补齐经验。

有限时段的合法右转物理试探没有碰撞，但仍超时，不把它贴成成功示范。训练碰撞已精确复现 25 帧，记录为原保守接触失败；没有通过修改判定把它重算为成功。详见 [原因诊断及证据](../../reports/stop_feedback_20261015/stop_support_diagnosis_20261016/README.md)。

换机后的推荐顺序：

1. 完成约束模型固定评估，与参考 BC、对照模型比较原成功任务、方向停靠、碰撞、等待、时间和路程。
2. 完成诊断 BC 与参考组的既定专项；它存在 6 条旧示范动作不一致，局部成功不代替正常导航回归。
3. 执行已固定的 W11 32 秒参考 BC 接管，验证“合法动作是否真正有用”；结果未知，不能提前当成功训练数据。
4. 有完整安全恢复证据后，再补相应历史的示范和实际 PPO 状态覆盖，进行单项对照。长等待探索损失默认关闭，先单独实验，不同时修改保护和奖励来归因。
5. 保留原 14 项成功、6 项方向停靠和合理停车，检查新增碰撞及假到达，再决定是否增大训练量。

## 3. 新电脑安装与检查

建议先在 Ubuntu 22.04、Python 3.10、CPU 环境复现。以下在同一终端执行；先用标准库核对文件，再安装依赖。

```bash
git clone --depth 1 --branch save https://github.com/gcusms/rl_learn_agv.git
cd rl_learn_agv
python3 tools/check_transfer.py

sudo apt-get install python3.10-venv g++ libeigen3-dev
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r training/requirements.txt
export PYTHONPATH=training:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# 可选完整安装检查：模型加载、输入合同、实现检查；输出目录须不存在。
nice -n 19 python tools/check_handoff.py --output training/runs/save_check
```

仓库文件约 1.1 GiB，环境安装建议另预留至少 8 GB，下载缓存另计。新机先单任务、单线程、`nice -n 19`、软预算 **0.35 核**，保留至少 4 GiB 可用内存。Jetson 使用与 JetPack 匹配的 PyTorch，不直接安装上述桌面 CPU 包；CUDA 不是复现检查的前提。

虚拟环境、缓存和平台 `.so` 不迁移。C++ 定位内核源码已包含，闭环运行前按下一节重新编译；编译不等于重新训练。`check_transfer.py` 只验证文件和完成状态，不启动训练或导航。`check_handoff.py` 会先重建一个定位检查包，再检查 13 份反馈模型、优化器、样本和实现，避免依赖旧机未上传的 `.so`。先前同一训练源码的 127 项实现检查全通过，记录保留在 DEV_VERIFICATION 中；本次换机检查见 [VERIFICATION.json](../../docs/handoff/VERIFICATION.json)，不能代替实车验收。加 `--build-native` 可进一步重建全部反馈包。

## 4. 模型及代码位置

所有路径相对仓库根目录。权重、manifest、观测语义、定位内核和保护配置需配套使用。

| 路径 | 用途 |
|---|---|
| `models/quality_recovery_20261014/policy_2000.zip` | 原实车部署模型，未替换 |
| `models/stop_feedback_20261015/w11_recovery_bc800_s231015/` | 参考 BC：`policy_bc.zip`、`bc_portable/`、优化器、RNG 和示范 |
| `models/stop_feedback_20261015/w11_control2000_s231015/` | 新对照 2k：最终 `policy.zip`、`portable/`、优化器及全部实际样本 |
| `models/stop_feedback_20261015/w11_retained2000_s231015/` | 新约束 2k，保留系数 0.05；完整训练产物同上 |
| `models/stop_feedback_20261015/early_transfer_bc800_s231015/` | 新诊断 BC，2,355 标签、800 次监督更新、PPO 0 步 |
| `models/stop_feedback_20261015/proactive_augmented_ppo_8000_s231015/` | 旧独立 8k PPO 完整样本、优化器与源码快照 |
| `models/localization_mapping_20261015/ray_consistent_candidate/portable/` | 独立定位算法候选，未合入本轮恢复对照 |
| `models/docking_selected_5000_20261015/` | 历史位置＋朝向停靠模型 |
| `training/neural_navigation/` | 地图、定位、路由、反馈、动作检查和方向目标运行时 |
| `training/experiments/` | 训练、监督初始化、实际样本审计和闭环评估 |
| `training/assets/real_recording_20261014/` | map_01、1,516 条实车决策、1,911 条反馈及轮速/IMU 精简数据 |
| `training/assets/localization_replay_20261015/` | 28 条冻结仿真回放；真值仅供监督和评价 |
| `reports/handoff_save_20261016/` | 本次固定进度、完成轨迹和原机任务状态 |
| `deployment/ros2_recording_adapter/` | ROS 2 记录与模型集成接口 |

原部署 SHA256 为 `7a3dad60f76060daf6b311b4bc6db921d3925ef0cdcf11683fe99e282173de11`。其他模型哈希见 SNAPSHOT。`policy_bc.zip` 是监督初始化；最终 `policy.zip` 才包含完整 PPO 更新。`policy_step_2000.zip` 在最后一次更新前保存，不能代替最终文件。恢复 ZIP 使用 `experiments.stop_recovery_policy.StopRecoveryPPO.load`，运行时按 manifest 的 `required_runtime` 解析。

## 5. 新电脑接着评估

在仓库根目录、已激活环境及第 3 节环境变量下执行。每个输出目录须不存在，各命令串行运行。下面完整重现约束组；不能将重复结果累计为新的独立成功样本。

```bash
model=w11_retained2000_s231015
python tools/prepare_native_bundle.py \
  --source models/stop_feedback_20261015/$model/portable \
  --output training/runs/save_$model

nice -n 19 python -m experiments.evaluate_docking_navigation \
  --candidate training/runs/save_$model --candidate-name $model \
  --cases reports/stop_feedback_20261015/w11_success_augmented_protocol_20261016/gate_cases.json \
  --output training/runs/save_${model}_navigation --cpu-cores 0.35
nice -n 19 python -m experiments.evaluate_feedback_recovery \
  --bundle training/runs/save_$model --label $model \
  --specifications reports/stop_feedback_20261015/frozen_recovery_cases.json \
  --output training/runs/save_${model}_prefixes --cpu-cores 0.35
nice -n 19 python -m experiments.evaluate_feedback_recovery \
  --bundle training/runs/save_$model --label $model \
  --specifications reports/stop_feedback_20261015/proactive_protocol_20261016/w11_prefixes.json \
  --output training/runs/save_${model}_w11 --cpu-cores 0.35
nice -n 19 python -m experiments.evaluate_docking_navigation \
  --candidate training/runs/save_$model --candidate-name $model \
  --cases reports/stop_feedback_20261015/w11_success_augmented_protocol_20261016/fresh_cases.json \
  --output training/runs/save_${model}_fresh --cpu-cores 0.35
```

本次 16 项已完成轨迹在固定 validation 中。评估入口没有自动跳过已完成行；若只运行余下任务，应从原 cases 按同一 id 筛选到新文件，保持参数不变，合并统计时按模型、任务和协议去重。要重现对照组，将 `model` 改为 `w11_control2000_s231015`。参考 BC 使用其 `bc_portable/`，完整配对示例见 [历史详细指南](../../docs/handoff/WORK_README.md)。

32 秒接管诊断可以脱离原机 systemd 队列直接运行；它是固定历史后的恢复测试，不是全程自主任务：

```bash
python tools/prepare_native_bundle.py \
  --source models/stop_feedback_20261015/w11_recovery_bc800_s231015/bc_portable \
  --output training/runs/save_reference_bc
nice -n 19 python -m experiments.evaluate_feedback_recovery \
  --bundle training/runs/save_reference_bc --label reference_bc_control_wait_t32 \
  --specifications reports/stop_feedback_20261015/reference_wait_t32_protocol_20261016/prefix.json \
  --output training/runs/save_reference_wait_t32 --cpu-cores 0.35
```

## 6. 训练入口与续训边界

入口为 `python -m experiments.train_prefix_recovery`。本轮完整 2k 配置见 [PLAN.json](../../reports/stop_feedback_20261015/retention_pair_20261016/PLAN.json)，可复制的相对路径训练命令见 [历史指南第 5 节](../../docs/handoff/WORK_README.md)。训练前先用 `tools/prepare_native_bundle.py` 重建父包，输出到新的 `training/runs/save_*` 目录。示范保留系数 0.05 与关闭组 0 分开对照；长等待探索系数本轮均为 0。

网络、优化器、RNG、课程配置、示范及实际 rollout 均保留，可加载分析和用于新实验设计。但现 CLI **不支持精确续跑同一半回合的物理状态、雷达历史和全部内部记忆**；不能将一次新初始化实验直接写成旧模型无缝追加步数。`--reference-bc` 是核对 BC 初始化，并非任意最终 PPO 检查点的续训开关。原机 `run.py`、`after_*.py` 的绝对路径、PID 与依赖关系只供审计；新机使用上述直接入口。

每组 2k 训练原机约 25.0 / 26.4 分钟，软预算 0.35 核，峰值 RSS 约 2.35 / 2.39 GiB。新机耗时应实测；评估与初始化不计入 PPO 步数。先完成已计划的验证，再根据失败来源决定更新恢复分支、常态 actor、定位器或路由器。

## 7. 定位、停车与实车条件

定位改善来自里程计传播、静态支持筛选、弱方向约束、时序融合和自由射线矛盾检查；本轮恢复 PPO 没有训练定位网络。独立定位对照在 22 条冻结轨迹、2,855 帧中，时序父版 / 射线候选位置 P95 为 18.51 / 18.04 cm，航向 P95 为 1.526 / 1.368°，CPU P95 为 7.88 / 14.59 ms；收益有限、耗时增加，尚未合入恢复对照。源码与验证见 [定位说明](../../docs/localization/README.md)。

实车录包没有独立绝对位姿真值，当前没有验证超过 AMCL、长期无漂移、任意搬车重定位或 Jetson Orin Nano 时延。已知地图、边导航边建图、开放稀疏场景仍需分别验收。

硬件合同保持：约 39.39×32.95 cm 车体、4 cm 余量；12 个离散动作，当前无横移、无倒车；最高前进 0.20 m/s、转向 0.50 rad/s。220 束原始雷达、约 219.44°，保留后方盲区。恢复训练 436 维、方向运行时 439 维，区别于原部署 300 维。位置＋朝向目标已接入，同位置不同朝向可转向，6 项方向停靠在固定回归集中。

当前闭环主要输入带噪声运动反馈；实车若仍为指令积分里程计，不能认为已经匹配。实车约 6%～12% 无效束、连续缺失、0.1/0.2 秒历史时间、10/12 米量程差异、新障碍未入静态图及低层超时停车还需实测。不能通过缩小车体或取消必要停车来提升表面成功率。

## 8. 迁移范围

当前研究源文件与必要产物已包含；历史工作区差异另有归档，不要求复刻本机的隐藏 worktree 目录。完整 `record.zip`、原始 rosbag/相机图像、虚拟环境、缓存和平台二进制未上传。当前停车分析与回放使用的精简数据已包含；若后续需重新提取原始话题，请另复制原机 `/home/ubuntu/rl_learn/record.zip`，源清单见 [MANIFEST.json](../../training/assets/real_recording_20261014/MANIFEST.json)。

仓库中的训练示范和本轮“额外条件”主要来自已用地图，不等于未见地图泛化。源记录中的旧绝对路径保留作审计，运行以本 README 的相对路径为准。原机后台评估仍可继续；本分支固定截止，结果更新需再次提交推送。
