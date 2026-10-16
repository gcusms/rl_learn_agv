# 后续诊断入口修正

原任务 `rl-post-aux-diagnostics-20261016.service` 在 08:12:16 确认退出，`MainPID=0`、`exit-code=1`。原因是 `training_case(spec['case'])` 将普通初始化任务 `motion_profile_015` 送入只接受 `train_failure_*` 的转换器。异常发生在 Python 求构造参数时，尚未创建物理课程或重放动作。原空输出目录、失败状态、源码与 134 项输入计划保留；本目录保存完整失败日志和 [ORIGINAL_FAILURE.json](ORIGINAL_FAILURE.json)。

修正仅将课程构造输入恢复为实际训练所用的 12 条已记录训练任务。`PrefixRecoveryTraining` 仍从同一 30 帧物理初始化历史到达 6 秒，并重放原 25 个合法采样动作。没有给普通任务伪造训练失败 id。每帧依旧严格核对 436 维观测、掩码、指令、原因、奖励与终止，接触观察不增加物理步进。

`prepare_fix.py` 已核实原失败 invocation、两个旧辅助进程终止、原 134 项输入未变、五份数据与任务文件逐字节相同，以及驱动顺序未变。新 [PLAN.json](PLAN.json) 固定 145 个输入，包含原计划、失败日志及修正版源码。原三项初测仍被其全标签条件拒绝；本研究仍仅是之前明确分开的故障诊断，不改变原准入条件。

服务 `rl-post-aux-fixed-20261016.service` 于 08:15:44 启动，复用空出的同一个辅助槽。软预算 0.15 核、硬限制 20%、Nice 19、内存 High 2.5 GiB / Max 3 GiB。主任务及部署模型不变。启动证据见 `ORIGINAL_MACHINE_UNIT.json`，过程见 `STATUS.json`。

顺序保持“原碰撞重放 → 新 BC 六条完整任务 → 参考 BC 三条附加任务”。重放完成前不进入后两项；任何一致性失败均保留并退出。没有自动扩大训练、放宽掩码、发布模型或把训练重现算成未见泛化。导航输出目录与原计划一致，启动前确认尚未创建。

本目录的入口变更已完成；物理结果以各实际 `RESULTS.json` 为准，不以准备成功或服务启动代替验证。

08:17:14 的 [碰撞重放结果](collision_replay/RESULTS.json) 已完成：原 25 帧的观测、掩码、指令、原因、奖励及终止均一致；30 帧初始化另计。原生接触最小正间距约 1.485 cm，原失败计数仍为 1，物理穿透保护为 0。其后已进入候选六条完整任务，整组结果仍待完成。完整原因分析见 [COLLISION_FORENSICS.json](../stop_support_diagnosis_20261016/COLLISION_FORENSICS.json)。
