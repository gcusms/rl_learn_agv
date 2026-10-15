# 停车反馈后的探索与恢复实验

这是 `next` 交接快照之后的继续开发，工作分支为 `learn`。目标仍是让模型理解实际停车原因，并在存在安全、有用的动作时恢复导航；目前未解决、未部署。

## 已完成的反证与诊断

V2 的 14 例配对评估完整结束：基线 12 成功、2 碰撞；V2 为 11 成功、3 碰撞，丢失 `train_failure_stop_go_variation_1`。共同成功任务路程 +0.27%、耗时 +1.54%。因此 V2 不采用，计时修复正确不足以证明行为改善。见 [完整结果](../pilot_v2_4000_final/SUMMARY.json)。

[等待审计](../v2_wait_audit_4000/RESULTS.json)显示这 14 例最长途中主动等待仅 0.2 s，不足以验收长期停滞。原基线途中主动等待 0.8 s、仅停止可选 22.8 s；V2 分别 0.6 s、29.0 s。不能用少等候 0.2 s 抵消多一次碰撞。

对 V2 保存的真实 PPO 样本筛选出 **326 帧**“已等待至少 2 s、仍有合法运动选项、继续选停止”。主要动作集合是 `[0,1,6,8,10]`（166 帧）和 `[0,8]`（115 帧）。部分原采样停车概率约 99.6%，局部路径点落在车身后方。这说明探索稀少，但不能仅凭路径点在后方认定路由错误。完整选择规则、逐帧数据和源文件哈希见 [STAGNATION.json](STAGNATION.json)。

同一横穿任务的四组早期干预：小右弧安全到达 88.6 s；原地右转后超时 140 s；快右弧碰撞 12.8 s；提前等待后安全到达 38.2 s。见 [物理干预](../earlier_action_alternatives_complete/RESULTS.json)。它们证明部分避让序列物理可行，**不属于训练模型成绩**。后续三组二次干预中，慢左转、左弧仍超时；短暂前进后安全到达 54.4 s，说明存在可行恢复序列，但单次合法动作不保证恢复。结果见 [二次干预](../turn_recovery_20261015/RESULTS.json)。

## 本轮唯一主要变量：探索分布

从原始 300 维父模型重新初始化，仍为 4,000 步、种子 `221015`、相同课程、奖励、12 动作、车体、速度和保护。保持 V2 436/439 维输入。新增 `sensor_temperature_v1`：有运动选项、远离目标且扫描/预测输入有效时，依据已有动态迹象或因果等待计时，将 logits 温度从 1 有界提高到 3；没有新增真值输入或教师命令。

正温度不改变确定性动作排序，所以部署仍由学习后的 actor 决策。训练采样、旧概率、PPO 更新、熵和 KL 使用同一温度分布；正常能力保留项比较未经温度缩放的网络偏好，避免将声明的探索本身当作回归。停车保持合法，保护不放宽。这只是待验证的探索改进，不保证导航改善。

35 项检查已通过，覆盖 V1/V2 默认兼容、真实停滞状态概率、掩码、采样/更新概率一致、保存重载、TorchScript 与方向分支。另 4 项前缀测试通过。首批 250 个真实物理样本的采样审计：118 帧启用温度，其中 112 次非零、6 次停止；旧概率最大重算差 3.58e-6，未采到非法动作，初始确定性排序保持。见 [采样审计](../exploration_sampling_audit_250/RESULTS.json)。这些不作为导航成功率。

## 当前训练结果与验收顺序

- 4,000 步训练已完成，1,677.84 s、峰值 2,459.81 MiB；16 个训练回合结束，10 成功、4 碰撞、2 超时，不是验证集成绩。4k 实际原因/计时审计通过；有运动选项的主动零速 58 帧，最长因果等待仅 0.6 s，说明本轮仍没有充分采到长等待恢复，不能只靠少停车断言问题解决。
- 新模型：`models/stop_feedback_20261015/ppo_explore_4000_s221015`；服务 `rl-stop-feedback-explore-20261015.service`，训练后自动做 14 例配对。软平均 0.35 核，硬上限半核，内存硬上限 4 GiB。
- 静态冻结恢复任务：[frozen_recovery_cases.json](../frozen_recovery_cases.json)。从两条物理仿真实际进入的停滞状态重放前 20 s 的扫描/动作，完整保留里程计、定位、扫描历史、保护记忆及等待状态，再由新旧模型各自接管。前缀不算模型的导航成绩；接管后单独报告到达、碰撞和长等待。两条前缀的物理/传感器复现检查均通过：100 帧前缀后仅交给模型 1 帧，不能据此评价恢复是否成功。
- 先完成本轮训练和配对安全回归，再跑完整恢复任务；改善须同时表现为恢复后能到达、碰撞不增加、已有成功与方向停车不退化。额外干预或功能检查不能代替闭环。
- 若探索增加而仍不恢复，下一步检查合法动作能否形成持续可行的动作序列、策略是否又回到原停滞姿态，以及实际进展/价值信用分配。不能凭本轮接口通过就宣布目标完成，也不能只追加步数。

## 复核命令

```bash
export PYTHONPATH=training:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m experiments.audit_exploration_rollouts \
  --model-run models/stop_feedback_20261015/ppo_explore_4000_s221015 \
  --output training/runs/exploration_sampling_audit

# 仅在最终 portable/ 已生成，并在本机重建所需原生包后执行。
nice -n 19 python -m experiments.evaluate_feedback_recovery \
  --bundle training/runs/rebuilt_exploration_bundle \
  --specifications reports/stop_feedback_20261015/frozen_recovery_cases.json \
  --output training/runs/exploration_recovery --label explored --cpu-cores .15
```

PPO 的采样与似然一致性按[原始论文](https://arxiv.org/abs/1707.06347)和本地 sb3-contrib 2.7.0 实现复核；本项目的状态温度设计是基于上述数据提出的实验，不是论文已证明的导航结论。[DAPG 作者项目](https://sites.google.com/view/deeprl-dexterous-manipulation/dapg)作为示范辅助的备选参考，本轮未加入示范损失或替代控制器。
