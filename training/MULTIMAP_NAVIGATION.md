# 多地图、随机任务与可观察导航决策

这一版采用模块化导航：激光地图定位 → 地图/激光代价地图规划 → PPO 局部速度控制。定位与 A* 不是神经网络学习出来的；PPO 输入连续三帧实际有限视场激光、定位后的局部路标和目标、里程计速度、上一动作与定位协方差，学习输出前进速度和转向速度。没有把 MuJoCo 真值位置或行人位置输入模型，也没有把成功演示解释为全部场景的成功率。

`assets/navigation_dataset/manifest.json` 保存六张训练、两张验证、三张测试地图的生成种子、地图标识与占据图 SHA256。每次采样新的起终点（间距至少 2 米、静态路径不超过 9 米）和整个圆周内随机朝向。这是不同布局上的初步泛化测试，不能证明任意地图、拥挤度和机器人速度都有效。

决策记录字段 `decision` 包含：

- `static_map_reachable`：已知地图和机器人膨胀尺寸下是否有可行路线。
- `state` / `reason`：跟随路线、重规划、等待动态区域清空、定位不确定、目标到达或超时的观测依据。
- `plan_revision`：路线变化次数；录像与网页按每帧记录更新青色路线。
- `unmatched_laser_returns`：与地图不匹配的激光返回点数量。这些点保留一秒并按机器人空间需求膨胀。
- `localization_std`：激光定位的估计不确定性，不是使用真值计算的误差。

PPO 请求动作保存在 `proposed_policy_action`。`supervisor_override` 为真表示规则监督器覆盖动作并等待，不能算作模型主动停车。物理接触导致停止另标为 `collision_blocked`。`policy_stop` 只统计没有监督器覆盖的模型低速请求。

实体机器人与行人使用 `mj_step` 和原生接触，正常运动不写位置。行人会加减速、停止和让路。透明碰撞外形的 alpha 必须非零：MuJoCo 射线会忽略 alpha=0 的几何体。修正后有原生激光返回及绕行检查。`multimap_warm`、`multimap_ppo`、`multimap_warm_validation`、`multimap_warm_stress` 记录受此问题影响，已标 `INVALIDATED.json`，不能用于结论。正确训练输出使用 `multimap_visible_*` 名称。

## 使用

```bash
source training/activate.bash
python training/generate_navigation_dataset.py
python training/warm_start_navigation.py --multi-map --episodes 24 --epochs 45 --resume training/runs/navigation_refined/policy.zip --seed 8100 --learning-rate 0.0001 --learner-mix 0.15 --output training/runs/multimap_visible_warm
python training/train_navigation.py --multi-map --localizer particle --steps 12288 --seed 9200 --learning-rate 0.00003 --resume training/runs/multimap_visible_warm/warm_start.zip --demo-anchor training/runs/multimap_visible_warm/demonstrations.npz --output training/runs/multimap_visible_ppo --max-episode-steps 700
python training/evaluate_navigation.py --multi-map --split test --localizer particle --policy training/runs/multimap_navigation/policy.zip --episodes 15 --seed 12000 --max-episode-steps 700 --record-success --output training/runs/multimap_visible_test
```

独立原生 Nav2 AMCL 检查使用 `--localizer ros-amcl`，每次切换地图会关闭旧定位节点并加载新地图。训练用粒子定位以减少运行开销，二者都通过激光匹配地图，不将真值提供给策略。

导入自己的 ROS YAML 地图：

```bash
python training/import_navigation_map.py /absolute/path/map.yaml --output training/assets/my_scene
python training/evaluate_navigation.py --scene-dir training/assets/my_scene --localizer ros-amcl --policy training/runs/multimap_navigation/policy.zip --start 1 1 0 --goal 8 6 --episodes 1 --record-success --output training/runs/my_scene_eval
MUJOCO_GL=glfw python training/render_success.py training/runs/my_scene_eval/successful_replay.json --output training/runs/my_scene_demo
```

起终点采用地图坐标，朝向单位弧度，需按自己的地图填写。导入工具把占据/未知像素合并成一米高墙体，不能从二维地图恢复物体真实高度。暂不支持旋转地图原点、多楼层和坡道；保留真实 URDF 模型与 SCLIDAR 前方约 219.44° 的扫描。随机出生限制在地图空闲区域。

本次实车不运行。运动模型是受水平约束的刚体及力伺服，不是经过实车标定的轮胎与人类运动模型。定位假设有粗略初始位姿；尚未验证无初始位姿的全地图定位、绑架恢复或长期运行。地图可达性判断不等于可校准的“成功概率”；动态障碍未来不可预知，最终成效必须看独立评估中的实际终点距离、超时、碰撞与失败回合。

本轮完成 12,288 步 PPO，模型选择仅使用两张验证地图上的 6 个任务：最终模型与 5,000 步检查点均 6/6 抵达且零接触；按总用时选择 5,000 步检查点，固定到 `runs/multimap_navigation/policy.zip`，选择依据保存在同目录 config.json。测试后不再按测试成功率切换模型。

独立测试：三张未见地图、随机起终点，粒子定位 14/15（93.3%），真实 Nav2 AMCL 6/6，实体行人在初始路线暂停 10 秒压力测试 3/3。以上 24 回合均零碰撞且零穿透回退。种子 12012 的 test_00 回合因监督器长时间等待而超时，同模型同种子复现了完全相同结果；完整失败激光与动作记录已保留。不能用后两组的小样本 100% 宣称通用成功率。

最终展示：`runs/multimap_demo/index.html`，成功回放：`runs/multimap_demo/replay.html`，验证报告：`runs/multimap_demo/validation.json`。AMCL 演示实际终点距离 0.208 米，12 次路线变化。地图不可达的两个空闲区域用原生墙体隔开的场景检查也已通过，见 `runs/disconnected_check/`。
