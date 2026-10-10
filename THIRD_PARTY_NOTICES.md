# 第三方来源与授权

- training/teb_ws/src/teb_local_planner：原工作区保留的上游 ROS 2 TEB 源码，包含本案例 Humble 兼容和 TF 接入修改。原 LICENSE 文件保留，见该目录下 teb_local_planner/LICENSE。
- training/teb_ws/src/costmap_converter：原工作区的上游源码；package.xml 中的许可证声明与文件头保留，请按相应声明使用。
- agv/：从用户原 agv 源码复制的离线机器人资源及 arm_state_bridge.py，保留原文件和 package.xml；未替资源作者赋予新的授权。
- Python/ROS/MuJoCo 依赖通过安装获取，不打包第三方虚拟环境或系统库。

本整理不添加新的根项目开源许可证，公开分发时仍需由权利人确定授权范围。
