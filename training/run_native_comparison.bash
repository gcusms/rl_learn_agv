#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
source training/activate.bash
source training/teb_ws/install/setup.bash
export ROS_DOMAIN_ID=72
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
python -u training/navigation_comparison.py --nav2-teb --output training/runs/navigation_comparison/nav2_teb_verified "$@"
