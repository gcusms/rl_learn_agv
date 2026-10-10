# Source from any directory; ROS is optional for learned MuJoCo navigation.
_rl_workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
fi
source "${_rl_workspace}/.venv/bin/activate"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
unset _rl_workspace
