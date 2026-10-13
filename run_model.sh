#!/usr/bin/env bash
# 默认值可在这里修改；命令行参数会覆盖这些默认值。
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_STEPS=100000
DEFAULT_SEED=31002
DEFAULT_MAX_LEG_STEPS=1800
DEFAULT_COMPONENTS="training/runs/remote_room_100k"
DEFAULT_OUTPUT="training/runs/manual_model_test"
DEFAULT_HEADLESS=0

usage() {
  cat <<EOF
用法：./run_model.sh [参数]
默认：原始 10 万步模型，四目标导航，两根柱子手动控制。

  --steps N              选择 PPO 检查点（已包含 10000、50000、100000）
  --policy PATH          指定策略文件，优先于 --steps
  --components PATH      定位/规划权重目录
  --seed N               随机种子，默认 ${DEFAULT_SEED}
  --max-leg-steps N      每个目标最多决策步数，默认 ${DEFAULT_MAX_LEG_STEPS}
  --output PATH          测试记录目录，默认 ${DEFAULT_OUTPUT}
  --headless             无窗口运行
  --gui                  打开交互窗口（默认）
  --dry-run              只显示启动命令
  -h, --help             查看帮助

相对路径均以脚本所在的仓库根目录为基准。
窗口操作：双击柱子后右键拖动，松开取消；空格暂停；R 重置。
此脚本运行原始模型，不使用视频演示的附加调参。
EOF
}
fail() { printf "错误：%s\n" "$*" >&2; exit 2; }
need_value() { [[ $# -ge 2 && "$2" != --* ]] || fail "$1 缺少参数值"; }
steps="$DEFAULT_STEPS"; seed="$DEFAULT_SEED"; max_steps="$DEFAULT_MAX_LEG_STEPS"
components="$DEFAULT_COMPONENTS"; output="$DEFAULT_OUTPUT"
headless="$DEFAULT_HEADLESS"; dry_run=0; policy=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps) need_value "$@"; steps="$2"; shift 2 ;;
    --policy) need_value "$@"; policy="$2"; shift 2 ;;
    --components) need_value "$@"; components="$2"; shift 2 ;;
    --seed) need_value "$@"; seed="$2"; shift 2 ;;
    --max-leg-steps) need_value "$@"; max_steps="$2"; shift 2 ;;
    --output) need_value "$@"; output="$2"; shift 2 ;;
    --headless) headless=1; shift ;;
    --gui) headless=0; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "未知参数 $1；使用 --help 查看帮助" ;;
  esac
done
[[ "$steps" =~ ^[0-9]+$ ]] || fail "--steps 必须是整数"
[[ "$seed" =~ ^[0-9]+$ ]] || fail "--seed 必须是非负整数"
[[ "$max_steps" =~ ^[0-9]+$ && "$max_steps" -gt 0 ]] || fail "--max-leg-steps 必须是正整数"
cd "$ROOT"
[[ -n "$policy" ]] || policy="training/runs/remote_room_100k/policy_${steps}.zip"
[[ -f "$policy" ]] || fail "找不到策略文件：$policy"
[[ -f "$components/localizer.pt" && -f "$components/router.pt" ]] || fail "定位/规划模型不完整：$components"
[[ -x "$ROOT/.venv/bin/python" ]] || fail "没有项目虚拟环境，请先按 README 安装"
cmd=("$ROOT/.venv/bin/python" "$ROOT/training/preview_live_navigation.py" --policy "$policy" --components "$components" --seed "$seed" --max-leg-steps "$max_steps" --output "$output")
if [[ "$headless" -eq 1 ]]; then cmd+=(--headless); fi
if [[ "$dry_run" -eq 1 ]]; then printf "%q " "${cmd[@]}"; printf "\n"; exit 0; fi
if [[ "$headless" -eq 0 ]]; then
  [[ -n "${DISPLAY:-}" ]] || fail "没有图形 DISPLAY。请在远程电脑桌面的终端启动，或使用 --headless"
  [[ -f "$ROOT/training/navrl/glfw_capture.so" ]] || bash "$ROOT/scripts/build_viewer.bash"
  export MUJOCO_GL=glfw
else
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi
source "$ROOT/training/activate.bash"
printf "模型：%s\n种子：%s；每目标最多 %s 步\n输出：%s\n" "$policy" "$seed" "$max_steps" "$output"
exec "${cmd[@]}"
