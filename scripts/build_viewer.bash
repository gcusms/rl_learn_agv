#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
gcc -shared -fPIC training/navrl/glfw_capture.c -o training/navrl/glfw_capture.so -Wl,--no-as-needed -l:libglfw.so.3 -ldl
