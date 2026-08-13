#!/usr/bin/env bash
# 30-second go/no-go check for whether a rented box can run Isaac Sim / OmniGibson.
#
# Run this BEFORE installing anything (Isaac Sim is a ~20GB pull).
#
# Why this exists: on AutoDL boxes, CUDA/torch work perfectly while Vulkan sees
# no GPU at all. Isaac Sim needs Vulkan for the RTX renderer, and when GPU
# foundation finds no device, omni.physx then reports "CUDA libs are present,
# but no suitable CUDA GPU was found!" and segfaults. nvidia-smi and
# torch.cuda.is_available() both look fine in that state, so they are NOT
# sufficient evidence that a machine will work.
#
# IMPORTANT: the stock nvidia_icd.json points at libGLX_nvidia.so.0, which
# initializes through X11 and always fails in a headless container. That is a
# CONFIGURATION problem, not a dead machine — libEGL_nvidia.so.0 works headless.
# So this script tests BOTH, and only condemns a host when neither works.
# Apply fix_vulkan_headless.sh when GLX fails but EGL works.
#
# Usage: bash gpu_precheck.sh

fail=0
needfix=0
say() { printf '%-30s %s\n' "$1" "$2"; }

PY=$(command -v python3 || command -v python || ls /root/miniconda3/bin/python3 2>/dev/null | head -1)

echo "=== Isaac Sim / OmniGibson host pre-check ==="

# 1. GPUs visible at all
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  say "GPUs visible:" "$(nvidia-smi -L | wc -l) x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  say "driver:" "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  # RT cores are required for the RTX renderer. A100/H100/V100 have none.
  case "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" in
    *A100*|*A800*|*H100*|*H800*|*H20*|*V100*)
      say "RT cores:" "NO  <-- FATAL, this GPU cannot run the RTX renderer"; fail=1 ;;
    *) say "RT cores:" "yes (RTX-class)" ;;
  esac
else
  say "GPUs visible:" "NO  <-- FATAL"; fail=1
fi

# 2. Can either NVIDIA ICD library hand out vkCreateInstance?
#    GLX = X11 path (fails headless), EGL = headless path (what we want).
probe() {
  "$PY" - "$1" <<'PY' 2>/dev/null
import ctypes, sys
try:
    lib = ctypes.CDLL(sys.argv[1])
    f = lib.vk_icdGetInstanceProcAddr
    f.restype = ctypes.c_void_p
    f.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    print("OK" if f(None, b"vkCreateInstance") else "NULL")
except Exception as e:
    print("LOADFAIL:%s" % e)
PY
}
glx=$(probe libGLX_nvidia.so.0)
egl=$(probe libEGL_nvidia.so.0)
say "ICD via libGLX (X11):" "$glx"
say "ICD via libEGL (headless):" "$egl"

if [ "$egl" != "OK" ] && [ "$glx" != "OK" ]; then
  say "verdict:" "neither ICD works  <-- FATAL"; fail=1
elif [ "$egl" = "OK" ] && [ "$glx" != "OK" ]; then
  needfix=1
fi

# 3. Does a real (non-llvmpipe) Vulkan device actually enumerate right now?
if command -v vulkaninfo >/dev/null 2>&1; then
  typ=$(vulkaninfo --summary 2>/dev/null | grep -m1 'deviceType' | sed 's/.*= //')
  dev=$(vulkaninfo --summary 2>/dev/null | grep -m1 'deviceName' | sed 's/.*= //')
  say "Vulkan device:" "${dev:-none} / ${typ:-none}"
  case "$typ" in *DISCRETE_GPU*) ;; *) [ $fail -eq 0 ] && needfix=1 ;; esac
else
  say "Vulkan device:" "vulkaninfo not installed (fix script installs it)"
fi

echo
if [ $fail -ne 0 ]; then
  echo "RESULT: FAIL — do NOT install here. No usable RTX/Vulkan path."
  exit 1
elif [ $needfix -ne 0 ]; then
  echo "RESULT: FIXABLE — the GPU is fine, the ICD config is wrong."
  echo "        Run:  bash fix_vulkan_headless.sh"
  echo "        Then: export VK_ICD_FILENAMES=/etc/vulkan/icd.d/my_nvidia_icd.json"
  exit 2
else
  echo "RESULT: PASS — Vulkan sees a discrete GPU. Safe to install Isaac Sim."
  exit 0
fi
