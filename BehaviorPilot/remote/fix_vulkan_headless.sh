#!/usr/bin/env bash
# Make Vulkan work in a headless AutoDL-style container (per AutoDL's own docs:
# https://www.autodl.com/docs/vulkan/).
#
# The problem: the ICD that ships with the driver mount points at
# libGLX_nvidia.so.0, which initializes through X11/GLX. In a container with no
# display, that path bails out immediately — vk_icdGetInstanceProcAddr returns
# NULL, so the Vulkan loader finds zero devices and falls back to llvmpipe.
# Isaac Sim then can't create a GPU device, and omni.physx reports
# "CUDA libs are present, but no suitable CUDA GPU was found!" before segfaulting.
#
# The fix: point the ICD at libEGL_nvidia.so.0 instead, which initializes
# headless via EGL.
#
# Usage: bash fix_vulkan_headless.sh
# Then:  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/my_nvidia_icd.json
#        (put that in ~/.bashrc and in any render/production script)

set -e
ICD=/etc/vulkan/icd.d/my_nvidia_icd.json

echo "== installing deps =="
# vulkan-tools    -> vulkaninfo, for verification
# libvulkan1      -> the loader itself
# libegl1         -> EGL dispatch, required by the libEGL_nvidia path
# libsm6/libxt6/libglu1-mesa -> Isaac Sim's omni.usd.libs and iray plugins fail
#                   to load without these (libXt.so.6 / libGLU.so.1)
apt-get update -qq
apt-get install -y -qq vulkan-tools libvulkan1 libsm6 libegl1 libxt6 libglu1-mesa

echo "== locating libEGL_nvidia =="
EGLLIB=$(ldconfig -p | grep -m1 'libEGL_nvidia.so.0' | awk '{print $NF}')
if [ -z "$EGLLIB" ]; then
  echo "FATAL: libEGL_nvidia.so.0 not found — the container was not given the"
  echo "       'graphics' driver capability. This fix cannot help; get another host."
  exit 1
fi
echo "   found: $EGLLIB"

echo "== writing $ICD =="
mkdir -p /etc/vulkan/icd.d
cat > "$ICD" <<EOF
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "$EGLLIB",
        "api_version" : "1.3.277"
    }
}
EOF
cat "$ICD"

echo "== verifying =="
export VK_ICD_FILENAMES="$ICD"
dev=$(vulkaninfo --summary 2>/dev/null | grep -m1 'deviceName' | sed 's/.*= //')
typ=$(vulkaninfo --summary 2>/dev/null | grep -m1 'deviceType' | sed 's/.*= //')
echo "   deviceName = ${dev:-<none>}"
echo "   deviceType = ${typ:-<none>}"

case "$typ" in
  *DISCRETE_GPU*)
    echo
    echo "RESULT: PASS — Vulkan sees the real GPU. Add this to ~/.bashrc:"
    echo "   export VK_ICD_FILENAMES=$ICD"
    ;;
  *)
    echo
    echo "RESULT: FAIL — still $typ (software rasterizer). This host is unusable"
    echo "        for Isaac Sim; see gpu_precheck.sh notes."
    exit 1
    ;;
esac
