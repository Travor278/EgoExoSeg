#!/usr/bin/env bash
# Bootstrap mint2027 (dell@100.110.216.0) for V2-SAM inference + BEHAVIOR-1K sim track.
#
# Usage:
#   bash mint2027_bootstrap.sh probe      # hardware/network report only (writes ~/egoexo_setup_report.md)
#   bash mint2027_bootstrap.sh v2sam      # conda env `v2sam` + repo + weights (SAM2-L / DINOv3 / V2-SAM ckpt)
#   bash mint2027_bootstrap.sh behavior   # clone BEHAVIOR-1K + env (assets NOT downloaded unless --with-assets)
#   bash mint2027_bootstrap.sh behavior --with-assets   # + OmniGibson dataset assets (needs >150GB free)
#
# Design notes:
# - Mirror selection is measured, not assumed: times a small download from each candidate
#   and exports PIP_INDEX_URL / HF_ENDPOINT for this run.
# - V2-SAM weights are self-hosted by the author on HF (jaychempan/sam2, jaychempan/dinov3,
#   jaychempan/V2-SAM) -> no Meta gating, hf-mirror works.
# - Pinned recipe from V2-SAM README: torch 2.3.1+cu121, mmcv 2.1.0 (cu121/torch2.3),
#   mmdet 3.3.0 etc. If the GPU is Blackwell (sm_120, RTX 50xx) this pin CANNOT work --
#   the script detects that and stops with instructions instead of producing a broken env.
set -uo pipefail
LOG=~/egoexo_setup.log
REPORT=~/egoexo_setup_report.md
exec > >(tee -a "$LOG") 2>&1

step() { echo -e "\n=== $* ==="; }

# ---------------------------------------------------------------- probe
probe() {
  step "hardware probe -> $REPORT"
  {
    echo "# mint2027 setup report ($(date '+%F %T'))"
    echo '```'
    hostname; uname -a
    echo "--- os ---"; head -3 /etc/os-release 2>/dev/null
    echo "--- gpu ---"; nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv 2>/dev/null || echo "NO NVIDIA GPU"
    echo "--- gpu busy ---"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null
    echo "--- disk ---"; df -h / /home /data 2>/dev/null | sed -n '1,5p'
    echo "--- mem ---"; free -h | head -2
    echo "--- cpu ---"; nproc
    echo "--- conda ---"; command -v conda mamba micromamba 2>/dev/null || echo none
    echo '```'
  } | tee "$REPORT"
}

# ---------------------------------------------------------------- mirrors
fastest_pip() {
  local best="" best_t=99
  declare -A cands=(
    [tuna]="https://pypi.tuna.tsinghua.edu.cn/simple"
    [aliyun]="https://mirrors.aliyun.com/pypi/simple"
    [ustc]="https://mirrors.ustc.edu.cn/pypi/simple"
    [sjtu]="https://mirror.sjtu.edu.cn/pypi/web/simple"
    [pypi]="https://pypi.org/simple"
  )
  for k in "${!cands[@]}"; do
    t=$(curl -o /dev/null -s -m 8 -w '%{time_total}' "${cands[$k]}/pip/" || echo 99)
    echo "  pip mirror $k: ${t}s" >&2
    awk -v a="$t" -v b="$best_t" 'BEGIN{exit !(a<b)}' && { best_t=$t; best=${cands[$k]}; }
  done
  echo "$best"
}

fastest_hf() {
  for ep in "https://hf-mirror.com" "https://huggingface.co"; do
    t=$(curl -o /dev/null -sL -m 8 -w '%{time_total}' "$ep/api/models?limit=1" || echo 99)
    echo "  hf endpoint $ep: ${t}s" >&2
    awk -v a="$t" 'BEGIN{exit !(a<15)}' && { echo "$ep"; return; }
  done
  echo "https://hf-mirror.com"
}

setup_mirrors() {
  step "measuring mirrors"
  export PIP_INDEX_URL=$(fastest_pip)
  export HF_ENDPOINT=$(fastest_hf)
  echo "chosen PIP_INDEX_URL=$PIP_INDEX_URL"
  echo "chosen HF_ENDPOINT=$HF_ENDPOINT"
}

# ---------------------------------------------------------------- conda
ensure_conda() {
  if command -v conda >/dev/null 2>&1; then return; fi
  if [ -x "$HOME/miniforge3/bin/conda" ]; then eval "$($HOME/miniforge3/bin/conda shell.bash hook)"; return; fi
  step "installing miniforge (TUNA mirror)"
  curl -L -o /tmp/miniforge.sh \
    "https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh" \
    || curl -L -o /tmp/miniforge.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
  eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
}

gpu_arch_guard() {
  local cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
  echo "GPU compute capability: ${cap:-unknown}"
  case "$cap" in
    12.*)
      echo "!! Blackwell GPU (sm_120): the pinned torch 2.3.1+cu121 / mmcv 2.1.0 recipe cannot run here."
      echo "!! Needed instead: torch>=2.7 cu128 + mmcv built from source. Stopping so we do not build a broken env."
      exit 2 ;;
  esac
}

# ---------------------------------------------------------------- v2sam
v2sam() {
  probe; setup_mirrors; ensure_conda; gpu_arch_guard
  eval "$(conda shell.bash hook)"

  step "conda env v2sam (python 3.10)"
  conda env list | grep -q '^v2sam ' || conda create -y -n v2sam python=3.10
  conda activate v2sam

  step "torch 2.3.1 cu121"
  python -c "import torch" 2>/dev/null || \
    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
      --index-url https://download.pytorch.org/whl/cu121
  python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

  step "clone V2-SAM"
  cd ~; [ -d V2-SAM ] || git clone https://github.com/jaychempan/V2-SAM.git
  cd V2-SAM

  step "python deps (via $PIP_INDEX_URL)"
  pip install -U "huggingface_hub[cli]"
  pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.3/index.html
  pip install -r requirements.txt

  step "weights via $HF_ENDPOINT (author-hosted, no gate)"
  mkdir -p weights/sam2 weights/dinov3 weights/v2sam
  huggingface-cli download jaychempan/sam2   --local-dir weights/sam2   --include "sam2_hiera_large.pt"
  huggingface-cli download jaychempan/dinov3 --local-dir weights/dinov3 --include "dinov3_vitl16_pretrain_lvd1689m*"
  huggingface-cli download jaychempan/V2-SAM --local-dir weights/v2sam
  ls -lh weights/sam2 weights/dinov3 weights/v2sam

  step "v2sam env DONE. Smoke test:"
  echo "  conda activate v2sam && cd ~/V2-SAM && bash tools/test.sh test projects/v2sam/configs/v2sam.py 1 weights/v2sam/<ckpt>"
}

# ---------------------------------------------------------------- behavior
behavior() {
  probe; setup_mirrors; ensure_conda
  eval "$(conda shell.bash hook)"
  local with_assets="${1:-}"

  free_gb=$(df --output=avail -BG "$HOME" | tail -1 | tr -dc '0-9')
  echo "free space in \$HOME: ${free_gb}GB"
  if [ "${free_gb:-0}" -lt 80 ]; then
    echo "!! <80GB free: Isaac Sim + OmniGibson needs ~60GB (+assets). Pick another disk or clean up first."; exit 2
  fi

  step "clone BEHAVIOR-1K"
  cd ~; [ -d BEHAVIOR-1K ] || git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
  cd BEHAVIOR-1K

  step "run official setup (new conda env 'behavior')"
  # --dataset pulls the OmniGibson scene/object assets; only with --with-assets
  local flags="--new-env --omnigibson --bddl"
  if [ "$with_assets" = "--with-assets" ]; then
    if [ "${free_gb:-0}" -lt 150 ]; then echo "!! assets need >150GB free"; exit 2; fi
    flags="$flags --dataset"
  fi
  # flag names vary across BEHAVIOR-1K versions; on failure show help instead of a half-built env
  if ! ./setup.sh $flags --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos 2>/dev/null; then
    if ! ./setup.sh $flags; then
      echo "!! setup.sh rejected the flags above; its actual interface:"; ./setup.sh --help || true; exit 2
    fi
  fi
  [ "$with_assets" = "--with-assets" ] || echo "NOTE: skipped scene/object assets (rerun with --with-assets, >150GB free)."

  step "behavior env DONE. Next (pilot):"
  echo "  1) grab 1-2 raw HDF5 demos:  HF_ENDPOINT=$HF_ENDPOINT huggingface-cli download behavior-robot-suite/... (see challenge dataset page)"
  echo "  2) replay with seg_instance: OmniGibson/scripts/learning/replay_obs.py + robot_sensor_config modalities += seg_instance"
}

case "${1:-probe}" in
  probe) probe ;;
  v2sam) v2sam ;;
  behavior) shift; behavior "${1:-}" ;;
  *) echo "usage: bash $0 {probe|v2sam|behavior [--with-assets]}"; exit 1 ;;
esac
