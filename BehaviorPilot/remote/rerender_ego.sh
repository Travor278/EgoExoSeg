#!/usr/bin/env bash
# Re-render ONLY the ego view for every demo that already has a bundle.
#
# Ego-only is 1/9 the cost of a full re-render: the exo frames, their seg buffers and
# their poses are untouched by the gaze change, so they are reused as they are.
#
# Picks up three fixes that all land in the ego path:
#   * gaze follows the manipulandum nearest a hand and leads toward its BDDL goal,
#     instead of staring at this card's target for the whole episode
#   * that aim is clamped to a 45 degree cone around the head's true forward, so a
#     head-mounted camera can look down at what it works on but never swings back
#     through the robot's own chassis (self-occlusion 24.7% -> 16.6%, target
#     visibility unchanged at 20%)
#   * blank renders are detected and re-rendered (25% -> 0%)
#
# Waits for any running production first so the two never share the GPU.
set -u
: "${PILOT_ROOT:=$HOME/pilot_v5}"
: "${PILOT_CONDA:=/root/miniconda3}"
: "${PILOT_TMPDIR:=/root/autodl-tmp/tmp}"
export PILOT_ROOT PILOT_TMPDIR
export TMPDIR="$PILOT_TMPDIR"
export OMNIGIBSON_GPU_ID="${GPU_ID:-0}"
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/my_nvidia_icd.json
export OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES
export PILOT_FAST_REPLAY=1 PILOT_NO_WRIST=1
export TARGET_FRAMES="${TARGET_FRAMES:-120}"
export PILOT_GAZE_MAX_DEG="${PILOT_GAZE_MAX_DEG:-45}"
mkdir -p "$PILOT_TMPDIR"

source "$PILOT_CONDA"/etc/profile.d/conda.sh
conda activate behavior

echo "waiting for production to finish..."
while pgrep -f 'sim_prod_autodl_v3.sh' >/dev/null || pgrep -f 'pilot_viewercam.py' >/dev/null; do
  sleep 60
done
echo "[$(date +%H:%M:%S)] production idle, starting ego re-render"

cp "$HOME"/pilot_viewercam.py "$PILOT_ROOT"/pilot_viewercam.py

# One ego pass per BASE demo: sibling cards (<demo>__<object>) share the same frames,
# only their mask lookup differs, so re-rendering per card would repeat identical work.
demos=$(ls "$PILOT_ROOT"/bundle 2>/dev/null | sed 's/__.*//' | sort -u)
total=$(echo "$demos" | wc -w)
i=0; ok=0; skip=0; fail=0
for d in $demos; do
  i=$((i + 1))
  t=$((d / 10000))
  ep="$PILOT_ROOT/2026-challenge-rawdata/$(printf 'task-%04d' $t)/$(printf 'episode_%08d.hdf5' $d)"
  if [ ! -s "$ep" ]; then
    echo "[$i/$total] $d SKIP (episode file gone)"
    skip=$((skip + 1)); continue
  fi
  rm -rf "$PILOT_ROOT/views/$d/head_gaze"
  for attempt in 1 2 3 4 5; do
    if env PILOT_DEMO="$d" PILOT_VIEW=head_gaze timeout 3600 \
         python "$PILOT_ROOT"/pilot_viewercam.py > "$PILOT_ROOT/ego_${d}.log" 2>&1; then
      echo "[$i/$total] $d OK (attempt $attempt)"
      ok=$((ok + 1)); break
    fi
    [ $attempt -eq 5 ] && { echo "[$i/$total] $d FAILED after 5"; fail=$((fail + 1)); }
  done
done
echo "EGO-RERENDER-DONE ok=$ok skip=$skip fail=$fail of $total"
