#!/usr/bin/env bash
# v3.3 completion batch: bring EVERY working demo up to the 7-view flagship standard
# (3 auto exo re-swept under seg_instance_id + real head + head_gaze + FOV left/right wrist)
# and add sibling tasks 94 (office) / 76 (hotel) to reach all 7 scene models.
# Markers: COMPLETE7-DONE / per-pass VIEWPASS-OK / C7-FAILED lines.
set -uo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate behavior
export OMNIGIBSON_HEADLESS=1
export OMNI_KIT_ACCEPT_EULA=YES
export HF_ENDPOINT=https://hf-mirror.com
cp ~/pilot_viewercam.py ~/behavior_pilot/pilot_viewercam.py
fails=0
run_pass() {
  local d=$1 v=$2
  echo "=== C7 $d/$v ==="
  rm -f ~/behavior_pilot/views/$d/$v/seg.h5 ~/behavior_pilot/views/$d/$v/seg.npz
  if PILOT_DEMO=$d PILOT_VIEW=$v timeout 10800 python ~/behavior_pilot/pilot_viewercam.py \
       > ~/behavior_pilot/pass_${d}_${v}.log 2>&1; then
    grep -a "VIEWPASS-OK" ~/behavior_pilot/pass_${d}_${v}.log | tail -1
  else
    echo "C7-FAILED $d/$v (exit $?)"
    fails=$((fails+1))
    return 1
  fi
}
fetch() {
  local d=$1
  local t=$((d / 10000)); local tdir=$(printf "task-%04d" $t); local f=$(printf "episode_%08d.hdf5" $d)
  mkdir -p ~/behavior_pilot/2026-challenge-rawdata/$tdir
  [ -s ~/behavior_pilot/2026-challenge-rawdata/$tdir/$f ] || curl -sfL -C - -m 2400 \
    -o ~/behavior_pilot/2026-challenge-rawdata/$tdir/$f \
    "https://hf-mirror.com/datasets/behavior-1k/2026-challenge-rawdata/resolve/main/$tdir/$f"
}
# force fresh seg_instance_id sweeps for the tasks whose old exos stared at walls
rm -f ~/behavior_pilot/views/poses_task-0005.json ~/behavior_pilot/views/poses_task-0007.json \
      ~/behavior_pilot/views/poses_task-0072.json ~/behavior_pilot/views/poses_task-0082.json \
      ~/behavior_pilot/views/poses_task-0094.json ~/behavior_pilot/views/poses_task-0076.json

# complete the 4 existing demos to 7 views (1550 already complete; its exos stay)
for d in 50220 71020 720270 822240; do
  for v in exo0 exo1 exo2 head_gaze left_wrist right_wrist; do run_pass $d $v || true; done
done
# sibling tasks: full 7 views each (new scenes: office_cubicles_right, hotel_suite_large)
for d in 942050 761590; do
  ok=0
  for i in 1 2 3; do fetch $d && { ok=1; break; }; sleep 30; done
  [ $ok -eq 1 ] || { echo "C7-FAILED download $d"; fails=$((fails+1)); continue; }
  for v in exo0 exo1 exo2 head head_gaze left_wrist right_wrist; do run_pass $d $v || true; done
done
echo "COMPLETE7-DONE fails=$fails"
rm -rf ~/behavior_pilot/bundle
python ~/make_card_bundle.py > ~/bundle.out 2>&1
tail -2 ~/bundle.out
