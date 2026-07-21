#!/usr/bin/env bash
# Repair-3: fresh sweeps (no zero-padding, jittered clones) for tasks with dead exos,
# jittered exo0 retries everywhere, then rebundle with the joint-visibility frame picker.
# Markers: REPAIR3-DONE.
set -uo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate behavior
export OMNIGIBSON_HEADLESS=1
export OMNI_KIT_ACCEPT_EULA=YES
cp ~/pilot_viewercam.py ~/behavior_pilot/pilot_viewercam.py
cp ~/make_card_bundle.py ~/make_card_bundle.py
fails=0
run_pass() {
  local d=$1 v=$2
  echo "=== R3 $d/$v ==="
  rm -f ~/behavior_pilot/views/$d/$v/seg.h5 ~/behavior_pilot/views/$d/$v/seg.npz
  if PILOT_DEMO=$d PILOT_VIEW=$v timeout 10800 python ~/behavior_pilot/pilot_viewercam.py \
       > ~/behavior_pilot/pass_${d}_${v}.log 2>&1; then
    grep -a "VIEWPASS-OK" ~/behavior_pilot/pass_${d}_${v}.log | tail -1
  else
    echo "R3-FAILED $d/$v (exit $?)"; fails=$((fails+1))
  fi
}
# fresh sweeps where old exos are dead/wall-facing
rm -f ~/behavior_pilot/views/poses_task-0007.json ~/behavior_pilot/views/poses_task-0076.json \
      ~/behavior_pilot/views/poses_task-0094.json ~/behavior_pilot/views/poses_task-0004.json
for d in 71020 761590 942050 42750; do
  for v in exo0 exo1 exo2; do run_pass $d $v; done
done
# jittered exo0 retries on cached-pose tasks
for d in 50220 720270 822240; do run_pass $d exo0; done
echo "REPAIR3-DONE fails=$fails"
rm -rf ~/behavior_pilot/bundle
python ~/make_card_bundle.py > ~/bundle.out 2>&1
tail -2 ~/bundle.out
tar -C ~/behavior_pilot -cf /tmp/bundle.tar bundle
echo TARRED
