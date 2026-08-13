#!/usr/bin/env bash
# v5 run: exo ring + ego, every camera and every frame preserved, one card per
# manipulated object. Writes to its own tree so nothing mixes with earlier runs.
#
# Usage: SHARD_MOD=4 SHARD_IDX=0 GPU_ID=0 bash start_v5.sh
export PILOT_ROOT="$HOME/pilot_v5"
# Pick the GPU through OmniGibson, NOT through CUDA_VISIBLE_DEVICES. gm.GPU_ID is
# the only thing that reaches Isaac Sim's active_gpu/physics_gpu; left unset, Isaac
# enumerates devices itself while CUDA_VISIBLE_DEVICES has already renumbered them,
# and the two disagree. On a 2-GPU box that made og.sim.render() segfault on 72% of
# renders (vs 0% on the single-GPU machines, where there is nothing to disagree
# about) -- the failure OmniGibson's own comment above this config warns about.
export OMNIGIBSON_GPU_ID="${GPU_ID:-0}"
unset CUDA_VISIBLE_DEVICES
export PILOT_EXO_RING="${PILOT_EXO_RING:-8}"   # 8 angles, card keeps the best two
export PILOT_NO_WRIST=1                        # wrists dropped: crash-prone, rarely on target
export PILOT_KEEP_VIEWS=1                      # keep rgb.mp4 + seg.h5 for every view
export PILOT_FAST_REPLAY=1
export MIN_TRACK=1                             # head_gaze is the only moving view now
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/my_nvidia_icd.json
mkdir -p "$PILOT_ROOT"/{views,bundle,2026-challenge-rawdata}
cp ~/prod_plan_100.txt "$PILOT_ROOT"/prod_plan_100.txt 2>/dev/null
TAG="${SHARD_IDX:-0}"
nohup bash ~/sim_prod_autodl_v3.sh > "$HOME/prod_v5_${TAG}.out" 2>&1 &
sleep 2
echo "v5 shard ${SHARD_IDX:-0}/${SHARD_MOD:-1} started -> $PILOT_ROOT"
