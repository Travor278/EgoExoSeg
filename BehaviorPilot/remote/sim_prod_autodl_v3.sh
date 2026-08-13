#!/usr/bin/env bash
# Production shard runner v3, AutoDL edition. Same 5-column recipe and episode
# fallback logic as sim_prod_run2.sh; only the host-specific plumbing differs:
#
#   1. conda lives at /root/miniconda3 (not ~/miniforge3)
#   2. hf_fetch is single-phase — hf-mirror.com is directly reachable here, so
#      the laptop reverse-tunnel proxy from the mint2027 setup is gone
#   3. VK_ICD_FILENAMES must be exported, or Vulkan falls back to llvmpipe and
#      every render segfaults. This is NOT inherited reliably from ~/.bashrc by
#      a nohup'd non-interactive shell, so it is set explicitly here.
#
# 5 columns per demo: 2 static exo + head_gaze + both wrists.
# Episode fallback: moving-camera (tracking) views hard-segfault on certain
# episodes' "poison frames" — deterministic, uncatchable, retry-proof; static
# exo views are unaffected. When tracking views crash, fall back to the NEXT
# candidate episode of the SAME task (prod_plan.txt lists several ids per line,
# primary first). First candidate whose tracking views survive becomes the card;
# if every candidate fails, the last is bundled anyway (2-exo partial > no card).
#
# Usage: SHARD=even|odd [MIN_TRACK=3] bash sim_prod_autodl_v3.sh
# Markers: PROD <demo> ... lines, PROD-SHARD-DONE.
set -uo pipefail
# Host-specific paths are env-overridable so this file can be deployed verbatim
# everywhere. They used to be patched in with sed after every scp, which lost the
# patch on the next deploy -- twice that silently reverted the plan filename and
# the runners exited on startup with nothing in the log to say why.
source "${PILOT_CONDA:-/root/miniconda3}"/etc/profile.d/conda.sh
conda activate behavior
export OMNIGIBSON_HEADLESS=1
export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/my_nvidia_icd.json
export TMPDIR="${PILOT_TMPDIR:-/root/autodl-tmp/tmp}"
# Jump straight to the strided frames instead of stepping every recorded step.
# playback_episode restores a recorded state per step rather than re-simulating,
# and only every STRIDE-th frame is rendered, so the skipped steps bought nothing.
# Measured on 41650 (10282 steps): 83 min -> 4.6 min per view, and demo 270's
# head_gaze reproduces the sequential path's visible_frac exactly (88% vs 88%).
# It does NOT prevent the OG#2312 poison-frame segfaults — it makes them cheap.
export PILOT_FAST_REPLAY="${PILOT_FAST_REPLAY:-1}"
SHARD="${SHARD:-even}"
# tracking views that must survive (max 3) to accept an episode; fewer => try the
# next candidate. 3 = insist on a full 5-column card; loosen to 2 if too many
# tasks have no fully-clean episode among their candidates.
MIN_TRACK="${MIN_TRACK:-3}"
# Everything this run produces lives under PILOT_ROOT.
export PILOT_ROOT="${PILOT_ROOT:-$HOME/behavior_pilot}"
PR=$(eval echo "$PILOT_ROOT")
mkdir -p "$PR"/{views,bundle,2026-challenge-rawdata}
source ~/hf_fetch.sh
cp ~/pilot_viewercam.py "$PR"/pilot_viewercam.py

# Every Isaac Sim launch leaves a ~600MB directory behind in TMPDIR and never
# reclaims it. Six views per demo across dozens of demos filled a 150GB disk and
# turned the rest of the run into 273 silent "FAILED-download" lines. Reap
# anything older than an hour before each demo — long enough that a live process's
# own temp dir is never touched.
reap_tmp() {
  # -mindepth 1 or find returns the starting point itself, and once TMPDIR's own
  # mtime passed 60 minutes this deleted the directory it was meant to clean. Every
  # later render then died on a missing temp dir, announced only by a stray
  # "df: /home/dell/tmp: No such file or directory" that looked like harmless noise.
  mkdir -p "$TMPDIR"
  find "$TMPDIR" -mindepth 1 -maxdepth 1 -mmin +60 -exec rm -rf {} + 2>/dev/null
  local free
  free=$(df -BG --output=avail "$TMPDIR" | tail -1 | tr -dc '0-9')
  [ "${free:-99}" -lt 10 ] && echo "PROD WARNING: only ${free}G free on TMPDIR"
  return 0
}

pass() {  # pass <demo> <view> [extra env...]
  local d=$1 v=$2; shift 2
  rm -f "$PR"/views/$d/$v/seg.h5 "$PR"/views/$d/$v/seg.npz
  env "$@" PILOT_DEMO=$d PILOT_VIEW=$v timeout 7200 python "$PR"/pilot_viewercam.py \
    > "$PR"/pass_${d}_${v}.log 2>&1
}

# render_demo <demo> — renders all 5 views; sets globals exo_ok, track_ok, good.
render_demo() {
  local demo=$1 v attempt
  exo_ok=0; track_ok=0; good=0
  # Two exo columns from two sweep angles. exo0 runs the visibility sweep AND
  # writes a fixed-ring poses file BEFORE it, so exo1/exo2 still render a usable
  # ring if the sweep's og.sim.render() segfaults. Take the first TWO that survive.
  # Render the EGO view FIRST. It is now the only moving view, so its death fails
  # the whole demo (track_ok < MIN_TRACK) — and doing it last meant discovering
  # that only after 8 successful exo passes had already been paid for (930950:
  # exo=7 track=0, ~3 GPU-hours thrown away). Ego is also the cheapest signal that
  # this episode is renderable at all.
  # Tracking views: retry each once. A one-off GPU-state flake recovers on retry;
  # a real poison frame kills all three on both attempts -> track_ok stays low ->
  # the caller falls back to the next candidate episode.
  # v4: wrists dropped. They were the dominant crash source (they drove nearly
  # every episode-fallback, at 3 full re-renders each) and even when they survived
  # the target was usually out of frame, because a wrist camera only sees the
  # object during the grasp itself. The ego view carries the same close-up
  # information and has been reliable (46-89% coverage). Several exo angles plus
  # ego is both cheaper and more of what the card actually needs.
  local tviews="head_gaze left_wrist right_wrist"
  [ "${PILOT_NO_WRIST:-0}" = "1" ] && tviews="head_gaze"
  # Retry the ego view hard. The annotator segfault is PROBABILISTIC, not the
  # deterministic poison frame this script originally assumed: the same view of
  # the same demo died after 150 reads once and after 50 the next time. Measured
  # rate is ~1%/read with seg_semantic, so a 120-frame pass completes ~30% of the
  # time and five attempts take that to ~83%. Each ego attempt is 2-5 min, against
  # ~30 min to re-render a whole candidate episode, so retrying is far cheaper
  # than the episode fallback it replaces.
  for v in $tviews; do
    for attempt in $(seq 1 "${PILOT_EGO_TRIES:-5}"); do
      if pass $demo $v; then good=$((good+1)); track_ok=$((track_ok+1)); break; fi
    done
  done
  if [ $track_ok -lt "${MIN_TRACK:-1}" ]; then
    echo "PROD $demo EGO-FAILED — skipping the exo ring for this candidate"
    return 0
  fi
  # RING mode (PILOT_EXO_RING=N): render the WHOLE ring and let the card pick the
  # two best by measured pixels. Stopping at the first two survivors, as the old
  # loop did, throws away exactly the choice we are trying to make. Without the
  # env var this collapses to the previous behaviour.
  local ring=${PILOT_EXO_RING:-0} vlist
  if [ "$ring" -gt 0 ]; then
    vlist=$(seq -f 'exo%g' 0 $((ring - 1)))
    # Retry each exo too. Exos used to be treated as reliable because on the
    # single-GPU boxes they are (mint2027: 0 crashes in 110 renders). The rented
    # 2-GPU box runs an older driver point release and segfaults inside the settle
    # renders on ~half of all passes, and an exo that died was simply dropped --
    # 1550 lost exo0 through exo5 in one sweep and bundled a card with a single
    # usable angle. A retry costs 3 minutes; a lost angle costs the card.
    for v in $vlist; do
      for attempt in $(seq 1 "${PILOT_EXO_TRIES:-2}"); do
        if pass $demo $v; then good=$((good+1)); exo_ok=$((exo_ok+1)); break; fi
      done
    done
  else
    for v in exo0 exo1 exo2; do
      if pass $demo $v; then good=$((good+1)); exo_ok=$((exo_ok+1)); [ $exo_ok -ge 2 ] && break; fi
    done
  fi
}

n=0
while read -r cand_line; do
  [ -z "$cand_line" ] && continue
  set -- $cand_line                    # candidates for ONE task, primary first
  primary=$1; t=$((primary / 10000))
  # N-way sharding by task id. SHARD=even/odd still works (mod 2); four GPUs
  # across three machines need mod 4, and the same file has to run everywhere, so
  # the split is expressed as SHARD_MOD/SHARD_IDX with the old names as aliases.
  # SHARD_MOD wins when set. Matching on $SHARD first was wrong: SHARD defaults to
  # "even" further up, so the case always took that branch and every shard, on every
  # machine, walked the same half of the plan.
  if [ -n "${SHARD_MOD:-}" ]; then
    smod=$SHARD_MOD; sidx=${SHARD_IDX:-0}
  else
    case "$SHARD" in
      odd) smod=2; sidx=1 ;;
      *)   smod=2; sidx=0 ;;
    esac
  fi
  if [ $((t % smod)) -ne "$sidx" ]; then continue; fi
  # already bundled under any candidate? skip the whole task.
  skip=0
  for demo in "$@"; do
    [ -d "$PR"/bundle/$demo ] && { echo "PROD $demo cached"; skip=1; break; }
  done
  [ $skip -eq 1 ] && continue

  cands=("$@"); ncand=${#cands[@]}
  for ((ci=0; ci<ncand; ci++)); do
    demo=${cands[$ci]}
    reap_tmp
    tdir=$(printf "task-%04d" $t); f=$(printf "episode_%08d.hdf5" $demo)
    mkdir -p "$PR"/2026-challenge-rawdata/$tdir
    if ! hf_fetch "$tdir/$f" "$PR"/2026-challenge-rawdata/$tdir/$f; then
      echo "PROD $demo FAILED-download (cand $((ci+1))/$ncand)"; continue
    fi
    echo "PROD $demo downloaded $(du -m "$PR"/2026-challenge-rawdata/$tdir/$f | cut -f1)MB (cand $((ci+1))/$ncand)"

    render_demo $demo
    last=$([ $ci -eq $((ncand - 1)) ] && echo 1 || echo 0)

    if [ $track_ok -ge $MIN_TRACK ] && [ $exo_ok -ge 2 ]; then
      echo "PROD $demo COMPLETE exo=$exo_ok track=$track_ok — accepting"
    elif [ $last -eq 1 ] && [ $track_ok -ge 1 ]; then
      echo "PROD $demo LAST-CANDIDATE exo=$exo_ok track=$track_ok — bundling partial"
    elif [ $last -eq 1 ]; then
      # Every moving view died on every candidate. Two static exos and nothing else
      # is not a manipulation card, and bundling it only adds a row to quality.txt
      # that has to be filtered out later. Skip the task instead.
      echo "PROD $demo GIVE-UP exo=$exo_ok track=0 — no moving view on any candidate"
      rm -rf "$PR"/views/$demo
      break
    else
      echo "PROD $demo POISON exo=$exo_ok track=$track_ok — trying next episode"
      rm -rf "$PR"/views/$demo   # drop this episode's views before next try
      continue
    fi

    # pass $demo: both shards share ~/behavior_pilot on this box, so bundling
    # must be scoped or each shard would bundle the other's half-written views.
    python ~/make_card_bundle.py $demo > ~/bundle_prod_${SHARD}.log 2>&1 || true
    if [ -d "$PR"/bundle/$demo ]; then
      # PILOT_KEEP_VIEWS=1 preserves views/<demo>/<view>/{rgb.mp4,seg.h5}: every
      # camera, every rendered frame, and the full instance-id buffer. The bundle
      # only ever keeps 8 frames of one object, so deleting these threw away ~100x
      # the annotation and could only be recovered by re-rendering.
      [ "${PILOT_KEEP_VIEWS:-0}" = "1" ] || rm -rf "$PR"/views/$demo
      echo "PROD $demo ok_views=$good bundled (episode $((ci+1))/$ncand of task $t)"
    else
      echo "PROD $demo NO-BUNDLE ok_views=$good"
    fi
    n=$((n+1))
    break
  done
done < "$PR"/"${PILOT_PLAN:-prod_plan_100.txt}"
echo "PROD-SHARD-DONE shard=$SHARD demos=$n"
