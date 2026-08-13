#!/usr/bin/env bash
# AutoDL variant of hf_fetch: single-phase, no proxy.
#
# The mint2027 version needed two phases because the SNI block there meant only a
# laptop reverse-tunnel could resolve the repo path into a signed CDN URL. On
# AutoDL, hf-mirror.com is directly reachable and curl -L follows the redirect to
# cas-bridge.xethub.hf.co on its own (~1.3MB/s measured), so the proxy hop is gone.
#
# Usage: hf_fetch <repo_relative_path> <output_file>
#   e.g. hf_fetch task-0094/episode_00942050.hdf5 /root/behavior_pilot/ep.hdf5

HF_BASE="https://hf-mirror.com/datasets/behavior-1k/2026-challenge-rawdata/resolve/main"

hf_fetch() {
  local rel=$1 out=$2

  # already complete? (nonzero size, and .part siblings are never named like this)
  [ -s "$out" ] && return 0

  for attempt in 1 2 3; do
    # Download to a private temp name and rename only on success: a half-written
    # file under the final name would be read by the next pass as "already done".
    local tmp="$out.$$.part"
    if curl -sfL -C - -m 2400 -o "$tmp" "$HF_BASE/$rel" 2>/dev/null && [ -s "$tmp" ]; then
      mv -f "$tmp" "$out"
      return 0
    fi
    rm -f "$tmp"
    echo "  hf_fetch: download failed (attempt $attempt) -> $rel" >&2
    sleep 15
  done
  return 1
}
