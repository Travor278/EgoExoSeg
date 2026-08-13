#!/usr/bin/env bash
# Pull a finished AutoDL production run into its own folder, render the cards,
# and score them — kept separate from the historical bundle/ + samples/ pair.
#
# Usage: PILOT_SSH_PW='<password>' bash BehaviorPilot/pull_autodl_run.sh <port> <host>
#   e.g. PILOT_SSH_PW='<password>' bash BehaviorPilot/pull_autodl_run.sh 40418 connect.bjb1.seetacloud.com
#
# The password comes from the environment, never from argv: arguments are visible
# to every other process via /proc/<pid>/cmdline and land in shell history.
#
# Run from the repo root.
set -euo pipefail

PORT="${1:?usage: PILOT_SSH_PW=... pull_autodl_run.sh <port> <host>}"
HOST="${2:?}"
PW="${PILOT_SSH_PW:?set PILOT_SSH_PW in the environment}"
HOSTKEY="SHA256:y2DhrdHAWfTCoY9uLz0ve3EzTKYtWNBBWm80QUTOkXg"   # AutoDL gateway key
OUT="BehaviorPilot/autodl_run40"

mkdir -p "$OUT/bundle" "$OUT/cards"

echo "== packing bundles on the remote =="
# The masks are uncompressed boolean .npy (~900KB each, ~36MB per demo) and gzip
# takes them down ~10x in about a second. Pulling the raw tree over pscp instead
# costs the better part of an hour for a full run; the tarball costs minutes.
printf 'y\n' | /d/Scoop/shims/plink -ssh -P "$PORT" -batch -hostkey "$HOSTKEY" -pw "$PW" \
  "root@$HOST" "rm -f /root/autodl-tmp/bundle.tgz && tar czf /root/autodl-tmp/bundle.tgz -C /root/behavior_pilot bundle && du -h /root/autodl-tmp/bundle.tgz"

echo "== pulling =="
printf 'y\n' | /d/Scoop/shims/pscp -P "$PORT" -batch -hostkey "$HOSTKEY" -pw "$PW" \
  "root@$HOST:/root/autodl-tmp/bundle.tgz" "$OUT/bundle.tgz"

echo "== extracting =="
# The tarball's top-level entry is already `bundle/`, so extract straight into
# $OUT — no --strip-components, which would scatter the demo dirs into $OUT.
tar xzf "$OUT/bundle.tgz" -C "$OUT"
rm -f "$OUT/bundle.tgz"

echo "== pulling shard logs (for crash / fallback forensics) =="
# pscp takes one remote source per invocation.
for log in prod_even.out prod_odd.out; do
  printf 'y\n' | /d/Scoop/shims/pscp -P "$PORT" -batch -hostkey "$HOSTKEY" -pw "$PW" \
    "root@$HOST:/root/$log" "$OUT/$log" || true
done

echo "== rendering cards =="
PILOT_BUNDLE="$OUT/bundle" PILOT_CARDS="$OUT/cards" python BehaviorPilot/render_bundle_cards.py

echo "== quality report =="
PILOT_BUNDLE="$OUT/bundle" python BehaviorPilot/bundle_quality.py | tee "$OUT/quality.txt"

echo
echo "cards  -> $OUT/cards"
echo "report -> $OUT/quality.txt"
