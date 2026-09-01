#!/usr/bin/env bash
set -euo pipefail

BASE=/inspire/ssd/project/luojianlan/zhubingwen-253108120125
HDD=/inspire/hdd/project/luojianlan/zhubingwen-253108120125
OUT_ROOT="$HDD/V2SAM_NewMetrics_20260830"
REPORT="$OUT_ROOT/audits/checkpoint_inventory.txt"

mkdir -p "$OUT_ROOT/audits" "$OUT_ROOT/author_metric" \
  "$OUT_ROOT/configs" "$OUT_ROOT/results" "$OUT_ROOT/logs"

exec >"$REPORT" 2>&1

printf 'INVENTORY_TIME=%s\n' "$(date -Is)"
printf 'HOST=%s\n' "$(hostname)"
printf 'DISK\n'
df -B1 "$HDD" "$BASE"

printf '\nRUNTIME\n'
for path in \
  "$BASE/v2sam-repro/.venv/bin/python" \
  "$BASE/v2sam-repro/.venv/bin/torchrun" \
  "$BASE/v2sam-repro/V2sam/mmengine" \
  "$BASE/v2sam-newmatcher-training-overlay-v12" \
  "$BASE/v2sam-repro/V2sam"
do
  if [ -e "$path" ]; then
    stat -c '%F %s %y %n' "$path"
  else
    printf 'MISSING %s\n' "$path"
  fi
done

printf '\nCHECKPOINTS\n'
for root in \
  "$HDD/V2SAM_Ego2Exo_202608" \
  "$HDD/V2SAM_Exo2Ego_20260825" \
  "$BASE/v2sam-repro/runs"
do
  printf 'ROOT %s\n' "$root"
  if [ -d "$root" ]; then
    find "$root" -maxdepth 7 -type f \
      \( -name 'epoch_*.pth' -o -name 'iter_*.pth' \) \
      -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS\t%p\n' | sort -k3
  else
    printf 'MISSING_ROOT\n'
  fi
done

printf '\nCONFIGS\n'
for root in \
  "$HDD/V2SAM_Ego2Exo_202608" \
  "$HDD/V2SAM_Exo2Ego_20260825" \
  "$BASE/v2sam-repro/frozen_configs"
do
  printf 'ROOT %s\n' "$root"
  if [ -d "$root" ]; then
    find "$root" -maxdepth 6 -type f -name '*.py' \
      -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS\t%p\n' | sort -k3
  else
    printf 'MISSING_ROOT\n'
  fi
done

printf '\nJSONS\n'
find "$BASE/v2sam-repro/train_inputs" -maxdepth 5 -type f \
  \( -name '*Ego2Exo*Test*.json' -o -name '*Exo2Ego*Test*.json' \) \
  -printf '%s\t%p\n' | sort -k2 || true

printf '\nREPORT=%s\n' "$REPORT"
