#!/usr/bin/env bash
set -uo pipefail

# Read-only watchdog for long V2-SAM evaluation/training jobs. It records a
# compact current snapshot plus an append-only history; it never starts, stops,
# or modifies a training process.

E18_LOG=${E18_LOG:-/inspire/hdd/project/luojianlan/zhubingwen-253108120125/V2SAM_Ego2Exo_202608/jobs/pccs_wrz_fusionfirst_e10_e18_20260825_v1/logs/e18_parallel.log}
E18_MARKER=${E18_MARKER:-/inspire/hdd/project/luojianlan/zhubingwen-253108120125/V2SAM_Ego2Exo_202608/jobs/pccs_wrz_fusionfirst_e10_e18_20260825_v1/e18_parallel_done.txt}
FUSION_RUN=${FUSION_RUN:-/inspire/hdd/project/luojianlan/zhubingwen-253108120125/V2SAM_Exo2Ego_20260825/runs/exo2ego-fusion-official24-seed530358027-v1}
SUPERVISOR_DIR=${SUPERVISOR_DIR:-/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/v2sam_tail_supervisor}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-300}
MAX_CHECKS=${MAX_CHECKS:-720}
ONCE=${ONCE:-0}

mkdir -p "$SUPERVISOR_DIR"
CURRENT="$SUPERVISOR_DIR/current.txt"
HISTORY="$SUPERVISOR_DIR/history.log"

snapshot() {
    local tmp fusion_log latest_checkpoint
    tmp="$CURRENT.tmp.$$"
    fusion_log=$(find "$FUSION_RUN" -type f -name '*.log' -print 2>/dev/null | sort | tail -n 1)
    latest_checkpoint=$(find "$FUSION_RUN" -maxdepth 1 -type f \
        \( -name 'epoch_*.pth' -o -name 'iter_*.pth' \) -print 2>/dev/null | sort -V | tail -n 1)

    {
        printf 'UTC='; date -u '+%F %T'
        printf 'E18_DONE='; [ -f "$E18_MARKER" ] && echo YES || echo NO
        printf 'E18_LATEST='
        grep -E 'Iter\(test\)|Best Mean IoU|Best Mean Dice|Mean Location Score|Mean Shape Accuracy' \
            "$E18_LOG" 2>/dev/null | tail -n 1
        printf 'E18_ERRORS='
        grep -Ec 'Traceback|Error:|Exception:' "$E18_LOG" 2>/dev/null || true
        printf 'FUSION_LATEST='
        grep -E 'Iter\(train\)|Iter\(val\)|Mean IoU|Mean Dice' "$fusion_log" 2>/dev/null | tail -n 1
        printf 'FUSION_ERRORS='
        grep -Ec 'Traceback|Error:|Exception:' "$fusion_log" 2>/dev/null || true
        if [ -n "$latest_checkpoint" ] && [ -f "$latest_checkpoint" ]; then
            printf 'FUSION_CHECKPOINT='
            stat -c '%n size=%s mtime=%y' "$latest_checkpoint"
        else
            printf 'FUSION_CHECKPOINT=NONE\n'
        fi
    } > "$tmp"

    mv -f "$tmp" "$CURRENT"
    {
        printf '%s\n' '---'
        cat "$CURRENT"
    } >> "$HISTORY"

    if [ -f "$E18_MARKER" ]; then
        grep -E \
            'Best Mean IoU|Best Mean Dice|Mean Location Score|Mean Shape Accuracy|Decoder选中|平均IoU|循环一致性使用次数|平均循环距离' \
            "$E18_LOG" 2>/dev/null | tail -n 30 > "$SUPERVISOR_DIR/e18_final_metrics.txt"
    fi
}

check=0
while [ "$check" -lt "$MAX_CHECKS" ]; do
    snapshot
    check=$((check + 1))
    [ "$ONCE" = 1 ] && break
    sleep "$INTERVAL_SECONDS"
done
