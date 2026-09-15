set -euo pipefail
R=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/pccs_pipeline_audit_20260916
P=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/context_pccs_20260914
export LD_LIBRARY_PATH="$P/runtime_lib:${LD_LIBRARY_PATH:-}"
"$P/runtime_env/bin/python" "$R/metric_regression.py" > "$R/metric_regression.log" 2>&1
