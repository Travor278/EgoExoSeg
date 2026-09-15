set -euo pipefail
R=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/pccs_pipeline_audit_20260916
P=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/context_pccs_20260914
F=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/pccs_gain_full_20260915
export PYTHONPATH="$F/python_deps:$F/code/mmengine:$F/code" LD_LIBRARY_PATH="$P/runtime_lib:${LD_LIBRARY_PATH:-}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
"$P/runtime_env/bin/python" "$R/runtime_probe.py" 2>&1 | tee "$R/runtime_probe.log"
