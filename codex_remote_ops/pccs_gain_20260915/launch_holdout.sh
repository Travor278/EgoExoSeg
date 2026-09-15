set -euo pipefail
R=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/pccs_gain_20260915
P=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/context_pccs_20260914
"$P/runtime_env/bin/python" "$R/prepare_holdout_run.py" > "$R/gate_training/prepare_holdout_run.log" 2>&1
nohup "$P/runtime_env/bin/python" "$R/run_holdout.py" > "$R/gate_training/holdout/pipeline.log" 2>&1 < /dev/null &
echo $! > "$R/gate_training/holdout/pipeline.pid"
