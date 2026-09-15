set -euo pipefail
R=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/pccs_gain_full_20260915
P=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/context_pccs_20260914
"$P/runtime_env/bin/python" "$R/prepare.py" > "$R/prepare.log" 2>&1
nohup "$P/runtime_env/bin/python" "$R/run_stage.py" --stage smoke4090 --gpus 0 > "$R/smoke4090_driver.log" 2>&1 < /dev/null &
echo $! > "$R/smoke4090.pid"
