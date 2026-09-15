set -euo pipefail
R=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/pccs_gain_full_20260915
P=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/context_pccs_20260914
exec 9>"$R/.h100.lock"
flock -n 9 || { echo 'FULL_GAIN_JOB_ALREADY_RUNNING'; exit 2; }
"$P/runtime_env/bin/python" "$R/run_stage.py" --stage smoke4 --gpus 0,1,2,3
"$P/runtime_env/bin/python" "$R/run_stage.py" --stage full --gpus 0,1,2,3
