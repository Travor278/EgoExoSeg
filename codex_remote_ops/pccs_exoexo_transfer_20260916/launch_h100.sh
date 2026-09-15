set -euo pipefail
R=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/pccs_exoexo_transfer_20260916
P=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/context_pccs_20260914
exec 9>"$R/.experiment.lock"
flock -n 9 || { echo 'EXOEXO_TRANSFER_ALREADY_RUNNING'; exit 2; }
"$P/runtime_env/bin/python" "$R/run_stage.py" --stage smokeH100 --gpus 0,1,2,3
"$P/runtime_env/bin/python" "$R/run_stage.py" --stage full --gpus 0,1,2,3
