set -euo pipefail
R=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/pccs_gain_20260915
P=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/context_pccs_20260914
cp "$R/gate_training/pipeline.log" "$R/gate_training/preflight_missing_sklearn.log"
cp "$R/gate_training/status.json" "$R/gate_training/preflight_missing_sklearn.json"
"$P/runtime_env/bin/python" -m pip install --quiet --disable-pip-version-check --timeout 10 --retries 1 --index-url https://pypi.tuna.tsinghua.edu.cn/simple --target "$R/gate_training/python_deps" --no-deps scikit-learn==1.5.2 joblib==1.4.2 threadpoolctl==3.5.0
PYTHONPATH="$R/gate_training/python_deps" "$P/runtime_env/bin/python" -c 'import sklearn,scipy,numpy,joblib;print(sklearn.__version__,scipy.__version__,numpy.__version__,joblib.__version__)' > "$R/gate_training/dependency_receipt.txt"
bash "$R/launch_gate_training.sh"
