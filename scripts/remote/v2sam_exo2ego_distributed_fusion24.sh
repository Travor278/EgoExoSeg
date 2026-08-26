#!/usr/bin/env bash
set -euo pipefail

BASE=/inspire/ssd/project/luojianlan/zhubingwen-253108120125
HDD=/inspire/hdd/project/luojianlan/zhubingwen-253108120125
REPRO="$BASE/v2sam-repro"
SOURCE="$REPRO/V2sam/projects/v2sam_fusion/configs/v2sam.py"
OVERLAY="$BASE/v2sam-fusion-fullmetric-24ae5a-overlay-v1"
KNOWN_MM="$REPRO/V2sam/mmengine"
PY="$REPRO/.venv/bin/python"
TORCHRUN="$REPRO/.venv/bin/torchrun"

EXP="$HDD/V2SAM_Exo2Ego_20260825"
JSON_TAR="$HDD/V2SAM_Ego2Exo_202608/assets/offline_assets/json.tar.gz"
TRAIN_JSON="$REPRO/train_inputs/egoexo_full_json/Exo2Ego_FullTrain.json"
TEST_JSON="$EXP/datasets/egoexo_json/exo2ego_test_framelevel.json"
TRAIN_IMAGES="$BASE/datasets/.Ego-Exo4D-Relation-Train-Mini.extract.incomplete.EYjUlFWH"
TEST_IMAGES="$BASE/datasets/Ego-Exo4D-Relation-Test/work/yuqian_fu/Ego/data_segswap_test"
SAM2_DIR="$HDD/v2sam_offline_assets"
DINO_REPO="$OVERLAY/third_parts/dinov3"
DINO_WEIGHTS="$HDD/v2sam_offline_assets/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

RUN="$EXP/runs/exo2ego-fusion-official24-seed530358027-v1"
CFG="$EXP/frozen_configs/exo2ego_fusion_official24_seed530358027_v1.py"
AUDIT="$EXP/audits/exo2ego_fusion_official24_launch.txt"

mkdir -p "$EXP/datasets/egoexo_json" "$EXP/frozen_configs" "$EXP/runs" "$EXP/audits"

if [ ! -s "$TEST_JSON" ]; then
  tmp_json=$(mktemp "$EXP/datasets/egoexo_json/.exo2ego_test_framelevel.json.incomplete.XXXXXX")
  tar -xOf "$JSON_TAR" 'json/Ego-Exo4D-Seg/frame-level/exo2ego_test_framelevel.json' >"$tmp_json"
  "$PY" -I - "$tmp_json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("r", encoding="utf-8") as stream:
    data = json.load(stream)
assert isinstance(data, dict) and data, type(data)
print(f"EXTRACTED_TEST_RECORDS={len(data)}")
PY
  mv "$tmp_json" "$TEST_JSON"
fi

for path in "$SOURCE" "$OVERLAY/tools/train.py" "$KNOWN_MM/mmengine/dist" \
  "$TRAIN_JSON" "$TEST_JSON" "$TRAIN_IMAGES" "$TEST_IMAGES" \
  "$SAM2_DIR/sam2_hiera_large.pt" "$DINO_REPO" "$DINO_WEIGHTS" \
  "$PY" "$TORCHRUN"; do
  [ -e "$path" ] || { printf 'FAIL missing=%s\n' "$path" >&2; exit 1; }
done

resume_path=""
if [ -f "$RUN/last_checkpoint" ]; then
  resume_path=$(head -n 1 "$RUN/last_checkpoint")
  case "$resume_path" in
    /*) ;;
    *) resume_path="$RUN/$resume_path" ;;
  esac
  [ -s "$resume_path" ] || { printf 'FAIL invalid_resume=%s\n' "$resume_path" >&2; exit 1; }
elif [ -e "$RUN" ]; then
  if find "$RUN" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    printf 'FAIL nonempty_run_without_checkpoint=%s\n' "$RUN" >&2
    exit 1
  fi
else
  mkdir -p "$RUN"
fi

tmp_cfg=$(mktemp "$EXP/frozen_configs/.exo2ego_fusion_official24.incomplete.XXXXXX")
"$PY" -I - "$SOURCE" "$tmp_cfg" "$TRAIN_JSON" "$TEST_JSON" \
  "$TRAIN_IMAGES" "$TEST_IMAGES" "$SAM2_DIR" "$DINO_REPO" \
  "$DINO_WEIGHTS" "$RUN" "$resume_path" <<'PY'
import re
import sys
from pathlib import Path

(
    source_raw,
    target_raw,
    train_json,
    test_json,
    train_images,
    test_images,
    sam2_dir,
    dino_repo,
    dino_weights,
    run_dir,
    resume_path,
) = sys.argv[1:]
text = Path(source_raw).read_text(encoding="utf-8")

def replace_once(pattern, replacement):
    global text
    text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    assert count == 1, (pattern, count)

replace_once(r"^        base_dir='weights/sam2',.*$", f"        base_dir={sam2_dir!r},")
replace_once(r"^        repo_path='third_parts/dinov3',$", f"        repo_path={dino_repo!r},")
replace_once(r"^        weights_path='weights/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd\.pth',$", f"        weights_path={dino_weights!r},")
replace_once(r"^    sam2_folder='/path/EGP/Ego-Exo4D-Seg/data_segswap',$", f"    sam2_folder={train_images!r},")
replace_once(r"^    expression_file='/path/EGP/json/Ego-Exo4D-Seg/frame-level/Exo2Ego_FullTrain\.json',$", f"    expression_file={train_json!r},")
replace_once(r"^    sam2_folder='/path/EGP/Ego-Exo4D-Seg/data_segswap_test',$", f"    sam2_folder={test_images!r},")
replace_once(r"^    expression_file='/path/EGP/json/Ego-Exo4D-Seg/frame-level/exo2ego_test_framelevel\.json',.*$", f"    expression_file={test_json!r},")
replace_once(r"^        save_optimizer=False,$", "        save_optimizer=True,\n        save_param_scheduler=True,\n        max_keep_ckpts=-1,")
replace_once(r"^randomness = dict\(seed=None, deterministic=False\)$", "randomness = dict(seed=530358027, deterministic=False)")

text += f"\nwork_dir = {run_dir!r}\n"
if resume_path:
    text += f"load_from = {resume_path!r}\nresume = True\n"
else:
    text += "load_from = None\nresume = False\n"

Path(target_raw).write_text(text, encoding="utf-8")
PY
mv "$tmp_cfg" "$CFG"

env PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$OVERLAY:$KNOWN_MM" \
  "$PY" - "$CFG" "$TRAIN_JSON" "$TEST_JSON" "$TRAIN_IMAGES" \
  "$TEST_IMAGES" "$RUN" "$resume_path" "$AUDIT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

overlay = Path(sys.path[0])
cfg_raw, train_json, test_json, train_images, test_images, run_dir, resume_path, audit_raw = sys.argv[1:]
sys.dont_write_bytecode = True

from mmengine.config import Config

cfg = Config.fromfile(cfg_raw)
train_leaf = cfg.train_dataloader.dataset.datasets[0]
val_leaf = cfg.val_dataloader.dataset.datasets[0]

assert cfg.model.type.__name__ == "V2SAM"
assert cfg.train_dataloader.batch_size == 16
assert cfg.optim_wrapper.accumulative_counts == 4
assert cfg.optim_wrapper.optimizer.lr == 4e-5
assert cfg.train_cfg.max_epochs == 24
assert cfg.train_cfg.val_interval == 2
assert len(cfg.param_scheduler) == 2
assert abs(float(cfg.param_scheduler[0].end) - 1.2) < 1e-9
assert abs(float(cfg.param_scheduler[1].begin) - 1.2) < 1e-9
assert float(cfg.param_scheduler[1].end) == 24.0
assert Path(train_leaf.expression_file) == Path(train_json)
assert Path(val_leaf.expression_file) == Path(test_json)
assert Path(train_leaf.sam2_folder) == Path(train_images)
assert Path(val_leaf.sam2_folder) == Path(test_images)
assert Path(cfg.work_dir) == Path(run_dir)
assert cfg.default_hooks.checkpoint.save_optimizer is True
assert cfg.default_hooks.checkpoint.save_param_scheduler is True
assert bool(cfg.resume) == bool(resume_path)
assert (str(cfg.load_from) if cfg.load_from else "") == resume_path

def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

with Path(train_json).open("r", encoding="utf-8") as stream:
    train_data = json.load(stream)
with Path(test_json).open("r", encoding="utf-8") as stream:
    test_data = json.load(stream)
assert isinstance(train_data, dict) and len(train_data) == 123381
assert isinstance(test_data, dict) and test_data
assert sha(train_json) == "212e7291990453b087f91dd848dd8a8a8f11a7a0a110fb257be760ca9c8c3058"

lines = [
    "CONTRACT=PASS",
    "METHOD=Fusion public V2SAM",
    "DIRECTION=Exo2Ego",
    f"CONFIG={cfg_raw}",
    f"TRAIN_RECORDS={len(train_data)}",
    f"TEST_RECORDS={len(test_data)}",
    f"TRAIN_SHA={sha(train_json)}",
    f"TEST_SHA={sha(test_json)}",
    "BATCH=16 ACCUM=4 LR=4e-5 EPOCHS=24",
    "SCHEDULER=LinearLR[0,1.2]+CosineAnnealingLR[1.2,24]",
    f"RESUME={int(bool(resume_path))} FROM={resume_path or 'NONE'}",
]
Path(audit_raw).write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export PYTHONPATH="$OVERLAY:$KNOWN_MM"
cd "$REPRO"
exec "$TORCHRUN" --nproc_per_node=4 --master_port=29657 \
  "$OVERLAY/tools/train.py" "$CFG" --launcher pytorch
