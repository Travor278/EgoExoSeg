#!/usr/bin/env bash
# BEHAVIOR-1K pilot: replay one official teleop demo with seg_instance + 3 external exo cams.
# Markers: PILOT-OK / PILOT-FAILED.
set -uo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate behavior
export HF_ENDPOINT=https://hf-mirror.com
export OMNIGIBSON_HEADLESS=1
export OMNI_KIT_ACCEPT_EULA=YES
export PILOT_TASK=turning_on_radio
PILOT=~/behavior_pilot
step(){ echo -e "\n=== $* ==="; }
fail(){ echo "PILOT-FAILED: $*"; exit 2; }

step "raw episodes (hf-mirror, resumable)"
mkdir -p $PILOT/2026-challenge-rawdata/task-0000
cd $PILOT/2026-challenge-rawdata/task-0000
for e in episode_00001550 episode_00001900; do
  curl -sfL -C - -m 900 -o $e.hdf5 \
    "https://hf-mirror.com/datasets/behavior-1k/2026-challenge-rawdata/resolve/main/task-0000/$e.hdf5" \
    || fail "download $e"
done
ls -lh

step "pilot_ext.py (3 exo cams around robot start, look-at quaternions)"
cat > $PILOT/pilot_ext.py <<'PYEOF'
import os
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
from omnigibson.macros import gm


def _lookat_quat(eye, target):
    z = np.asarray(eye, float) - np.asarray(target, float)   # USD camera looks along -Z
    z = z / np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return R.from_matrix(np.stack([x, y, z], axis=1)).as_quat().tolist()  # xyzw


def build_external_sensors():
    task = os.environ.get("PILOT_TASK", "turning_on_radio")
    yml = os.path.join(gm.DATA_PATH, "2026-challenge-task-instances", "metadata", "available_tasks.yaml")
    with open(yml) as f:
        at = yaml.safe_load(f)
    base = np.asarray(at[task][0]["robot_start_position"], float)
    target = (base + np.array([0.0, 0.0, 1.0])).tolist()

    def cam(name, offset):
        eye = (base + np.asarray(offset, float)).tolist()
        return {
            "sensor_type": "VisionSensor",
            "name": name,
            "modalities": ["rgb", "seg_instance"],
            "sensor_kwargs": {"image_height": 480, "image_width": 640},
            "position": eye,
            "orientation": _lookat_quat(eye, target),
        }

    return [
        cam("exo0", [2.2, 1.6, 1.8]),
        cam("exo1", [2.2, -1.6, 1.8]),
        cam("exo2", [-1.8, 2.0, 1.9]),
    ]


EXTERNAL_SENSORS = build_external_sensors()
PYEOF

step "patch a copy of replay_obs.py"
cp ~/BEHAVIOR-1K/OmniGibson/scripts/learning/replay_obs.py $PILOT/pilot_replay.py
python - <<'PYEOF'
import os, re
p = os.path.expanduser("~/behavior_pilot/pilot_replay.py")
src = open(p).read()
old_mod = 'robot_obs_modalities=list(PROPRIOCEPTION_INDICES["R1Pro"].keys()),'
assert 'depth_linear' in src
src = src.replace(
    'robot_obs_modalities=["proprio", "rgb", "depth_linear"],',
    'robot_obs_modalities=["proprio", "rgb", "seg_instance"],', 1)
src = src.replace(
    'include_contacts=False,',
    'include_contacts=False,\n        external_sensors_config=EXTERNAL_SENSORS,', 1)
src = src.replace(
    'log = create_module_logger(module_name="replay_obs")',
    'import sys; sys.path.insert(0, os.path.expanduser("~/behavior_pilot"))\n'
    'from pilot_ext import EXTERNAL_SENSORS\n'
    'log = create_module_logger(module_name="replay_obs")', 1)
open(p, "w").write(src)
n1 = src.count("seg_instance"); n2 = src.count("EXTERNAL_SENSORS")
print("patched: seg_instance refs:", n1, "EXTERNAL_SENSORS refs:", n2)
assert n1 >= 1 and n2 >= 2
PYEOF

step "run replay (demo 1550, headless; first Isaac launch takes minutes)"
cd $PILOT
python pilot_replay.py --data_folder $PILOT --demo_id 1550 --output_format hdf5 || fail replay
ls -lh $PILOT/replayed/ || true

step "output schema peek"
python - <<'PYEOF'
import h5py, os
p = os.path.expanduser("~/behavior_pilot/replayed/episode_00001550.hdf5")
with h5py.File(p) as f:
    def walk(name, obj):
        if isinstance(obj, h5py.Dataset) and len(walk.shown) < 40:
            walk.shown.append(name)
            print(f"{name}  {obj.shape}  {obj.dtype}")
    walk.shown = []
    f.visititems(walk)
PYEOF
echo PILOT-OK
