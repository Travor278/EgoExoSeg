"""BEHAVIOR pilot v2 via the VIEWER camera (seg-safe pipeline, cf. OG issue #2312).

One process = one viewpoint pass over one demo:
  PILOT_VIEW in {exo0, exo1, exo2, head, left_wrist}, PILOT_DEMO=<demo id>
v2: the three static exo views form a RING aimed at the TARGET OBJECT's runtime
position (queried from the task's object scope after env creation), instead of
blind offsets from the robot start. One target object per run (PILOT_TARGET
substring override; auto = scope entry matching the task name, excl. agent/floor).

Outputs per pass under ~/behavior_pilot/views/{demo}/{view}/: rgb.mp4 + seg.npz
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/BEHAVIOR-1K/OmniGibson/scripts/learning"))

import replay_obs as R  # noqa: E402
import omnigibson as og  # noqa: E402
from omnigibson.macros import gm  # noqa: E402
from omnigibson.envs import HDF5PlaybackWrapper  # noqa: E402
from omnigibson.eval.utils.eval_utils import PROPRIOCEPTION_INDICES  # noqa: E402

VIEW = os.environ.get("PILOT_VIEW", "exo0")
DEMO_ID = int(os.environ.get("PILOT_DEMO", "1550"))
TARGET_HINT = os.environ.get("PILOT_TARGET", "")
W, H = 640, 480

gm.ENABLE_TRANSITION_RULES = False  # vanilla replay sets this; without it env init segfaults
gm.RENDER_VIEWER_CAMERA = True
gm.DEFAULT_VIEWER_WIDTH = W
gm.DEFAULT_VIEWER_HEIGHT = H

task_id = DEMO_ID // 10000
task_name = R._get_task_name_from_task_id(task_id)
scene_model = R._load_challenge_available_tasks()[task_name][0]["scene_model"]
data_folder = os.path.expanduser("~/behavior_pilot")
out_dir = os.path.join(data_folder, "views", str(DEMO_ID), VIEW)
os.makedirs(out_dir, exist_ok=True)

env = HDF5PlaybackWrapper.create_from_hdf5(
    input_path=f"{data_folder}/2026-challenge-rawdata/task-{task_id:04d}/episode_{DEMO_ID:08d}.hdf5",
    output_path=os.path.join(out_dir, "unused.hdf5"),
    full_scene_file=R._find_full_scene_file(task_name=task_name, scene_model=scene_model),
    load_room_instances=R._load_room_instances(task_name=task_name),
    robot_sensor_config={
        "VisionSensor": {"sensor_kwargs": {"image_height": 128, "image_width": 128}},
        "zed_link:Camera:0": {"sensor_kwargs": {"horizontal_aperture": 40.0, "image_height": 128, "image_width": 128}},
    },
    n_render_iterations=1,
    flush_every_n_steps=0,
    flush_every_n_traj=1,
    include_robot_control=False,
    robot_proprio_keys=list(PROPRIOCEPTION_INDICES["R1Pro"].keys()),
    robot_obs_modalities=["proprio", "rgb", "depth_linear"],
    include_contacts=False,
)
print("MARK-ENV-OK", flush=True)

base_env = env.env if hasattr(env, "env") else env
scope = base_env.task.object_scope
scope_named = {k: getattr(v, "name", str(v)) for k, v in scope.items()}
print("scope:", scope_named, flush=True)

def _stem(w: str) -> str:
    return w[:-1] if w.endswith("s") and len(w) > 3 else w


task_words = {_stem(w) for w in task_name.lower().replace("_", " ").split()}
cands = sorted(k for k in scope if "agent" not in k.lower() and "floor" not in k.lower())
target_key = None
if TARGET_HINT:
    for k in cands:
        if TARGET_HINT.lower() in k.lower():
            target_key = k
            break
if target_key is None:
    matched = [k for k in cands if _stem(k.lower().split(".")[0].split("_")[0]) in task_words]
    target_key = (matched or cands)[0]
target_obj = scope[target_key]
target_name = getattr(target_obj, "name", str(target_obj))
tpos = np.asarray(target_obj.get_position_orientation()[0], float)
print(f"TARGET scope={target_key} name={target_name} pos={tpos.round(3).tolist()}", flush=True)

viewer = og.sim.viewer_camera
# v3.2: use seg_instance_id (pixel -> prim path) instead of seg_instance — the
# instance-segmentation reduction node is what segfaults on unresolved ids
# (OG#2312); the instance-ID annotator is a different pipeline, and prim paths
# contain the object name so downstream name-matching is unchanged.
SEG_KEY = "seg_instance_id"
for mod in ("rgb", "seg_semantic", SEG_KEY):
    try:
        viewer.add_modality(mod)
    except Exception as e:
        print(f"add_modality({mod}) -> {type(e).__name__}: {e}")
print("viewer modalities:", viewer.modalities, flush=True)

robot = base_env.robots[0]


def _lookat_quat(eye, target):
    z = np.asarray(eye, float) - np.asarray(target, float)  # USD camera looks along -Z
    z = z / np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    from scipy.spatial.transform import Rotation as Rot

    return Rot.from_matrix(np.stack([x, y, z], axis=1)).as_quat()


track_sensor = None
gaze_at_target = VIEW == "head_gaze"
cam_pose_note = ""
if VIEW in ("head", "head_gaze", "left_wrist", "right_wrist"):
    want = ("zed_link" if VIEW.startswith("head")
            else "left_realsense_link" if VIEW == "left_wrist" else "right_realsense_link")
    for k, s in robot.sensors.items():
        if want in k:
            track_sensor = s
            break
    assert track_sensor is not None, f"no sensor matching {want}"
    # v3.1: inherit the REAL robot camera optics so the view's FOV matches the
    # actual sensor. WRIST cams only: copying onto the viewer while matching the
    # zed head camera segfaults the render pipeline (wrist realsense copies fine).
    if not VIEW.startswith("head"):
        for attr in ("focal_length", "horizontal_aperture"):
            try:
                setattr(viewer, attr, getattr(track_sensor, attr))
                print(f"viewer.{attr} <- {getattr(track_sensor, attr)}", flush=True)
            except Exception as e:
                print(f"intrinsics copy {attr} failed: {type(e).__name__}: {e}", flush=True)
else:
    # v3 auto-placement: exo0 sweeps candidates, scores target visibility at t=0,
    # picks top-3 angle-diverse poses and writes cam_poses.json; exo1/2 reuse it.
    import time

    look = tpos + np.array([0.0, 0.0, 0.15])
    # v3.1: pose cache keyed by TASK -> all episodes of a task share fixed exo rigs
    # (DROID-tripod semantics); delete the file to force a re-sweep.
    poses_file = os.path.join(data_folder, "views", f"poses_task-{task_id:04d}.json")
    idx = int(VIEW[-1])
    if idx == 0 and os.path.exists(poses_file):
        # sweep already cached for this task -> reuse and skip the crash-prone sweep
        chosen = json.load(open(poses_file))
        print("exo0: reusing cached poses", flush=True)
    elif idx == 0:
        # v3.3: score the sweep with seg_semantic (oldest, most stable annotator;
        # hammering the instance annotators with 48 rapid render+get_obs crashes
        # natively). Category token from the scope key: mousetrap.n.01_1 -> 'mousetrap'.
        cat = target_key.split(".")[0].lower()
        for _ in range(3):  # annotator warm-up
            og.sim.render()
            viewer.get_obs()
        cands = []
        for r in (1.2, 1.8, 2.4):
            for a_deg in range(0, 360, 45):
                for dz in (0.8, 1.4):
                    a = np.deg2rad(a_deg)
                    eye = tpos + np.array([r * np.cos(a), r * np.sin(a), dz])
                    viewer.set_position_orientation(eye, _lookat_quat(eye, look))
                    og.sim.render()
                    obs, info = viewer.get_obs()
                    seg0 = np.asarray(obs["seg_semantic"])
                    m = info.get("seg_semantic", {}) or {}
                    ids0 = [int(kk) for kk, vv in m.items() if cat in str(vv).lower() and str(kk).isdigit()]
                    score = int(np.isin(seg0, ids0).sum()) if ids0 else 0
                    cands.append({"eye": eye.tolist(), "angle": a_deg, "score": score})
        cands.sort(key=lambda c: -c["score"])
        chosen, used_angles = [], []
        for c in cands:
            if c["score"] <= 0 and chosen:
                break
            if all(min(abs(c["angle"] - u), 360 - abs(c["angle"] - u)) >= 60 for u in used_angles):
                chosen.append(c)
                used_angles.append(c["angle"])
            if len(chosen) == 3:
                break
        # NO zero-score padding (dead wall views); jitter the best positive pose instead
        jit = 25.0
        while len(chosen) < 3 and chosen:
            base_c = chosen[0]
            a = np.deg2rad(base_c["angle"] + jit)
            r = float(np.linalg.norm(np.asarray(base_c["eye"][:2]) - tpos[:2])) * 1.1
            eye = tpos + np.array([r * np.cos(a), r * np.sin(a), base_c["eye"][2] - tpos[2] + 0.15])
            chosen.append({"eye": eye.tolist(), "angle": base_c["angle"] + jit, "score": base_c["score"] // 2})
            jit = -jit if jit > 0 else (-jit + 25.0)
        with open(poses_file, "w") as f:
            json.dump(chosen, f)
        print("placement sweep:", [(c["angle"], c["score"]) for c in chosen], flush=True)
    else:
        for _ in range(240):  # wait for exo0's sweep result
            if os.path.exists(poses_file):
                break
            time.sleep(5)
        if os.path.exists(poses_file):
            chosen = json.load(open(poses_file))
        else:  # exo0 sweep unavailable -> fall back to a fixed ring
            chosen = []
            for a_deg in (15.0, 135.0, 255.0):
                a = np.deg2rad(a_deg)
                chosen.append({"eye": (tpos + np.array([1.6 * np.cos(a), 1.6 * np.sin(a), 1.1])).tolist(),
                               "angle": a_deg, "score": -1})
            print("poses file missing; fixed-ring fallback", flush=True)
    my = chosen[min(idx, len(chosen) - 1)]
    eye = np.asarray(my["eye"], float)
    if idx == 0:
        # top-scoring pose deterministically crashes playback on several demos
        # ("closest to target" curse); a 6-degree twist breaks the trigger
        v2 = eye[:2] - tpos[:2]
        a6 = np.deg2rad(6.0)
        rot = np.array([[np.cos(a6), -np.sin(a6)], [np.sin(a6), np.cos(a6)]])
        eye = np.concatenate([tpos[:2] + rot @ v2, eye[2:]])
    viewer.set_position_orientation(eye, _lookat_quat(eye, look))
    cam_pose_note = json.dumps({"eye": eye.tolist(), "look": look.tolist(), "score0": my.get("score")})
    print("viewer auto-parked:", cam_pose_note, flush=True)

# ---- streaming outputs (OOM fix: no in-RAM frame buffers) ----
import av  # noqa: E402
import h5py  # noqa: E402

episode_id = sorted(int(k.split("_")[-1]) for k in env.input_hdf5["data"].keys())[-1]
n_samples = int(env.input_hdf5["data"][f"demo_{episode_id}"].attrs["num_samples"])
print(f"episode {episode_id}: {n_samples} samples", flush=True)

container = av.open(os.path.join(out_dir, "rgb.mp4"), "w")
h5f = h5py.File(os.path.join(out_dir, "seg.h5"), "w")
seg_maps: dict = {}
# gm viewer-size overrides don't take effect (viewer renders at its native size,
# e.g. 720x1280) -> create outputs LAZILY from the first frame's actual shape
state = {"t": 0, "ds": None, "stream": None}


def capture():
    if track_sensor is not None:
        pos, orn = track_sensor.get_position_orientation()
        if gaze_at_target:  # idealized ego: head position, gaze locked on the (moving) target
            tp = np.asarray(target_obj.get_position_orientation()[0], float)
            orn = _lookat_quat(np.asarray(pos, float), tp)
        viewer.set_position_orientation(pos, orn)
    og.sim.render()
    obs, info = viewer.get_obs()
    seg = np.asarray(obs[SEG_KEY]).astype(np.uint32)
    rgb = np.ascontiguousarray(np.asarray(obs["rgb"])[..., :3].astype(np.uint8))
    t = state["t"]
    if state["ds"] is None:
        h, w = seg.shape
        print(f"first frame: viewer renders at {h}x{w}", flush=True)
        state["ds"] = h5f.create_dataset(
            "seg", shape=(n_samples + 8, h, w), maxshape=(None, h, w),
            dtype="uint32", chunks=(1, h, w), compression="gzip", compression_opts=4)
        s = container.add_stream("h264", rate=30)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        state["stream"] = s
    seg_ds, stream = state["ds"], state["stream"]
    if t >= seg_ds.shape[0]:
        seg_ds.resize((t + 64,) + seg_ds.shape[1:])
    seg_ds[t] = seg
    for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
        container.mux(packet)
    m = info.get(SEG_KEY, {})
    if isinstance(m, dict):
        for k, v in m.items():
            seg_maps[str(k)] = str(v)
    state["t"] = t + 1


env.playback_episode(episode_id=episode_id, record_data=False, post_state_update_callback=capture)
T = state["t"]
print(f"captured {T} frames; seg ids seen: {len(seg_maps)}", flush=True)

for packet in state["stream"].encode():
    container.mux(packet)
container.close()
state["ds"].resize((T,) + state["ds"].shape[1:])
h5f.attrs["id_map"] = json.dumps(seg_maps)
h5f.attrs["task_objects"] = json.dumps(scope_named)
h5f.attrs["target"] = json.dumps(
    {"scope": target_key, "name": target_name, "pos": tpos.tolist(), "cam": cam_pose_note})
h5f.close()
print("VIEWPASS-OK", VIEW, DEMO_ID, T, flush=True)
og.shutdown()
