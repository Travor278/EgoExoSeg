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
import re
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/BEHAVIOR-1K/OmniGibson/scripts/learning"))

import replay_obs as R  # noqa: E402
import omnigibson as og  # noqa: E402
from omnigibson.macros import gm  # noqa: E402
from omnigibson.envs import HDF5PlaybackWrapper  # noqa: E402
from omnigibson.eval.utils.eval_utils import PROPRIOCEPTION_INDICES  # noqa: E402
from omnigibson.utils.python_utils import h5py_group_to_torch  # noqa: E402

VIEW = os.environ.get("PILOT_VIEW", "exo0")
DEMO_ID = int(os.environ.get("PILOT_DEMO", "1550"))
TARGET_HINT = os.environ.get("PILOT_TARGET", "")
# v3.10: hard cap on replayed physics steps (0 = uncapped). Long episodes are
# doubly punished: wall-clock scales with step count (the renders are already
# fixed at TARGET_FRAMES by the stride), AND the moving-camera seg pipeline
# segfaults the deeper into an episode it gets (41280 died at step 4000/9206),
# which then costs a full re-run on the next candidate episode. Capping cuts the
# CPU-side physics stepping and keeps playback out of the crash-prone tail.
MAX_STEPS = int(os.environ.get("PILOT_MAX_STEPS", "0"))
# v3.11: replay by jumping straight to the strided frames instead of stepping every
# recorded step. Opt-in until validated against the sequential path.
FAST_REPLAY = os.environ.get("PILOT_FAST_REPLAY", "0") == "1"
# Rendered frames per view, independent of episode length. Defined up here because
# PILOT_SEG_BOUNDS is expressed in these card-space indices (what the card's "step
# N" labels show), so the sweep needs it to convert them to fractions.
TARGET_FRAMES = 300


class _PlaybackDone(Exception):
    """Raised from capture() to stop playback_episode early (it has no step limit)."""
# exo sweep ring radii (metres). Default frames a whole robot+object workspace;
# override with e.g. PILOT_EXO_RADII="1.2,1.9" to re-run SMALL-object demos closer
# so a tiny target (mousetrap etc.) is more than a few pixels in the distant exo.
EXO_RADII = tuple(float(x) for x in os.environ.get("PILOT_EXO_RADII", "1.6,2.2,3.0").split(","))
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
FIXTURE = ("agent", "floor", "wall", "ceiling", "room", "driveway", "lawn")
cands = sorted(k for k in scope if not any(x in k.lower() for x in FIXTURE))
matched = [k for k in cands if _stem(k.lower().split(".")[0].split("_")[0]) in task_words]

# v3.7: choose the target from the BDDL GOAL. Recorded state only restores the
# robot, not the objects (verified: robot moves 0.28m across an episode while
# every object reads 0.00m), so motion can't be measured. But the task goal names
# exactly what is manipulated: in a literal like (inside ?pumpkin ?cabinet) the
# FIRST arg is the object being placed and the rest are containers/anchors. The
# manipulanda are the first args that never appear only as a container. This
# fixes demos silently aimed at a container (a cabinet) instead of the pumpkin.
episode_id = sorted(int(k.split("_")[-1]) for k in env.input_hdf5["data"].keys())[-1]
MOVE_PREDS = {"inside", "ontop", "under", "nextto", "onfloor", "filled", "covered",
              "overlaid", "attached", "draped", "contains", "saturated", "inside"}
manip_types = []
try:
    bddl = os.path.expanduser(
        f"~/BEHAVIOR-1K/bddl3/bddl/activity_definitions/{task_name}/problem0.bddl")
    goal_txt = open(bddl).read().split("(:goal", 1)[-1]
    subj, cont = [], set()
    for pred, a, b in re.findall(
            r"\(([a-z]+)\s+\??([a-z_]+\.n\.\d+)(?:_\d+)?\s+\??([a-z_]+\.n\.\d+)?", goal_txt):
        if pred in MOVE_PREDS:
            subj.append(a)
            if b:
                cont.add(b)
    manip_types = [t for t in dict.fromkeys(subj) if t not in cont]
    print(f"BDDL manipulanda={manip_types} containers={sorted(cont)}", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"BDDL parse failed ({e}); using name/alpha fallback", flush=True)

target_key = None
if TARGET_HINT and any(TARGET_HINT.lower() in k.lower() for k in cands):
    target_key = next(k for k in cands if TARGET_HINT.lower() in k.lower())
else:
    for t in manip_types:  # first manipulated type that has a concrete instance
        inst = sorted(k for k in cands if t in k)
        if inst:
            target_key = inst[0]
            break
    if target_key is None:
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


def _safe_get_obs(viewer):
    """One guarded viewer.get_obs(). Returns (obs, info), or (None, None) on a
    failed read.

    An empty seg buffer makes OmniGibson's vision_utils.remap blow up (th.max on
    a 0-element tensor). Sometimes that surfaces as a catchable RuntimeError and
    sometimes the Kit extension segfaults on unwind. Two hard rules keep the
    process alive:
      1. Never read until the pipeline has been flushed with plain render()s and
         the camera is pointed somewhere with geometry (see the warm-up below).
      2. Never touch the annotator again after a failed read — retrying a broken
         read is exactly what turns the catchable error into a hard segfault.
    So this reads exactly once and, on RuntimeError, bails to (None, None).
    """
    try:
        return viewer.get_obs()
    except RuntimeError:
        return None, None


track_sensor = None
# v3.9: per-segment static exo poses, filled in by the exo branch below. Empty for
# tracking views and for pose files written before v3.9.
EXO_SEGMENTS: list = []
# Normalised segment boundaries, e.g. [0.0, 0.76, 1.0]. Segments are NOT equal
# width — the cut follows where the target actually moves — so capture() must look
# the boundary up rather than dividing the episode into equal parts.
EXO_EDGES: list = []
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
        # BDDL category tokens use DOUBLE underscores to separate words
        # (can__of__soda.n.01), but OmniGibson's seg_semantic label is SINGLE
        # underscore (can_of_soda) — so the raw token never matched multi-word
        # categories and every exo pose scored 0 (empty exo column). Collapse
        # __ -> _ so the label match works. Single-word cats (pumpkin) unaffected.
        cat = target_key.split(".")[0].lower().replace("__", "_")
        # ---- peak-frame sweep --------------------------------------------------
        # WHEN we judge a static exo pose matters: early in the episode the robot is
        # far from the target, so from a target-aimed ring robot_px=0 -> every pose
        # scores 0 even when the object is plainly visible (verified on 11770). Both
        # are only co-visible mid-manipulation. But scoring across MANY frames means
        # many rapid seg reads, and OG#2312 hard-segfaults under that (36 reads killed
        # 600). So: (1) cheaply find the PEAK-manipulation frame by scanning POSITIONS
        # only (no get_obs -> can't trip OG#2312), then (2) run one 12-read sweep there
        # — the historically-safe budget. Objects DO move under load_state (the old
        # "objects read 0.00m" was an aliasing artifact), so frame jumps are real.
        _grp = env.input_hdf5["data"][f"demo_{episode_id}"]
        _tr = h5py_group_to_torch(_grp)
        # Honour the cap here too: poses and segment aims must be optimised for the
        # range that will actually be rendered, not for a tail we never replay.
        _N = int(_tr["state"].shape[0])
        if MAX_STEPS:
            _N = min(_N, MAX_STEPS)

        def _load_frame(fi):
            og.sim.load_state(_tr["state"][fi, : int(_tr["state_size"][fi])], serialized=True)
            og.sim.step()

        def _tcenter():
            """Target AABB centre — the actual geometry position. Some objects have a
            pivot/origin metres from their mesh (11770's can reads y=6.4 at the origin
            but y=4.0 at the AABB centre), so aim/segment centroids MUST use the AABB
            centre to match the frustum test below — otherwise the camera aims at empty
            space while scoring says the target is framed, and the column renders 0%."""
            try:
                lo_, hi_ = target_obj.aabb
                return (np.asarray(lo_, float) + np.asarray(hi_, float)) / 2.0
            except Exception:  # noqa: BLE001
                return np.asarray(target_obj.get_position_orientation()[0], float)

        # ---- geometric placement (design 2026-07-26) -----------------------
        # Score poses ANALYTICALLY from positions + PhysX raycasts: no renders, no
        # get_obs, so ZERO OG#2312 exposure (the sweep's crash source is deleted).
        # Over F trajectory frames x 36 candidate poses, count frames where the
        # target is in-frustum, big enough (projected radius), and un-occluded, AND
        # the robot is in frame. That count / F directly estimates the visible_frac
        # make_card_bundle measures later. Affordable because scoring never renders.
        from omnigibson.utils.sampling_utils import raytest  # noqa: E402

        try:
            _focal, _hap = float(viewer.focal_length), float(viewer.horizontal_aperture)
        except Exception:  # noqa: BLE001
            _focal, _hap = 17.0, 20.995
        _IMW, _IMH = 1280.0, 720.0  # the viewer renders 720x1280
        half_h = float(np.arctan(_hap / (2.0 * _focal)))
        half_v = float(np.arctan((_hap * _IMH / _IMW) / (2.0 * _focal)))
        f_px = _IMW * _focal / _hap  # projected px = world_size / distance * f_px

        def _basis(eye, look_pt):
            fwd = np.asarray(look_pt, float) - np.asarray(eye, float)
            fwd /= np.linalg.norm(fwd) + 1e-9
            right = np.cross(fwd, [0.0, 0.0, 1.0])
            right /= np.linalg.norm(right) + 1e-9
            return fwd, right, np.cross(right, fwd)

        def _in_frustum(eye, look_pt, world_pt):
            fwd, right, up = _basis(eye, look_pt)
            v = np.asarray(world_pt, float) - np.asarray(eye, float)
            fz = float(np.dot(v, fwd))
            if fz <= 0.1:
                return False, fz
            return (abs(np.arctan2(float(np.dot(v, right)), fz)) < half_h
                    and abs(np.arctan2(float(np.dot(v, up)), fz)) < half_v), fz

        def _proj_area_px(eye, look_pt, lo, hi):
            """Projected area (px^2) of the target's AABB, from its 8 corners.

            The circumscribed-sphere radius (|hi-lo|/2) overestimates thin and flat
            objects enormously — a pizza or a mousetrap has a long diagonal but
            almost no silhouette — so a sphere-based size gate passes targets that
            render as a few dozen pixels. Projecting the corners and taking the 2D
            bounding box tracks the real silhouette closely enough to gate on.
            """
            fwd, right, up = _basis(eye, look_pt)
            us, ws = [], []
            for sx in (lo[0], hi[0]):
                for sy in (lo[1], hi[1]):
                    for sz in (lo[2], hi[2]):
                        v = np.array([sx, sy, sz], float) - np.asarray(eye, float)
                        fz = float(np.dot(v, fwd))
                        if fz <= 0.1:
                            return 0.0
                        us.append(float(np.dot(v, right)) / fz * f_px)
                        ws.append(float(np.dot(v, up)) / fz * f_px)
            return (max(us) - min(us)) * (max(ws) - min(ws))

        # aim (camera orientation target) = target centroid over the mid-episode.
        # Cheap 3-frame estimate; scoring below uses REAL per-frame positions, so
        # target motion within the FOV is handled exactly.
        _aims = []
        for _f in (0.35, 0.55, 0.75):
            _load_frame(min(_N - 1, int(_f * _N)))
            _aims.append(_tcenter())
        aim = np.mean(_aims, axis=0)
        look = aim + np.array([0.0, 0.0, 0.15])
        print(f"geometric aim={aim.round(2).tolist()} hfov={np.degrees(2 * half_h):.0f}deg", flush=True)

        # mid-aimed fixed-ring fallback (kept if scoring finds nothing / errors)
        _ring = []
        for a_deg in (15.0, 135.0, 255.0):
            a = np.deg2rad(a_deg)
            _ring.append({"eye": (aim + np.array([1.9 * np.cos(a), 1.9 * np.sin(a), 1.2])).tolist(),
                          "angle": a_deg, "score": -1, "look": look.tolist()})
        with open(poses_file, "w") as f:
            json.dump(_ring, f)

        # ---- multi-frame sweep: MEASURE real seg pixels (not predict) -----------
        # The geometric predictor mis-estimated small/moving/occluded targets (11770's
        # can rendered 0 despite a 100% prediction). The sweep RENDERS each candidate
        # and counts actual target+robot pixels, so what it picks is what actually
        # shows — this is the v12 method that produced 270=99%, 20980=85%, 11770=64%.
        # It can OG#2312-segfault (12 poses x 3 frames = 36 seg reads, the known
        # threshold); the pre-sweep fixed-ring guard (already written) + sim_prod
        # candidate-fallback cover that. cat is underscore-normalised.
        cat = target_key.split(".")[0].lower().replace("__", "_")
        SWEEP_FIDX = [min(_N - 1, int(f * _N)) for f in (0.4, 0.6, 0.8)]
        warm_eye = aim + np.array([0.0, -2.0, 1.4])
        viewer.set_position_orientation(warm_eye, _lookat_quat(warm_eye, look))
        for _ in range(20):
            og.sim.render()

        def _score_pose(eye):
            viewer.set_position_orientation(eye, _lookat_quat(eye, look))
            for _ in range(2):
                og.sim.render()
            obs, info = _safe_get_obs(viewer)
            if obs is None:
                return None
            seg0 = np.asarray(obs["seg_semantic"])
            m = info.get("seg_semantic", {}) or {}
            obj_ids = [int(kk) for kk, vv in m.items() if cat in str(vv).lower() and str(kk).isdigit()]
            robot_ids = [int(kk) for kk, vv in m.items()
                         if ("robot" in str(vv).lower() or "agent" in str(vv).lower()) and str(kk).isdigit()]
            obj_px = int(np.isin(seg0, obj_ids).sum()) if obj_ids else 0
            robot_px = int(np.isin(seg0, robot_ids).sum()) if robot_ids else 0
            if obj_px == 0:
                return 0  # target not in frame -> useless exo
            return min(obj_px, robot_px) * 1000 + obj_px + robot_px

        # 2 radii x 6 azimuths = 12 poses; scored at 3 mid-frames = 36 seg reads.
        cand_eyes = []
        for r in (1.8, 2.6):
            for a_deg in range(0, 360, 60):
                a = np.deg2rad(a_deg)
                cand_eyes.append((a_deg, aim + np.array([r * np.cos(a), r * np.sin(a), 1.3])))
        agg = [0] * len(cand_eyes)
        seen = [False] * len(cand_eyes)
        for fi in SWEEP_FIDX:
            _load_frame(fi)
            for _ in range(3):
                og.sim.render()
            for i, (a_deg, eye) in enumerate(cand_eyes):
                s = _score_pose(eye)
                if s is not None and s > 0:
                    seen[i] = True
                    agg[i] += s
        ranked = sorted((i for i in range(len(cand_eyes)) if seen[i]), key=lambda i: -agg[i])
        chosen, used_angles = [], []
        for i in ranked:
            ang = cand_eyes[i][0]
            if all(min(abs(ang - u), 360 - abs(ang - u)) >= 60 for u in used_angles):
                chosen.append({"angle": ang, "score": agg[i],
                               "eye": cand_eyes[i][1].tolist(), "look": look.tolist()})
                used_angles.append(ang)
            if len(chosen) == 3:
                break
        if chosen:
            with open(poses_file, "w") as f:
                json.dump(chosen, f)
            print("sweep placement:", [(c["angle"], c["score"]) for c in chosen], flush=True)
        else:
            chosen = _ring  # nothing visible -> keep the mid-aimed fixed-ring
            print("sweep found nothing; keeping mid-aimed fixed-ring", flush=True)

        # restore frame 0 so the main playback starts from the beginning
        og.sim.load_state(_tr["state"][0, : int(_tr["state_size"][0])], serialized=True)
        og.sim.step()
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
    # the multi-frame sweep stores its aim ("look") so every exo view frames the same
    # manipulation centroid; older/fixed-ring poses have none -> keep the t=0 look.
    if "look" in my:
        look = np.asarray(my["look"], float)
    pivot = look[:2]

    def _twist(e, piv):
        """The top-scoring pose deterministically crashes playback on several demos
        ("closest to target" curse); a 6-degree twist breaks the trigger."""
        e = np.asarray(e, float)
        v2 = e[:2] - piv
        a6 = np.deg2rad(6.0)
        rot = np.array([[np.cos(a6), -np.sin(a6)], [np.sin(a6), np.cos(a6)]])
        return np.concatenate([piv + rot @ v2, e[2:]])

    if idx == 0:
        eye = _twist(eye, pivot)
    # v3.9: piecewise-static exo — capture() re-parks the camera at each segment
    # boundary. Older pose files have no "segments" key, so this stays a no-op and
    # the view behaves exactly as before.
    for sp in my.get("segments", []):
        e, lk = np.asarray(sp["eye"], float), np.asarray(sp["look"], float)
        EXO_SEGMENTS.append({"eye": _twist(e, lk[:2]) if idx == 0 else e, "look": lk})
    EXO_EDGES.extend(my.get("seg_edges") or
                     [i / max(1, len(EXO_SEGMENTS)) for i in range(len(EXO_SEGMENTS) + 1)])
    if EXO_SEGMENTS:
        eye, look = EXO_SEGMENTS[0]["eye"], EXO_SEGMENTS[0]["look"]
        print(f"exo piecewise: {len(EXO_SEGMENTS)} segments", flush=True)
    viewer.set_position_orientation(eye, _lookat_quat(eye, look))
    cam_pose_note = json.dumps({"eye": eye.tolist(), "look": look.tolist(), "score0": my.get("score")})
    print("viewer auto-parked:", cam_pose_note, flush=True)

# ---- streaming outputs (OOM fix: no in-RAM frame buffers) ----
import av  # noqa: E402
import h5py  # noqa: E402

# episode_id was already resolved above, when this episode's initial state was
# restored so the cameras could be aimed at where the target actually starts.
n_samples = int(env.input_hdf5["data"][f"demo_{episode_id}"].attrs["num_samples"])
print(f"episode {episode_id}: {n_samples} samples", flush=True)

# v3.8: render on a STRIDE, not every playback step. The tracking views
# (head_gaze / wrists) move the camera every frame, and moving-camera + render
# is exactly what makes OmniGibson's seg pipeline hard-segfault deep into a long
# episode (41280 died at step 4000/9206 on head_gaze while the fixed exo1 pass
# survived). Rendering ~TARGET_FRAMES evenly-spaced frames instead of all N both
# avoids that accumulation and cuts render cost ~30x on long demos. ~300 frames
# still covers the whole manipulation and leaves plenty for card sampling.
# n_play is the length we actually replay. The stride is derived from it, so a
# capped episode still yields ~TARGET_FRAMES frames — the cap changes which part
# of the episode is covered, not how many frames the card can sample from.
n_play = min(n_samples, MAX_STEPS) if MAX_STEPS else n_samples
if n_play < n_samples:
    print(f"capping playback: {n_samples} -> {n_play} steps", flush=True)
STRIDE = max(1, n_play // TARGET_FRAMES)
N_RENDER = n_play // STRIDE + 2
print(f"downsample: stride={STRIDE} -> ~{N_RENDER} render frames", flush=True)

container = av.open(os.path.join(out_dir, "rgb.mp4"), "w")
h5f = h5py.File(os.path.join(out_dir, "seg.h5"), "w")
seg_maps: dict = {}
# gm viewer-size overrides don't take effect (viewer renders at its native size,
# e.g. 720x1280) -> create outputs LAZILY from the first frame's actual shape
state = {"t": 0, "ds": None, "stream": None}


def capture():
    # Advance the raw playback counter every step, but only move-camera + render
    # on STRIDE boundaries. Skipped steps still update sim state (playback keeps
    # going); they just don't render, which is what keeps the seg pipeline alive.
    r = state.get("raw", 0)
    state["raw"] = r + 1
    if r >= n_play:
        raise _PlaybackDone  # playback_episode takes no step limit; unwind instead
    if r % STRIDE != 0:
        return
    _grab(r)


def _grab(r):
    """Move the camera for raw step @r, render, and append one frame to the outputs.
    Split out of capture() so fast replay can call it directly at the strided
    frames without going through playback_episode's per-step callback."""
    if track_sensor is not None:
        pos, orn = track_sensor.get_position_orientation()
        if gaze_at_target:  # idealized ego: head position, gaze locked on the (moving) target
            tp = np.asarray(target_obj.get_position_orientation()[0], float)
            orn = _lookat_quat(np.asarray(pos, float), tp)
        viewer.set_position_orientation(pos, orn)
    elif EXO_SEGMENTS:
        # Static within a segment, re-parked only when crossing a boundary.
        frac = r / max(1, n_play)
        s = sum(1 for e in EXO_EDGES[1:-1] if frac >= e)
        s = min(s, len(EXO_SEGMENTS) - 1)
        if s != state.get("seg"):
            state["seg"] = s
            e, lk = EXO_SEGMENTS[s]["eye"], EXO_SEGMENTS[s]["look"]
            viewer.set_position_orientation(e, _lookat_quat(e, lk))
            print(f"exo -> segment {s}/{len(EXO_SEGMENTS)} at raw step {r}", flush=True)
    og.sim.render()
    obs, info = _safe_get_obs(viewer)
    if obs is None:  # skip this one frame rather than segfaulting the whole view
        print(f"capture: frame {state['t']} unreadable, skipped", flush=True)
        return
    seg = np.asarray(obs[SEG_KEY]).astype(np.uint32)
    rgb = np.ascontiguousarray(np.asarray(obs["rgb"])[..., :3].astype(np.uint8))
    t = state["t"]
    if state["ds"] is None:
        h, w = seg.shape
        print(f"first frame: viewer renders at {h}x{w}", flush=True)
        state["ds"] = h5f.create_dataset(
            "seg", shape=(N_RENDER, h, w), maxshape=(None, h, w),
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


if FAST_REPLAY:
    # playback_episode restores a RECORDED state per step — it does not re-simulate
    # from actions. Since only every STRIDE-th frame is ever rendered, stepping the
    # other 97% is pure cost: it is the CPU-side physics stepping that dominates
    # wall-clock on long episodes, and the accumulated depth is what the moving-
    # camera seg pipeline segfaults on. So jump straight to the frames we render.
    _fgrp = env.input_hdf5["data"][f"demo_{episode_id}"]
    _ftr = h5py_group_to_torch(_fgrp)
    _fN = min(n_play, int(_ftr["state"].shape[0]))
    frames = list(range(0, _fN, STRIDE))
    print(f"fast replay: loading {len(frames)} states directly (skipping {_fN - len(frames)} steps)",
          flush=True)
    import time as _time
    _t0 = _time.time()
    for k, i in enumerate(frames):
        og.sim.load_state(_ftr["state"][i, : int(_ftr["state_size"][i])], serialized=True)
        og.sim.step()
        for _ in range(2):
            og.sim.render()  # settle the jumped-to state before reading the annotator
        state["raw"] = i
        _grab(i)
        # Progress marker: a segfault leaves both rgb.mp4 and seg.h5 unfinalised and
        # therefore unreadable, so the log is the only record of how far a crashed
        # pass actually got.
        if k % 25 == 0:
            print(f"fast replay: frame {k}/{len(frames)} (raw {i}) "
                  f"{_time.time() - _t0:.0f}s elapsed", flush=True)
else:
    try:
        env.playback_episode(episode_id=episode_id, record_data=False, post_state_update_callback=capture)
    except _PlaybackDone:
        print(f"playback stopped at the {n_play}-step cap", flush=True)
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
