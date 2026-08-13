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
# Rendered frames per view. Every frame costs one annotator read, and OG#2312
# segfaults after some number of them: with seg_instance_id demo 351240's ego view
# died at frame 50, with seg_semantic at frame 150. So this is the crash budget,
# not just a density knob. 300 was arbitrary; the card samples 8 rows and export
# pairs need far fewer, so trading density for completed passes is worth it.
TARGET_FRAMES = int(os.environ.get("PILOT_TARGET_FRAMES", "120"))


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
# PILOT_ROOT lets a run write to its own tree. Everything (episodes, views,
# bundles) hangs off it, so a new run never mixes with an old one.
data_folder = os.path.expanduser(os.environ.get("PILOT_ROOT", "~/behavior_pilot"))
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

if os.environ.get("PILOT_PROBE_ROOMS") == "1":
    # Dump what the scene knows about rooms, then stop. Lives inside this file
    # because a standalone probe segfaulted during env creation -- this setup is the
    # only path known to load a scene reliably. Every lookup is guarded: the point is
    # to find out WHICH of these APIs exist, so an AttributeError is a result, not a
    # failure, and must not take the probe down with it.
    def _p(label, fn):
        try:
            print(f"PROBE: {label} = {fn()}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"PROBE: {label} !! {type(e).__name__}: {e}", flush=True)

    # og.sim has no .scene in this build; the env owns it.
    _p("og.sim attrs", lambda: [a for a in dir(og.sim)
                                if "scene" in a.lower() and not a.startswith("_")])
    _p("base_env attrs", lambda: [a for a in dir(base_env)
                                  if "scene" in a.lower() and not a.startswith("_")])
    _sc = getattr(base_env, "scene", None) or getattr(og.sim, "scenes", [None])[0]
    _p("scene type", lambda: type(_sc).__name__)
    _p("scene attrs", lambda: {a: hasattr(_sc, a) for a in
                               ("seg_map", "room_ins_map", "objects_by_category", "aabb")})
    _p("seg_map type", lambda: type(_sc.seg_map).__name__)
    _p("seg_map members", lambda: [a for a in dir(_sc.seg_map) if not a.startswith("_")])
    _p("n objects", lambda: len(list(_sc.objects)))
    _p("categories", lambda: sorted({getattr(o, "category", "?") for o in _sc.objects})[:40])

    def _struct():
        out = []
        for o in _sc.objects:
            c = getattr(o, "category", "?")
            if any(w in c for w in ("floor", "wall")):
                try:
                    lo, hi = o.aabb
                    out.append((c, getattr(o, "name", "?"),
                                np.round(np.asarray(lo, float), 2).tolist(),
                                np.round(np.asarray(hi, float), 2).tolist()))
                except Exception:
                    out.append((c, getattr(o, "name", "?"), "aabb-failed", ""))
        return out
    try:
        for row in _struct()[:24]:
            print(f"PROBE: struct {row}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"PROBE: struct !! {type(e).__name__}: {e}", flush=True)

    _p("robot pos", lambda: np.round(
        np.asarray(base_env.robots[0].get_position_orientation()[0], float), 2).tolist())
    _p("target pos", lambda: np.round(tpos, 2).tolist() if "tpos" in dir() else "n/a")
    _p("room by point", lambda: _sc.seg_map.get_room_instance_by_point(
        np.asarray(base_env.robots[0].get_position_orientation()[0], float)[:2]))
    print("PROBE-DONE", flush=True)
    sys.exit(0)

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
goal_of: dict = {}   # manipulandum type -> destination type, from the goal literals
try:
    bddl = os.path.expanduser(
        f"~/BEHAVIOR-1K/bddl3/bddl/activity_definitions/{task_name}/problem0.bddl")
    goal_txt = open(bddl).read().split("(:goal", 1)[-1]
    subj, cont = [], set()
    # Keep WHERE each object is going, not just what moves. A goal literal
    # (inside ?pumpkin ?cabinet) names both the manipulandum and its destination,
    # and the destination is what an egocentric gaze drifts toward while carrying.
    for pred, a, b in re.findall(
            r"\(([a-z]+)\s+\??([a-z_]+\.n\.\d+)(?:_\d+)?\s+\??([a-z_]+\.n\.\d+)?", goal_txt):
        if pred in MOVE_PREDS:
            subj.append(a)
            if b:
                cont.add(b)
                goal_of.setdefault(a, b)
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
        # Fall back to the SMALLEST candidate, not the alphabetically first one.
        # Sorting by name is effectively arbitrary and biases toward appliances:
        # task 45's scope is {hotdog x2, electric_refrigerator, microwave} and
        # "electric_refrigerator" sorts first, so the card was built around the
        # fridge -- a 171k-pixel mask of the thing the hotdog goes into. A
        # manipulandum is almost always far smaller than the fixture it is placed
        # in, and the AABB is available here, so size is a much better tie-break.
        def _vol(k):
            try:
                lo, hi = scope[k].aabb
                d = np.asarray(hi, float) - np.asarray(lo, float)
                return float(np.prod(np.clip(d, 1e-3, None)))
            except Exception:  # noqa: BLE001
                return float("inf")

        pool = matched or cands
        target_key = min(pool, key=_vol)
        if len(pool) > 1:
            print("target fallback by size: "
                  + ", ".join(f"{k}={_vol(k):.3f}m3" for k in sorted(pool, key=_vol)[:4]),
                  flush=True)

target_obj = scope[target_key]
target_name = getattr(target_obj, "name", str(target_obj))
tpos = np.asarray(target_obj.get_position_orientation()[0], float)
print(f"TARGET scope={target_key} name={target_name} pos={tpos.round(3).tolist()}", flush=True)

viewer = og.sim.viewer_camera
# v3.2: use seg_instance_id (pixel -> prim path) instead of seg_instance — the
# instance-segmentation reduction node is what segfaults on unresolved ids
# (OG#2312); the instance-ID annotator is a different pipeline, and prim paths
# contain the object name so downstream name-matching is unchanged.
# v5.1: pick the CHEAPEST annotator that can still isolate this target, and
# register only that one.
#
# Always seg_instance_id. An earlier revision preferred seg_semantic whenever the
# target's category had a single instance in scope, on the theory that it "never
# resolves instance ids" and so avoids the OG#2312 reduction-node crash. That came
# from one comparison run and is backwards. Measured over 74 ego renders:
#
#     seg_semantic     42 crashes / 51 renders   82%
#     seg_instance_id   4 crashes / 23 renders   17%
#
# and an A/B on demo 621890 -- same host, same episode, annotator the only variable
# -- reached frame 50/121 with seg_instance_id versus frame 0/121 with seg_semantic.
# Because a single instance in scope is the COMMON case, that rule routed most ego
# renders into the annotator that fails five times more often, which is what made
# whole tasks die as EGO-FAILED.
#
# Dropping seg_semantic also removes the reason target_ids needed a category-name
# fallback: semantic labels are bare categories, so sibling instances of one
# category were indistinguishable and silently shared a mask.
# Registering just one annotator also halves the per-frame pipeline work: both
# were being computed even though only one was ever read.
_cat = target_key.split(".")[0]
_same_cat = [k for k in scope if k.split(".")[0] == _cat]
SEG_KEY = os.environ.get("PILOT_SEG_KEY") or "seg_instance_id"
print(f"annotator: {SEG_KEY} ({len(_same_cat)} instance(s) of {_cat} in scope)", flush=True)
for mod in ("rgb", SEG_KEY):
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
        try:
            viewer.focal_length = track_sensor.focal_length
            print(f"viewer.focal_length <- {viewer.focal_length}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"intrinsics copy focal_length failed: {type(e).__name__}: {e}", flush=True)
        # v3.12: correct horizontal_aperture for ASPECT RATIO before copying.
        #
        # The wrist sensors are configured square (128x128) but the viewer renders
        # 16:9. horizontal_aperture only fixes the HORIZONTAL fov; the vertical one
        # follows aperture * H/W, so copying the number verbatim shrank the wrist
        # view's vertical fov from 63.4deg to 38.3deg — a 40% vertical crop. A wrist
        # camera looks down the gripper, so the target sits low in frame: exactly the
        # band that crop removes. That is why wrist columns so often rendered
        # "object not visible" (measured visible_frac 7-42%).
        #
        # Scaling the aperture by W/H preserves the sensor's vertical fov and widens
        # horizontal to ~95deg, which is also much closer to the wrist cameras real
        # datasets use (DROID's ZED Mini is ~90deg+), where the manipulated object is
        # essentially always in view.
        # NOTE: do not trust viewer.image_width/height here. The viewer reports
        # 1080x1080 while actually rendering 720x1280 (the gm size overrides never
        # take effect, see the lazy-output comment further down), so reading the
        # property yields aspect 1.0 and silently makes this correction a no-op —
        # which is exactly what happened on the first attempt: 39% vs 41%, unchanged.
        # Use the real render aspect, overridable if the viewer size ever changes.
        RENDER_W, RENDER_H = 1280.0, 720.0
        try:
            ap = float(track_sensor.horizontal_aperture)
            # Off by default: the correction was measured on demo 270 right_wrist and
            # moved visible_frac only 41% -> 43%, so the vertical crop was NOT what
            # kept the target out of wrist frames (that turned out to be the card's
            # frame-spread constraint, fixed in make_card_bundle.py). Widening to
            # 95deg also shrinks the object, which works against "the manipulated
            # object should be large in the wrist view". Set PILOT_WRIST_ASPECT_FIX=1
            # to enable if a future robot's wrist sensor really is square.
            aspect = (RENDER_W / RENDER_H) if os.environ.get("PILOT_WRIST_ASPECT_FIX") == "1" else 1.0
            corrected = ap * aspect * float(os.environ.get("PILOT_WRIST_FOV_SCALE", "1.0"))
            viewer.horizontal_aperture = corrected
            hfov = np.degrees(2 * np.arctan(corrected / (2 * float(viewer.focal_length))))
            vfov = np.degrees(2 * np.arctan(corrected * RENDER_H / RENDER_W
                                            / (2 * float(viewer.focal_length))))
            print(f"viewer.horizontal_aperture <- {corrected:.2f} (sensor {ap:.2f}, "
                  f"render {RENDER_W:.0f}x{RENDER_H:.0f}) -> hfov {hfov:.0f}deg vfov {vfov:.0f}deg",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"intrinsics copy horizontal_aperture failed: {type(e).__name__}: {e}", flush=True)
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

        # ---- piecewise-static segments, computed BEFORE scoring ----------------
        # A moving target (11770's can carried into the ashcan) has no single good
        # aim, so the exo is piecewise-static: one pose per segment, re-aimed at that
        # segment's centroid. CRUCIAL: score against the SAME per-segment aim we will
        # render — scoring a global aim while rendering a segment aim made the column
        # render empty (11770: scored 60/60 at y=4.6, rendered blank aimed at y=5.4).
        n_seg = max(1, int(os.environ.get("PILOT_EXO_SEGMENTS", "2")))
        SAMP = 40
        s_idx = [min(_N - 1, int(i * _N / SAMP)) for i in range(SAMP)]
        traj = []
        for fi in s_idx:
            _load_frame(fi)
            traj.append(_tcenter())  # AABB centre
        traj = np.asarray(traj)
        override = os.environ.get("PILOT_SEG_BOUNDS", "").strip()
        if override:
            cuts = [(float(t) if float(t) <= 1.0 else float(t) / 300.0) for t in override.split(",")]
            print(f"exo segments: PILOT_SEG_BOUNDS override -> {cuts}", flush=True)
        elif n_seg > 1:
            # best single cut = the split that most separates the object's positions
            best, cuts = None, [0.5]
            for k in range(3, SAMP - 3):
                a, b = traj[:k], traj[k:]
                cost = a.var(axis=0).sum() * len(a) + b.var(axis=0).sum() * len(b)
                if best is None or cost < best:
                    best, cuts = cost, [k / SAMP]
            print(f"exo segments: auto cut at {cuts[0]:.0%} "
                  f"(target moved {float(np.linalg.norm(traj[-1] - traj[0])):.2f}m end-to-end)", flush=True)
        else:
            cuts = []
        seg_edges = [0.0] + sorted(cuts) + [1.0]
        seg_aims = []
        for s in range(len(seg_edges) - 1):
            sel = traj[max(0, int(seg_edges[s] * SAMP)):max(1, int(seg_edges[s + 1] * SAMP))]
            seg_aims.append(sel.mean(axis=0) if len(sel) else traj.mean(axis=0))
        print("exo segment aims:", [a.round(2).tolist() for a in seg_aims], flush=True)

        def _seg_of(frac):
            return sum(1 for e in seg_edges[1:-1] if frac >= e)

        # v3.15: CORNER mode — cameras in the corners of the ROOM, like the tripods in
        # Ego-Exo4D, instead of on a circle around the target.
        #
        # The ring is placed blind: no line-of-sight check, so on cramped scenes most
        # of its cameras end up inside walls. Measured over 177 cards, only 2.3 of 8
        # ring cameras were usable, and 40% of cards could not field even two.
        #
        # Corners need no occlusion test at all. A camera in the corner of a room, at
        # standing height, aimed into that room, has clear line of sight by
        # construction -- the geometry does the work that scoring kept getting wrong.
        #
        # The room comes from the task scope, which names the floor of the room the
        # activity happens in (floor.n.01_1 -> floors_kxcpgy_0), and each floor object
        # carries a tight AABB of exactly that room.
        CORNERS_N = int(os.environ.get("PILOT_EXO_CORNERS", "0"))
        if CORNERS_N:
            floor_obj = None
            for _k, _v in scope.items():
                if _k.startswith("floor") and _v is not None:
                    floor_obj = _v
                    break
            room_box = None
            if floor_obj is not None:
                try:
                    _lo, _hi = floor_obj.aabb
                    room_box = (np.asarray(_lo, float), np.asarray(_hi, float))
                except Exception as _e:  # noqa: BLE001
                    print(f"corner: floor aabb failed: {type(_e).__name__}: {_e}", flush=True)
            if room_box is None:
                print("corner: no room floor in scope, falling back to ring", flush=True)
                CORNERS_N = 0
            else:
                lo, hi = room_box
                margin = float(os.environ.get("PILOT_CORNER_MARGIN", "0.6"))
                height = float(os.environ.get("PILOT_CORNER_HEIGHT", "2.0"))
                x0, x1 = lo[0] + margin, hi[0] - margin
                y0, y1 = lo[1] + margin, hi[1] - margin
                z = lo[2] + height
                corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)][:CORNERS_N]
                print(f"corner: room {getattr(floor_obj, 'name', '?')} "
                      f"x[{lo[0]:.2f},{hi[0]:.2f}] y[{lo[1]:.2f},{hi[1]:.2f}] "
                      f"-> {len(corners)} cameras at z={z:.2f}", flush=True)
                chosen = []
                for _i, (cx, cy) in enumerate(corners):
                    eye = [float(cx), float(cy), float(z)]
                    chosen.append({
                        "angle": -1, "score": -1, "pred_vis": -1, "corner": _i,
                        "eye": eye, "look": look.tolist(),
                        # Fixed for the whole episode: a corner sees the entire room,
                        # so there is nothing to re-park for as the target moves.
                        "segments": [{"eye": eye, "look": (a + np.array([0.0, 0.0, 0.15])).tolist()}
                                     for a in seg_aims],
                        "seg_edges": seg_edges,
                    })
                with open(poses_file, "w") as f:
                    json.dump(chosen, f)

        # v3.13: RING mode — render a whole ring and let the CARD choose.
        #
        # Every scorer we have tried commits to 2-3 poses from a cheap estimate, and
        # both mispredict the cases that matter: the seg-pixel sweep scored 0 for
        # every pose on multi-word categories, and the analytic frustum+raycast
        # estimate is an unoccluded upper bound (mousetrap 783px^2 predicted vs 60px
        # measured). Measured pixels are the only signal that has never lied, and
        # make_card_bundle already computes them. So stop predicting: place N
        # cameras evenly around the target, render them all, and let the card keep
        # the two with the highest MEASURED visibility.
        # Costs ~4 extra passes per demo (~20 min at fast-replay speed) and deletes
        # the scoring step, which was also the crash-prone part of the exo path.
        # Corner mode already wrote the pose file; the ring would overwrite it.
        RING_N = 0 if CORNERS_N else int(os.environ.get("PILOT_EXO_RING", "0"))
        if RING_N:
            ring_r = EXO_RADII[len(EXO_RADII) // 2]
            # Height from a DEPRESSION ANGLE rather than a fixed offset. A fixed 1.3 m
            # above the target gives ~28 degrees at this radius, which is too shallow to
            # see into a drawer and shallow enough to shoot straight out through a
            # window: demo 822240's battery sits at z=0.43 in a drawer and both of its
            # cameras ended up looking in from outside.
            #
            # Capped in absolute height because the alternative overshoots the ceiling.
            # An earlier sweep tried offsets of 1.9 and 2.5 m and every frame came back
            # showing nothing but `ceilings_*` -- the camera was above the room.
            RING_ELEV = float(os.environ.get("PILOT_RING_ELEV_DEG", "40"))
            RING_ZMAX = float(os.environ.get("PILOT_RING_ZMAX", "2.15"))
            _want = ring_r * np.tan(np.radians(RING_ELEV))
            _headroom = RING_ZMAX - float(aim[2])
            RING_Z = float(os.environ.get("PILOT_RING_Z") or max(0.6, min(_want, _headroom)))
            print(f"exo ring: target z={aim[2]:.2f} -> camera z={aim[2] + RING_Z:.2f} "
                  f"(offset {RING_Z:.2f}, depression {np.degrees(np.arctan2(RING_Z, ring_r)):.0f}deg)",
                  flush=True)
            # Pull each ring camera inside the room. Measured on demo 500130, the ring
            # is not uniformly bad -- exo0/1/2 saw the target 71/81/84% of the time,
            # while exo3..6 saw it 0-3% because they sit inside walls. So the fix is
            # not a different layout (corners were tried: 0-3% visible and 51px,
            # because 3-5m across a room leaves a countertop object too small) but
            # keeping this radius and refusing to place a camera outside the room.
            # The room comes from the floor named in the task scope.
            room = None
            for _k, _v in scope.items():
                if _k.startswith("floor") and _v is not None:
                    try:
                        _lo, _hi = _v.aabb
                        room = (np.asarray(_lo, float), np.asarray(_hi, float))
                    except Exception:  # noqa: BLE001
                        pass
                    break
            _m = float(os.environ.get("PILOT_ROOM_MARGIN", "0.35"))

            def _fit(ang_deg):
                """Largest radius <= ring_r whose position stays inside the room."""
                a = np.deg2rad(ang_deg)
                d = np.array([np.cos(a), np.sin(a)])
                if room is None:
                    return ring_r
                lo, hi = room
                for r in np.arange(ring_r, 0.6, -0.1):
                    p = aim[:2] + d * r
                    if (lo[0] + _m <= p[0] <= hi[0] - _m) and (lo[1] + _m <= p[1] <= hi[1] - _m):
                        return float(r)
                return None      # this direction is wall from the target outward

            # Offset the azimuths by half a step and alternate two radii. A reviewer
            # rejecting a camera as "blocked by a wall / a chandelier / a tree" needs
            # DIFFERENT vantage points, not the same eight re-picked: the old ring put
            # every camera on one circle at one height, so an obstruction that hid the
            # target from one angle usually hid it from its neighbours too. Half-step
            # offsets guarantee none of the new poses repeats an old one, and mixing a
            # near and a far radius varies what each can see past.
            AZ_OFF = float(os.environ.get("PILOT_RING_AZ_OFFSET", "0"))
            # Scale the ring to the TARGET'S SIZE. Targets are restricted to the
            # objects the task manipulates, and those range from a board game to an
            # ice cube; a radius tuned for the former leaves the latter at ~90 median
            # pixels, which a reviewer rightly calls unusable. Pixel area goes as
            # 1/d^2, so pulling a 5 cm object in from 2.2 m to 0.8 m is worth ~7x.
            # The file already anticipated this ("re-run SMALL-object demos closer")
            # but only as a manual override that was never actually used.
            try:
                _lo, _hi = target_obj.aabb
                _size = float(np.linalg.norm(np.asarray(_hi, float) - np.asarray(_lo, float)))
            except Exception:  # noqa: BLE001
                _size = 0.3
            _scale = float(np.clip(_size / 0.30, 0.36, 1.36))   # 0.3m object keeps 2.2m
            r_near = float(np.clip(EXO_RADII[0] * _scale, 0.8, 3.0))
            r_far = float(np.clip((EXO_RADII[-1] if len(EXO_RADII) > 1 else ring_r) * _scale,
                                  0.8, 3.0))
            print(f"exo ring: target size {_size:.2f}m -> radii {r_near:.2f}/{r_far:.2f}m",
                  flush=True)
            chosen = []
            for i in range(RING_N):
                ang = 360.0 * i / RING_N + AZ_OFF
                ring_r = r_near if i % 2 == 0 else r_far
                r = _fit(ang)
                if r is None:
                    print(f"exo ring: angle {ang:.0f}deg is outside the room, dropped", flush=True)
                    continue
                chosen.append({"angle": ang, "score": -1, "pred_vis": -1, "radius": r,
                               "off": [r * np.cos(np.deg2rad(ang)),
                                       r * np.sin(np.deg2rad(ang)), RING_Z]})
            if room is not None:
                print(f"exo ring: room x[{room[0][0]:.2f},{room[1][0]:.2f}] "
                      f"y[{room[0][1]:.2f},{room[1][1]:.2f}]; kept {len(chosen)}/{RING_N} angles, "
                      f"radii {[round(c['radius'], 1) for c in chosen]}", flush=True)
            if not chosen:   # degenerate room box: fall back to the blind ring
                chosen = [{"angle": 360.0 * i / RING_N, "score": -1, "pred_vis": -1,
                           "off": [ring_r * np.cos(np.deg2rad(360.0 * i / RING_N)),
                                   ring_r * np.sin(np.deg2rad(360.0 * i / RING_N)), RING_Z]}
                          for i in range(RING_N)]
            for c in chosen:
                c["eye"] = (aim + np.asarray(c["off"], float)).tolist()
                c["look"] = look.tolist()
                c["segments"] = [{"eye": (a + np.asarray(c["off"], float)).tolist(),
                                  "look": (a + np.array([0.0, 0.0, 0.15])).tolist()}
                                 for a in seg_aims]
                c["seg_edges"] = seg_edges
            with open(poses_file, "w") as f:
                json.dump(chosen, f)
            print(f"exo ring: {RING_N} cameras at r={ring_r}m every {360.0 / RING_N:.0f}deg; "
                  f"the card keeps the best two by measured pixels", flush=True)

        # candidate grid: 3 radii x 12 azimuths. Each candidate is an OFFSET
        # (angle+radius); per frame it is applied to that frame's SEGMENT aim, so the
        # score measures exactly the piecewise-static pose that gets rendered.
        cand = []
        for r in EXO_RADII:
            for a_deg in range(0, 360, 30):
                a = np.deg2rad(a_deg)
                cand.append({"angle": a_deg, "off": np.array([r * np.cos(a), r * np.sin(a), 1.3])})
        MIN_PROJ_PX = 200.0  # same bar make_card_bundle uses -> pred_vis ~ measured
        tgt_body = getattr(target_obj, "prim_path", "") or target_name
        if not RING_N:  # RING mode already wrote its poses; skip all scoring
            F = int(os.environ.get("PILOT_TRAJ_SAMPLES", "60"))
            nvis = [0] * len(cand)
            area = [0.0] * len(cand)
            for k in range(F):
                frac = (k + 0.5) / F
                _load_frame(min(_N - 1, int(frac * _N)))
                try:
                    lo, hi = target_obj.aabb
                    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
                    tc = (lo + hi) / 2.0
                except Exception:  # noqa: BLE001
                    tc = np.asarray(target_obj.get_position_orientation()[0], float)
                    lo, hi = tc - 0.08, tc + 0.08
                rp = np.asarray(robot.get_position_orientation()[0], float)
                sa = seg_aims[_seg_of(frac)]                      # this frame's segment aim
                slook = sa + np.array([0.0, 0.0, 0.15])
                for i, c in enumerate(cand):
                    eye = sa + c["off"]                            # the pose we WILL render this segment
                    infr, fz = _in_frustum(eye, slook, tc)
                    if not infr:
                        continue
                    if _proj_area_px(eye, slook, lo, hi) < MIN_PROJ_PX:
                        continue
                    # Occlusion cast TARGET -> EYE (target's own body ignored): a camera
                    # buried in a wall sees no backface on the way out, so eye->target
                    # falsely reports clear; from the target side the wall is a front face.
                    try:
                        if raytest(tc.tolist(), eye.tolist(), ignore_bodies=[tgt_body]).get("hit"):
                            continue
                    except Exception:  # noqa: BLE001
                        pass  # raycast unavailable -> treat as visible
                    if not _in_frustum(eye, slook, rp)[0]:
                        continue  # robot not in frame
                    nvis[i] += 1
                    area[i] += _proj_area_px(eye, slook, lo, hi)
            ranked = sorted((i for i in range(len(cand)) if nvis[i] > 0),
                            key=lambda i: (-nvis[i], -area[i]))
            chosen, used_angles = [], []
            for i in ranked:
                ang = cand[i]["angle"]
                if all(min(abs(ang - u), 360 - abs(ang - u)) >= 45 for u in used_angles):
                    segs = [{"eye": (a + cand[i]["off"]).tolist(),
                             "look": (a + np.array([0.0, 0.0, 0.15])).tolist()} for a in seg_aims]
                    chosen.append({"angle": ang, "score": nvis[i], "pred_vis": round(nvis[i] / F, 2),
                                   "eye": segs[0]["eye"], "look": segs[0]["look"],
                                   "segments": segs, "seg_edges": seg_edges})
                    used_angles.append(ang)
                if len(chosen) == 3:
                    break
            if chosen:
                with open(poses_file, "w") as f:
                    json.dump(chosen, f)
                print("geometric placement (per-segment):",
                      [(c["angle"], c["score"], c["pred_vis"]) for c in chosen], flush=True)
            else:
                chosen = _ring  # nothing visible from any pose -> keep the mid-aimed ring
                print("geometric: no visible pose; keeping mid-aimed fixed-ring", flush=True)

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


# Objects the gaze may settle on: every manipulandum the goal names, not just the one
# this card happens to be about, each paired with where the goal sends it.
GAZE_OBJS = []
for _t in manip_types:
    _dest = None
    _dt = goal_of.get(_t)
    if _dt:
        _dk = sorted(k for k in scope if _dt in k)
        if _dk:
            _dest = scope[_dk[0]]
    for _k in sorted(k for k in cands if _t in k):
        GAZE_OBJS.append((scope[_k], _dest))
if not GAZE_OBJS:
    GAZE_OBJS = [(target_obj, None)]
print(f"gaze targets: {[(getattr(o, 'name', '?'), getattr(d, 'name', None)) for o, d in GAZE_OBJS]}",
      flush=True)
# The hands. The wrist sensors sit on them, which avoids depending on a robot-class
# specific end-effector accessor.
HANDS = [s for k, s in robot.sensors.items() if "realsense_link" in k]
_gaze_state = {"pair": GAZE_OBJS[0]}
_pos = lambda o: np.asarray(o.get_position_orientation()[0], float)


def _gaze_point():
    """Where an egocentric view would actually be looking.

    This used to be locked to THIS card's target for the whole episode — the comment
    called it "idealized ego". It reads wrong: while the robot works on something else
    the view stares across the room at an object it is not touching. It also demands
    far more angular velocity than a real head, because holding a distant object
    centred means sweeping against every base rotation, and moving-camera renders are
    what the seg pipeline chokes on (all 32 blank frames in one review set were on
    this view, the only steered one).

    Two corrections, both from what a person's gaze does while manipulating:
      * attend to the manipulandum nearest a hand — the thing being carried, or the
        thing being reached for as the hand closes on it;
      * bias toward that object's DESTINATION as it gets picked up. Gaze leads the
        hand: once you are holding something you are already looking at where it is
        going. Blending by grip proximity gives that for free — far from the object
        the camera watches the object, once holding it the camera drifts to the goal.
    """
    if not HANDS:
        return _pos(_gaze_state["pair"][0])
    hands = [np.asarray(h.get_position_orientation()[0], float) for h in HANDS]
    dist = lambda p: min(float(np.linalg.norm(p - h)) for h in hands)

    best, best_d = None, np.inf
    for pair in GAZE_OBJS:
        d = dist(_pos(pair[0]))
        if d < best_d:
            best, best_d = pair, d
    cur = _gaze_state["pair"]
    cur_d = dist(_pos(cur[0]))
    # Switch only on a clear win, or the gaze jitters between neighbours every frame.
    if best is not cur and best_d < 0.8 * cur_d:
        _gaze_state["pair"] = cur = best
        cur_d = best_d

    obj, dest = cur
    op = _pos(obj)
    if dest is None:
        return op
    # w=0 while the object is out of reach, rising to 0.5 once it is in the hand:
    # the gaze never overshoots past the midpoint, it only leads toward the goal.
    w = 0.5 * max(0.0, min(1.0, (0.35 - cur_d) / 0.35))
    return (1.0 - w) * op + w * _pos(dest)


GAZE_MAX_DEG = float(os.environ.get("PILOT_GAZE_MAX_DEG", "70"))


def _clamped_lookat(eye, target, head_orn):
    """Aim at @target, but never further than GAZE_MAX_DEG off where the head faces.

    Neither extreme works here. BEHAVIOR's own head camera is essentially level and
    does not tilt down to what is being manipulated, so the object sits at the edge of
    frame or outside it. But overriding the orientation outright — which is what this
    did — lets a camera MOUNTED ON THE HEAD swing backwards and stare through the
    robot's own head and back, which no real head camera can do; several frames came
    out as nothing but black chassis.

    Clamping keeps the useful half: the camera may pitch down and turn toward the
    object like a real active head, and stops at the cone boundary when the object is
    behind, where it then shows what the robot is facing.
    """
    from scipy.spatial.transform import Rotation as Rot

    fwd = Rot.from_quat(np.asarray(head_orn, float)).apply([0.0, 0.0, -1.0])
    fwd = fwd / (np.linalg.norm(fwd) or 1.0)
    want = np.asarray(target, float) - np.asarray(eye, float)
    n = np.linalg.norm(want)
    if n < 1e-6:
        return head_orn
    want = want / n

    cos_max = np.cos(np.radians(GAZE_MAX_DEG))
    c = float(np.clip(np.dot(fwd, want), -1.0, 1.0))
    if c < cos_max:
        # Rotate `want` back toward `fwd` about their common perpendicular, landing
        # exactly on the cone edge. Slerp rather than a blend so the result stays unit
        # length and the direction is exactly GAZE_MAX_DEG off forward.
        axis = np.cross(fwd, want)
        na = np.linalg.norm(axis)
        if na < 1e-6:          # exactly antiparallel: no unique plane, keep facing
            return head_orn
        axis = axis / na
        want = Rot.from_rotvec(axis * np.radians(GAZE_MAX_DEG)).apply(fwd)
    return _lookat_quat(eye, np.asarray(eye, float) + want * n)


def _grab(r):
    """Move the camera for raw step @r, render, and append one frame to the outputs.
    Split out of capture() so fast replay can call it directly at the strided
    frames without going through playback_episode's per-step callback."""
    if track_sensor is not None:
        pos, orn = track_sensor.get_position_orientation()
        if gaze_at_target:
            orn = _clamped_lookat(np.asarray(pos, float), _gaze_point(), orn)
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
    # Some restored states render blank however many settle renders preceded them:
    # 31 of 122 ego frames on demo 500130, and at the SAME steps across two runs with
    # different camera orientations, so it follows the state rather than the camera or
    # a random GPU flake. It is recoverable though — rendering that state again lands
    # it. Detect on the segmentation (a blank frame is one id where a real interior
    # has ~15) and re-render rather than shipping a black frame the sampler cannot
    # tell apart from "the object isn't visible".
    for _try in range(int(os.environ.get("PILOT_BLANK_RETRIES", "3"))):
        if len(np.unique(np.asarray(obs[SEG_KEY])[::8, ::8])) > 2:
            break
        for _ in range(2):
            og.sim.render()
        obs2, _ = _safe_get_obs(viewer)
        if obs2 is None:
            break
        obs = obs2
        state["blank_fixed"] = state.get("blank_fixed", 0) + 1
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
        # Settle the jumped-to state before reading the annotator. Two renders left
        # the occasional frame black in EVERY view at once (631220 step 223), which
        # points at the state not being fully applied rather than at a per-process
        # glitch. Renders are cheap and, unlike get_obs, carry no OG#2312 risk.
        for _ in range(int(os.environ.get("PILOT_SETTLE_RENDERS", "4"))):
            og.sim.render()
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
# Several tasks manipulate more than one object ("put the apple AND the banana
# away"). Each is a valid card from these very same frames — only the mask lookup
# differs — so record every manipulandum the BDDL goal named, resolved to the
# concrete scene instances, and let the bundler emit a card per object.
_alts = []
for _t in manip_types:
    for _k in sorted(k for k in cands if _t in k):
        _n = getattr(scope[_k], "name", None)
        if _n and _n not in _alts:
            _alts.append(_n)
h5f.attrs["manipulanda"] = json.dumps(_alts)
print(f"manipulanda recorded: {_alts}", flush=True)
h5f.close()
print("VIEWPASS-OK", VIEW, DEMO_ID, T, flush=True)
og.shutdown()
