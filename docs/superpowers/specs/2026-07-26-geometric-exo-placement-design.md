# Geometry-driven exo camera placement

**Status:** approved, implementation deferred until the current 40-task production run finishes.
**Applies to:** `BehaviorPilot/remote/pilot_viewercam.py` (exo branch only).

## Problem

Exo camera poses are chosen by a sweep that scores 12 candidate poses at 3 sampled
frames, using `get_obs` reads of the `seg_semantic` annotator. Three things go wrong:

1. **It gambles on 3 frames.** A pose that frames the target at 40/60/80% of the
   episode can be blind for the rest. What we actually care about — and what we
   measure afterwards as `visible_frac` — is the fraction of *all* frames where the
   target is visible. The sweep never measures that.
2. **It cannot be widened.** Scoring is 36 `get_obs` reads, already at the OG#2312
   crash threshold (demo 600's `exo0` dies inside the sweep). More poses or more
   frames means more crashes.
3. **Failures are only discovered afterwards.** A blind column is detected at card
   render time by `visible_frac < 0.05` and dropped, wasting a full episode replay.

Root cause: placement is decided by expensive sampling, then verified post-hoc,
instead of being computed from information we can obtain cheaply.

## Key enabler

The target's 3D pose is available on every frame for free:
`target_obj.get_position_orientation()` after `og.sim.load_state()` — no render, no
`get_obs`, no OG#2312 exposure. Occlusion is available just as cheaply through
PhysX scene queries (`omnigibson.utils.sampling_utils.raytest_batch`), which do not
touch the renderer. Camera intrinsics come from the `VisionSensor`
(`focal_length`, `horizontal_aperture`).

So "is the target inside this camera's frustum, big enough, and unoccluded?" can be
answered analytically for every frame and every candidate pose.

## Design

### 1. Trajectory scan

Sample `F` frames evenly over the episode (default 60, `PILOT_TRAJ_SAMPLES`). Per
frame: `load_state` + `og.sim.step()`, then record target position, target AABB, and
robot base position. No rendering, no annotator reads.

Replaces 36 × (2 renders + `get_obs`) with 60 × `load_state`, so it is both faster
and free of the crash source.

### 2. Candidate grid

Widen from 2 radii × 6 azimuths (12) to **3 radii (1.6 / 2.2 / 3.0) × 12 azimuths
(30°) = 36**. Affordable only because scoring no longer renders.

### 3. Per-pose, per-frame score

- **Frustum:** transform the target into camera space via the look-at orientation;
  require `|atan2(x, z)| < hfov/2` and `|atan2(y, z)| < vfov/2`, with
  `hfov = 2·atan(horizontal_aperture / (2·focal_length))`.
- **Projected size:** `target_radius / distance × focal`, required to clear the
  pixel-area bar that `make_card_bundle.py` already uses (`VIS_PX = 200`).
- **Occlusion:** raycast from the eye to the target AABB centre plus a few corners;
  visible if a ray's first hit belongs to the target's rigid body.
- **Robot framing:** the same frustum test on the robot base, no occlusion test
  (the robot is large enough that partial occlusion still reads).

**Pose score = number of frames where the target is visible and large enough AND the
robot is in frame.** Ties broken by mean projected target area. This is a direct
estimator of the `visible_frac` we measure later.

### 4. Interaction with segments

The episode is already split into `PILOT_EXO_SEGMENTS` (default 2) piecewise-static
segments, each re-aimed at that segment's target centroid.

Score each candidate azimuth **per segment**, then for each exo slot pick the azimuth
maximising the **minimum score across segments** — a pose that holds up in both
halves, rather than one that is perfect early and blind later. Per-segment re-aiming
(already implemented) stays.

Only if every azimuth scores 0 in some segment does the slot fall back to choosing
each segment's best azimuth independently, which makes the viewpoint jump at the
boundary. Log clearly when this happens.

### 5. Fallbacks

- Trajectory scan raises → keep the existing pre-sweep mid-aimed fixed ring.
- Every pose scores 0 in every segment → keep the fixed ring, log loudly.

### 6. Deletions

`_score_pose`, the 36 `seg_semantic` reads, and the 20 warm-up renders before the
sweep all go away. This removes the single largest crash source in the exo path.

### 7. Built-in acceptance test

Each chosen pose records its **predicted** visible-frame fraction into the pose file
and the log. `make_card_bundle.py` already records the **measured** `visible_frac`
per view. Predicted and measured should correlate strongly; that correlation is the
acceptance criterion for this change, replacing eyeball inspection.

## Not in scope

- Parallelising exo0/1/2. The scan no longer needs `get_obs`, so in principle every
  exo process could compute its own poses and the `poses_task-*.json` handshake
  (and its 240 × 5s wait) could go away. Deliberately deferred — this change is
  about placement quality, not scheduling.
- Per-column frame selection. The card keeps one shared set of 8 steps so rows stay
  time-aligned across views.

## Open question

Section 4's "maximise the minimum across segments" rule is the author's proposal and
has not been confirmed against a case where the object changes room mid-episode.
Revisit if such demos show up in the run's `quality.txt`.
