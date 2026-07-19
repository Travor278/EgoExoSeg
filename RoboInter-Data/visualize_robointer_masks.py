"""Visualize RoboInter segmentation masks on DROID episodes (LeRobot low-res pack).

Two figures per DROID base episode (both exterior views must be in the local
chunk-000 parquets and have npz masks in segmentation_npz/):

  samples/{base}_crossview_mask.png
      columns = exterior_1 (+mask) | exterior_2 (+mask) | wrist (no mask),
      rows = 4 sampled time steps. Shows that RoboInter masks exist ONLY on
      the two exterior cameras; the wrist stream ships without masks.

  samples/{base}_annotation_suite.png
      exterior_1 only, with the full per-frame intermediate-representation
      suite overlaid: segmentation mask + object_box + gripper_box + trace
      (future 10 gripper waypoints) + contact points + placement proposal.
      Shows what the mask is FOR: policy-oriented object grounding, not
      cross-view correspondence.

Prereqs:
  - videos/lerobot_droid_chunk-000.tar downloaded (mp4s are extracted lazily)
  - segmentation npz fetched via fetch_segmentation_npz.py

Usage:
  python visualize_robointer_masks.py            # default bases
  python visualize_robointer_masks.py 10020 0
"""
from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from crossview_viz import build_montage, decode_video_frames, sample_indices  # noqa: E402

CHUNK_DIR = ROOT / "data" / "lerobot_droid_chunk-000" / "chunk-000"
VIDEO_TAR = ROOT / "videos" / "lerobot_droid_chunk-000.tar"
VIDEO_DIR = ROOT / "videos"
NPZ_DIRS = [ROOT / "segmentation_npz", ROOT / "sample_npz"]
OUT_DIR = ROOT / "samples"

SCALE = 2  # draw overlays at 2x for crisper contours (320x180 -> 640x360)
MASK_COLOR = (0, 220, 130)      # green fill for the annotated object
MASK_EDGE = (0, 255, 170)
OBJ_BOX = (255, 210, 40)        # yellow  - manipulated object bbox
GRIP_BOX = (60, 200, 255)       # cyan    - gripper bbox
TRACE_COL = (255, 80, 200)      # magenta - future gripper trace
CONTACT_COL = (255, 60, 60)     # red     - contact points
PLACE_COL = (240, 240, 240)     # white   - placement proposal


def episode_pairs() -> dict[str, dict[str, tuple[int, str]]]:
    """base -> {'1': (episode_index, episode_name), '2': (...)} from local parquets."""
    pairs: dict[str, dict[str, tuple[int, str]]] = {}
    for p in sorted(CHUNK_DIR.glob("*.parquet")):
        t = pq.read_table(p, columns=["episode_index", "episode_name", "camera_view"]).to_pandas().iloc[0]
        name = str(t["episode_name"])
        for view in ("1", "2"):
            suffix = f"_exterior_image_{view}_left"
            if name.endswith(suffix):
                pairs.setdefault(name[: -len(suffix)], {})[view] = (int(t["episode_index"]), name)
    return pairs


def ensure_videos(ep_indices: list[int]) -> None:
    """Extract primary+wrist mp4s for the given episode indices from the chunk tar."""
    needed = []
    for idx in ep_indices:
        for cam in ("observation.images.primary", "observation.images.wrist"):
            rel = f"chunk-000/{cam}/episode_{idx:06d}.mp4"
            if not (VIDEO_DIR / rel).exists():
                needed.append(rel)
    if not needed:
        return
    if not VIDEO_TAR.exists():
        raise FileNotFoundError(f"video tar missing: {VIDEO_TAR}")
    with tarfile.open(VIDEO_TAR) as tf:
        by_suffix = {m.name.lstrip("./"): m for m in tf.getmembers() if m.isfile()}
        for rel in needed:
            member = by_suffix.get(rel) or next((m for k, m in by_suffix.items() if k.endswith(rel)), None)
            if member is None:
                raise KeyError(f"{rel} not found in {VIDEO_TAR.name}")
            member.name = rel
            tf.extract(member, VIDEO_DIR)
            print(f"  extracted {rel}")


def load_masks(episode_name: str) -> np.ndarray | None:
    """(T, H, W) bool masks of the single annotated object, or None."""
    for d in NPZ_DIRS:
        p = d / f"{episode_name}.npz"
        if p.exists():
            m = np.load(p)["masks"]          # (n_obj, T, 1, H, W)
            return m[0, :, 0].astype(bool)   # all released episodes have n_obj == 1
    return None


def video_path(idx: int, cam: str) -> Path:
    return VIDEO_DIR / f"chunk-000/observation.images.{cam}/episode_{idx:06d}.mp4"


def upscale(rgb: np.ndarray) -> np.ndarray:
    return cv2.resize(rgb, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LANCZOS4)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray | None, alpha: float = 0.45) -> np.ndarray:
    img = upscale(rgb)
    if mask is None:
        strip = img.copy()
        cv2.rectangle(strip, (0, 0), (238, 26), (30, 30, 34), -1)
        cv2.putText(strip, "RoboInter mask: none", (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 150, 60), 1, cv2.LINE_AA)
        return strip
    m = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    mb = m.astype(bool)
    img[mb] = (img[mb] * (1 - alpha) + np.array(MASK_COLOR) * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, MASK_EDGE, 2, cv2.LINE_AA)
    if not mb.any():
        cv2.rectangle(img, (0, 0), (268, 26), (30, 30, 34), -1)
        cv2.putText(img, "object not visible (empty)", (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 210), 1, cv2.LINE_AA)
    return img


def _load_json(cell) -> object | None:
    s = str(cell).strip()
    if not s or s in ("nan", "None", "null"):
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _rect(img, box, color, thickness=2, label=None):
    (x1, y1), (x2, y2) = box
    p1, p2 = (int(x1) * SCALE, int(y1) * SCALE), (int(x2) * SCALE, int(y2) * SCALE)
    cv2.rectangle(img, p1, p2, color, thickness, cv2.LINE_AA)
    if label:
        cv2.putText(img, label, (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def overlay_suite(rgb: np.ndarray, mask: np.ndarray | None, row) -> np.ndarray:
    img = overlay_mask(rgb, mask)
    roi_pts: list[tuple[float, float]] = []  # source-res points for the zoom inset
    if mask is not None and mask.any():
        ys, xs = np.nonzero(mask)
        roi_pts += [(xs.min(), ys.min()), (xs.max(), ys.max())]
    box = _load_json(row["annotation.object_box"])
    if box:
        _rect(img, box, OBJ_BOX, 2, "object")
        roi_pts += [tuple(box[0]), tuple(box[1])]
    gbox = _load_json(row["annotation.gripper_box"])
    if gbox:
        _rect(img, gbox, GRIP_BOX, 2, "gripper")
        roi_pts += [tuple(gbox[0]), tuple(gbox[1])]
    pbox = _load_json(row["annotation.placement_proposal"])
    if pbox:
        _rect(img, pbox, PLACE_COL, 1, "placement")
        roi_pts += [tuple(pbox[0]), tuple(pbox[1])]
    trace = _load_json(row["annotation.trace"])
    if trace and len(trace) >= 2:
        pts = np.array([[int(x) * SCALE, int(y) * SCALE] for x, y in trace], np.int32)
        cv2.polylines(img, [pts], False, TRACE_COL, 2, cv2.LINE_AA)
        for x, y in pts:
            cv2.circle(img, (x, y), 2, TRACE_COL, -1, cv2.LINE_AA)
        cv2.circle(img, tuple(pts[-1]), 4, TRACE_COL, 2, cv2.LINE_AA)
    cpts = _load_json(row["annotation.contact_points"])
    if cpts:
        cpts = cpts if isinstance(cpts[0], (list, tuple)) else [cpts]
        for x, y in cpts:
            cv2.drawMarker(img, (int(x) * SCALE, int(y) * SCALE), CONTACT_COL, cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
    if roi_pts:
        img = _paste_zoom_inset(img, roi_pts)
    return img


def _paste_zoom_inset(img: np.ndarray, roi_pts: list[tuple[float, float]], zoom_h: int = 210) -> np.ndarray:
    """Crop around the annotated region (everything is tiny in DROID exo views)
    and paste a magnified copy in the bottom-left corner."""
    H, W = img.shape[:2]
    xs = [p[0] * SCALE for p in roi_pts]
    ys = [p[1] * SCALE for p in roi_pts]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half_w = max(60 * SCALE // 2, int((max(xs) - min(xs)) * 0.75)) + 8
    half_h = max(42 * SCALE // 2, int((max(ys) - min(ys)) * 0.75)) + 8
    half_w = max(half_w, int(half_h * 4 / 3))
    half_h = max(half_h, int(half_w * 3 / 4))
    x1, x2 = int(np.clip(cx - half_w, 0, W - 1)), int(np.clip(cx + half_w, 1, W))
    y1, y2 = int(np.clip(cy - half_h, 0, H - 1)), int(np.clip(cy + half_h, 1, H))
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return img
    zw = int(zoom_h * crop.shape[1] / crop.shape[0])
    zoom = cv2.resize(crop, (zw, zoom_h), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(zoom, (0, 0), (zw - 1, zoom_h - 1), (255, 255, 255), 2)
    cv2.putText(zoom, "zoom", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    oy, ox = H - zoom_h - 8, 8
    if oy >= 0 and ox + zw <= W:
        img[oy : oy + zoom_h, ox : ox + zw] = zoom
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1)
    return img


def render_base(base: str, pairs) -> None:
    info = pairs.get(base)
    if not info or "1" not in info or "2" not in info:
        print(f"!! base {base}: both exterior views not in local chunk-000, skip")
        return
    (idx1, name1), (idx2, name2) = info["1"], info["2"]
    m1, m2 = load_masks(name1), load_masks(name2)
    if m1 is None or m2 is None:
        print(f"!! base {base}: npz missing (run fetch_segmentation_npz.py), skip")
        return
    ensure_videos([idx1, idx2])

    df1 = pq.read_table(CHUNK_DIR / f"episode_{idx1:06d}.parquet").to_pandas()
    task = str(df1.iloc[0]["annotation.instruction_add"]) or "(no instruction)"
    n = min(len(df1), m1.shape[0], m2.shape[0])
    frames_idx = sample_indices(n, 4)

    ext1 = decode_video_frames(video_path(idx1, "primary"), frames_idx)
    ext2 = decode_video_frames(video_path(idx2, "primary"), frames_idx)
    wrist = decode_video_frames(video_path(idx1, "wrist"), frames_idx)

    rows = []
    for k, f in enumerate(frames_idx):
        skill = str(df1.iloc[f]["annotation.primitive_skill"]) or "-"
        rows.append(
            (
                f"frame {f}",
                f"t={f / 10:.1f}s · {skill}",
                [overlay_mask(ext1[k], m1[f]), overlay_mask(ext2[k], m2[f]), overlay_mask(wrist[k], None)],
            )
        )
    out1 = build_montage(
        OUT_DIR / f"{base}_crossview_mask.png",
        f"RoboInter mask coverage · DROID episode {base}",
        f'"{task}"  ·  masks: SAM2 + human review, exterior cameras ONLY (1 object/video, 180x320@10fps)',
        [f"exterior_1 (ep{idx1:06d}) + mask", f"exterior_2 (ep{idx2:06d}) + mask", f"wrist (ep{idx1:06d})"],
        rows,
        footer=f"source: InternRobotics/RoboInter-Data lerobot_droid_anno · npz: Annotation_raw/segmentation_npz · {name1} / {name2}",
        tile=(360, 203),
    )
    print(f"wrote {out1}")

    rows2 = []
    for k, f in enumerate(frames_idx):
        r = df1.iloc[f]
        sub = str(r["annotation.substask"]) or "-"
        rows2.append((f"frame {f}", (sub[:26] + "…") if len(sub) > 27 else sub, [overlay_suite(ext1[k], m1[f], r)]))
    out2 = build_montage(
        OUT_DIR / f"{base}_annotation_suite.png",
        f"RoboInter intermediate representations · DROID {base} · exterior_1",
        f'"{task}"  ·  mask + object/gripper/placement boxes + 10-step gripper trace + contact points',
        [f"exterior_1 (ep{idx1:06d}) · full annotation suite"],
        rows2,
        footer="all 2D annotations live in the PRIMARY (exterior) camera pixel frame - designed to condition/supervise a manipulation policy (plan-then-execute VLA), not for cross-view correspondence",
        tile=(640, 360),
    )
    print(f"wrote {out2}")


def main(bases: list[str]) -> None:
    pairs = episode_pairs()
    print(f"local chunk-000: {len(pairs)} bases with at least one exterior view")
    for base in bases:
        render_base(base, pairs)


if __name__ == "__main__":
    main(sys.argv[1:] or ["10020", "10010", "0"])
