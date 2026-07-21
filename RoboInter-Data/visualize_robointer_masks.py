"""Visualize RoboInter segmentation masks on DROID / RH20T episodes (LeRobot pack).

Figures per base (a DROID video_id, or an RH20T take):

  samples/{base}_crossview_mask.png
      columns = camera A (+mask) | camera B (+mask) | wrist,
      rows = 4 sampled time steps. RoboInter masks exist ONLY on the two
      exterior / global cameras; the wrist stream ships without masks.
      If a SAM2 wrist prediction npz exists (see sam2_wrist_preview.py), the
      wrist column shows it in ORANGE, clearly tagged as a prediction.

  samples/{base}_annotation_suite.png   (DROID only)
      camera A with the full per-frame intermediate-representation suite.

Prereqs:
  - videos/lerobot_droid_chunk-000.tar and/or videos/lerobot_rh20t_chunk-000.tar
  - segmentation npz fetched via fetch_segmentation_npz.py

Usage:
  python visualize_robointer_masks.py 10020 0            # DROID bases
  python visualize_robointer_masks.py --rh20t auto       # best local RH20T take
  python visualize_robointer_masks.py --rh20t RH20T_cfg1_task_0001_user_0001_scene_0001_cfg_0001
"""
from __future__ import annotations

import json
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from crossview_viz import build_montage, decode_video_frames, sample_indices  # noqa: E402

DATASETS = {
    "droid": {
        "chunk_dir": ROOT / "data" / "lerobot_droid_chunk-000" / "chunk-000",
        "tar": ROOT / "videos" / "lerobot_droid_chunk-000.tar",
        "extract_root": ROOT / "videos",  # legacy layout: videos/chunk-000/...
    },
    "rh20t": {
        "chunk_dir": ROOT / "data" / "lerobot_rh20t_chunk-000" / "chunk-000",
        "tar": ROOT / "videos" / "lerobot_rh20t_chunk-000.tar",
        "extract_root": ROOT / "videos" / "rh20t",
    },
}
NPZ_DIRS = [ROOT / "segmentation_npz", ROOT / "sample_npz"]
SAM2_DIR = ROOT / "sam2_wrist"
OUT_DIR = ROOT / "samples"

SCALE = 2  # draw overlays at 2x (320x180 -> 640x360; matches RH20T mask res)
MASK_COLOR = (0, 220, 130)      # green fill - RoboInter GT mask
MASK_EDGE = (0, 255, 170)
PRED_COLOR = (255, 140, 0)      # orange fill - SAM2 predicted wrist mask
PRED_EDGE = (255, 190, 80)
OBJ_BOX = (255, 210, 40)
GRIP_BOX = (60, 200, 255)
TRACE_COL = (255, 80, 200)
CONTACT_COL = (255, 60, 60)
PLACE_COL = (240, 240, 240)

SEG_MAP = json.loads((ROOT / "VideoID_2_SegmentationNPZ.json").read_text(encoding="utf-8"))


def scan_chunk(ds: str) -> list[tuple[int, str]]:
    """[(episode_index, episode_name)] for the local chunk of a dataset."""
    rows = []
    for p in sorted(DATASETS[ds]["chunk_dir"].glob("*.parquet")):
        t = pq.read_table(p, columns=["episode_index", "episode_name"]).to_pandas().iloc[0]
        rows.append((int(t["episode_index"]), str(t["episode_name"])))
    return rows


def droid_pairs() -> dict[str, dict[str, tuple[int, str]]]:
    pairs: dict[str, dict[str, tuple[int, str]]] = {}
    for idx, name in scan_chunk("droid"):
        for view in ("1", "2"):
            suffix = f"_exterior_image_{view}_left"
            if name.endswith(suffix):
                pairs.setdefault(name[: -len(suffix)], {})[view] = (idx, name)
    return pairs


def rh20t_takes() -> dict[str, list[tuple[int, str]]]:
    """take -> [(episode_index, episode_name)] for cams that HAVE npz, local chunk only."""
    takes: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for idx, name in scan_chunk("rh20t"):
        if SEG_MAP.get(name):
            takes[name.rsplit("_", 1)[0]].append((idx, name))
    return takes


def ensure_videos(ds: str, ep_indices: list[int]) -> None:
    cfg = DATASETS[ds]
    needed = []
    for idx in ep_indices:
        for cam in ("observation.images.primary", "observation.images.wrist"):
            rel = f"chunk-000/{cam}/episode_{idx:06d}.mp4"
            if not (cfg["extract_root"] / rel).exists():
                needed.append(rel)
    if not needed:
        return
    if not cfg["tar"].exists():
        raise FileNotFoundError(f"video tar missing: {cfg['tar']}")
    with tarfile.open(cfg["tar"]) as tf:
        by_name = {m.name.lstrip("./"): m for m in tf.getmembers() if m.isfile()}
        for rel in needed:
            member = by_name.get(rel) or next((m for k, m in by_name.items() if k.endswith(rel)), None)
            if member is None:
                raise KeyError(f"{rel} not found in {cfg['tar'].name}")
            member.name = rel
            tf.extract(member, cfg["extract_root"])
            print(f"  extracted {ds}:{rel}")


def video_path(ds: str, idx: int, cam: str) -> Path:
    return DATASETS[ds]["extract_root"] / f"chunk-000/observation.images.{cam}/episode_{idx:06d}.mp4"


def load_masks(episode_name: str) -> np.ndarray | None:
    for d in NPZ_DIRS:
        p = d / f"{episode_name}.npz"
        if p.exists():
            m = np.load(p)["masks"]
            return m[0, :, 0].astype(bool)
    return None


def load_sam2_wrist(base: str) -> tuple[np.ndarray, str] | None:
    """(T,H,W) predicted wrist masks + a short provenance tag, if present."""
    p = SAM2_DIR / f"{base}_wrist_sam2.npz"
    if not p.exists():
        return None
    d = np.load(p)  # our own output; tag stored as a plain unicode array, no pickle needed
    tag = str(d["tag"]) if "tag" in d.files else "SAM2 prediction"
    return d["masks"].astype(bool), tag


def upscale(rgb: np.ndarray) -> np.ndarray:
    return cv2.resize(rgb, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LANCZOS4)


def _tag(img: np.ndarray, text: str, color=(200, 200, 210)) -> None:
    w = 12 + int(cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0])
    cv2.rectangle(img, (0, 0), (w, 26), (30, 30, 34), -1)
    cv2.putText(img, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def overlay_mask(
    rgb: np.ndarray,
    mask: np.ndarray | None,
    alpha: float = 0.45,
    fill=MASK_COLOR,
    edge=MASK_EDGE,
    none_text: str = "RoboInter mask: none",
    tag_text: str | None = None,
) -> np.ndarray:
    img = upscale(rgb)
    if mask is None:
        _tag(img, none_text, (255, 150, 60))
        return img
    m = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    mb = m.astype(bool)
    img[mb] = (img[mb] * (1 - alpha) + np.array(fill) * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, edge, 2, cv2.LINE_AA)
    if not mb.any():
        _tag(img, "object not visible (empty)")
    elif tag_text:
        _tag(img, tag_text, (255, 190, 80))
    return img


def _load_json(cell):
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
    roi_pts: list[tuple[float, float]] = []
    if mask is not None and mask.any():
        ys, xs = np.nonzero(mask)
        sc = rgb.shape[1] / mask.shape[1]  # mask may be higher-res than the frame
        roi_pts += [(xs.min() * sc, ys.min() * sc), (xs.max() * sc, ys.max() * sc)]
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


def _paste_zoom_inset(img: np.ndarray, roi_pts, zoom_h: int = 210) -> np.ndarray:
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


def wrist_column(base: str, wrist_frames, frames_idx) -> tuple[list[np.ndarray], str]:
    """Wrist tiles: SAM2 prediction overlay if available, else 'mask: none'."""
    pred = load_sam2_wrist(base)
    if pred is None:
        return [overlay_mask(w, None) for w in wrist_frames], "wrist"
    masks, tag = pred
    tiles = [
        overlay_mask(w, masks[min(f, len(masks) - 1)], fill=PRED_COLOR, edge=PRED_EDGE, tag_text=tag)
        for w, f in zip(wrist_frames, frames_idx)
    ]
    return tiles, "wrist + SAM2 pred"


def render_droid(base: str, pairs) -> None:
    info = pairs.get(base)
    if not info or len(info) < 2:
        print(f"!! droid base {base}: both exterior views not local, skip")
        return
    (idx1, name1), (idx2, name2) = info["1"], info["2"]
    m1, m2 = load_masks(name1), load_masks(name2)
    if m1 is None or m2 is None:
        print(f"!! droid base {base}: npz missing, skip")
        return
    ensure_videos("droid", [idx1, idx2])
    df1 = pq.read_table(DATASETS["droid"]["chunk_dir"] / f"episode_{idx1:06d}.parquet").to_pandas()
    task = str(df1.iloc[0]["annotation.instruction_add"]) or "(no instruction)"
    n = min(len(df1), m1.shape[0], m2.shape[0])
    frames_idx = sample_indices(n, 4)

    ext1 = decode_video_frames(video_path("droid", idx1, "primary"), frames_idx)
    ext2 = decode_video_frames(video_path("droid", idx2, "primary"), frames_idx)
    wrist = decode_video_frames(video_path("droid", idx1, "wrist"), frames_idx)
    wtiles, wlabel = wrist_column(base, wrist, frames_idx)

    rows = []
    for k, f in enumerate(frames_idx):
        skill = str(df1.iloc[f]["annotation.primitive_skill"]) or "-"
        rows.append((f"frame {f}", f"t={f / 10:.1f}s · {skill}",
                     [overlay_mask(ext1[k], m1[f]), overlay_mask(ext2[k], m2[f]), wtiles[k]]))
    out = build_montage(
        OUT_DIR / f"{base}_crossview_mask.png",
        f"RoboInter mask coverage · DROID episode {base}",
        f'"{task}"  ·  green = RoboInter GT (exterior only) · orange = SAM2 prediction (NOT official GT)',
        [f"exterior_1 (ep{idx1:06d}) + mask", f"exterior_2 (ep{idx2:06d}) + mask", f"{wlabel} (ep{idx1:06d})"],
        rows,
        footer=f"source: InternRobotics/RoboInter-Data lerobot_droid_anno · {name1} / {name2}",
        tile=(360, 203),
    )
    print(f"wrote {out}")

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
        footer="all 2D annotations live in the PRIMARY (exterior) camera pixel frame - policy-oriented, not cross-view",
        tile=(640, 360),
    )
    print(f"wrote {out2}")


def render_rh20t(take: str | None, takes: dict | None = None) -> None:
    takes = takes if takes is not None else rh20t_takes()
    if not takes:
        print("!! no RH20T takes with npz in local chunk")
        return
    if take in (None, "auto"):
        # prefer takes with >=2 annotated cams, earliest episodes (cheap tar extraction)
        cands = [(max(i for i, _ in cams), t) for t, cams in takes.items() if len(cams) >= 2]
        if not cands:
            print("!! no RH20T take has >=2 annotated cams locally")
            return
        take = min(cands)[1]
    cams = sorted(takes[take])[:2]
    (idxA, nameA), (idxB, nameB) = cams
    mA, mB = load_masks(nameA), load_masks(nameB)
    if mA is None or mB is None:
        print(f"!! RH20T take {take}: fetch npz first:")
        print(f"   python fetch_segmentation_npz.py {nameA} {nameB}")
        return
    ensure_videos("rh20t", [idxA, idxB])
    dfA = pq.read_table(DATASETS["rh20t"]["chunk_dir"] / f"episode_{idxA:06d}.parquet").to_pandas()
    task = str(dfA.iloc[0]["annotation.instruction_add"]) or "(no instruction)"
    n = min(len(dfA), mA.shape[0], mB.shape[0])
    frames_idx = sample_indices(n, 4)

    serA, serB = nameA.rsplit("_", 1)[1], nameB.rsplit("_", 1)[1]
    camA = decode_video_frames(video_path("rh20t", idxA, "primary"), frames_idx)
    camB = decode_video_frames(video_path("rh20t", idxB, "primary"), frames_idx)
    wrist = decode_video_frames(video_path("rh20t", idxA, "wrist"), frames_idx)
    short = take.replace("RH20T_cfg1_", "")
    wtiles, wlabel = wrist_column(short, wrist, frames_idx)

    rows = []
    for k, f in enumerate(frames_idx):
        skill = str(dfA.iloc[f]["annotation.primitive_skill"]) or "-"
        rows.append((f"frame {f}", f"t={f / 10:.1f}s · {skill}",
                     [overlay_mask(camA[k], mA[f]), overlay_mask(camB[k], mB[f]), wtiles[k]]))
    out = build_montage(
        OUT_DIR / f"{short}_crossview_mask.png",
        f"RoboInter mask coverage · RH20T {short}",
        f'"{task}"  ·  green = RoboInter GT (global cams, 360x640) · orange = SAM2 prediction (NOT official GT)',
        [f"global cam {serA} + mask", f"global cam {serB} + mask", f"{wlabel} (ep{idxA:06d})"],
        rows,
        footer=f"source: InternRobotics/RoboInter-Data lerobot_rh20t_anno · {nameA} / {nameB}",
        tile=(360, 203),
    )
    print(f"wrote {out}")


def main(argv: list[str]) -> None:
    if argv and argv[0] == "--rh20t":
        render_rh20t(argv[1] if len(argv) > 1 else "auto")
        return
    pairs = droid_pairs()
    for base in argv or ["10020", "10010", "0"]:
        render_droid(base, pairs)


if __name__ == "__main__":
    main(sys.argv[1:])
