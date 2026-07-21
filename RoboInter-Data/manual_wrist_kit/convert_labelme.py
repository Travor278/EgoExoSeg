"""Convert labelme polygon JSONs back to our mask format + render comparison cards.

Reads frames/{base}/f*.json (labelme, polygons labeled anything — all shapes count
as the target object), downscales to the native 320x180, writes
manual_masks/{base}_wrist_manual.npz (masks (T,H,W) bool filled at annotated
frames), then renders samples/{base}_manual_vs_sam2.png:
    exterior_1 + RoboInter GT | wrist + SAM2 auto (orange) | wrist + MANUAL (blue)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
RID = ROOT.parent
sys.path.insert(0, str(RID.parent))
sys.path.insert(0, str(RID))
from crossview_viz import build_montage, decode_video_frames  # noqa: E402
import visualize_robointer_masks as viz  # noqa: E402

MANUAL_COLOR, MANUAL_EDGE = (60, 130, 255), (120, 180, 255)  # blue-ish


def polygons_to_mask(shapes, scale: int, h: int = 180, w: int = 320) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    for s in shapes:
        pts = np.asarray(s["points"], np.float32) / scale
        cv2.fillPoly(m, [pts.astype(np.int32)], 1)
    return m.astype(bool)


def convert(base: str):
    d = ROOT / "frames" / base
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    T, scale = meta["total_frames"], meta["scale"]
    full = np.zeros((T, 180, 320), bool)
    n = 0
    for jf in sorted(d.glob("f*.json")):
        f_idx = int(jf.stem[1:])
        data = json.loads(jf.read_text(encoding="utf-8"))
        full[f_idx] = polygons_to_mask(data.get("shapes", []), scale)
        n += 1
    out = ROOT / "manual_masks"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / f"{base}_wrist_manual.npz", masks=full, tag=np.array("manual-labelme"))
    print(f"{base}: {n} annotated frames -> {out / (base + '_wrist_manual.npz')}")
    return meta, full, n


def render(base: str, meta: dict, manual: np.ndarray) -> None:
    pairs = viz.droid_pairs()
    info = pairs.get(base)
    if not info or len(info) < 2:
        print(f"{base}: exterior pair not local, skip card")
        return
    (i1, n1), _ = info["1"], info["2"]
    gt = viz.load_masks(n1)
    sam2 = viz.load_sam2_wrist(base)
    frames_idx = meta["frames"]
    ext1 = decode_video_frames(str(viz.video_path("droid", i1, "primary")), frames_idx)
    wrist = decode_video_frames(str(viz.video_path("droid", meta["episode_index"], "wrist")), frames_idx)
    rows = []
    for k, f in enumerate(frames_idx):
        tiles = [viz.overlay_mask(ext1[k], gt[f] if gt is not None else None)]
        if sam2 is not None:
            tiles.append(viz.overlay_mask(wrist[k], sam2[0][min(f, len(sam2[0]) - 1)],
                                          fill=viz.PRED_COLOR, edge=viz.PRED_EDGE, tag_text=sam2[1]))
        else:
            tiles.append(viz.overlay_mask(wrist[k], None))
        tiles.append(viz.overlay_mask(wrist[k], manual[f], fill=MANUAL_COLOR, edge=MANUAL_EDGE,
                                      tag_text="manual (labelme)"))
        rows.append((f"frame {f}", "", tiles))
    out = viz.OUT_DIR / f"{base}_manual_vs_sam2.png"
    build_montage(
        out,
        f"Wrist manual annotation test · DROID {base}",
        f'"{meta["instruction"]}"  ·  green = exterior GT · orange = SAM2 auto · blue = MANUAL labelme',
        ["exterior_1 + GT", "wrist + SAM2 auto", "wrist + MANUAL"],
        rows,
        footer="manual_wrist_kit: 6 frames/episode hand-polygon in labelme; compare boundary quality vs SAM2 auto",
        tile=(360, 203),
    )
    print("wrote", out)


def main() -> None:
    bases = [p.name for p in (ROOT / "frames").iterdir() if p.is_dir()]
    for base in bases:
        if not list((ROOT / "frames" / base).glob("f*.json")):
            print(f"{base}: no labelme jsons yet, skip")
            continue
        meta, manual, n = convert(base)
        if n:
            render(base, meta, manual)


if __name__ == "__main__":
    main()
