"""Visualize what the labelme line buys us: sparse human anchors -> dense masks.

Per episode, 4 rows: 2 ANCHOR frames (human-labeled) + 2 BETWEEN-anchor frames
(no human input there — pure propagation). Columns:
  exterior_1 + RoboInter GT | wrist + OLD auto SAM2 (orange) | wrist + PROPAGATED (blue)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RID = ROOT.parent
sys.path.insert(0, str(RID.parent))
sys.path.insert(0, str(RID))
from crossview_viz import build_montage, decode_video_frames  # noqa: E402
import visualize_robointer_masks as viz  # noqa: E402

BLUE, BLUE_E = (60, 130, 255), (120, 180, 255)


def render(base: str) -> None:
    meta = json.loads((ROOT / "frames" / base / "meta.json").read_text(encoding="utf-8"))
    prop = np.load(ROOT / "propagated" / f"{base}_wrist_prop.npz")["masks"]
    manual = np.load(ROOT / "manual_masks" / f"{base}_wrist_manual.npz")["masks"]
    anchors = [f for f in meta["frames"] if manual[f].any()]
    if len(anchors) < 2:
        return
    mids = [(anchors[i] + anchors[i + 1]) // 2 for i in range(len(anchors) - 1)]
    rows_idx = [(anchors[0], "anchor (human)"), (mids[0], "between anchors"),
                (mids[len(mids) // 2], "between anchors"), (anchors[-1], "anchor (human)")]

    pairs = viz.droid_pairs()
    info = pairs.get(base)
    gt = viz.load_masks(info["1"][1]) if info and "1" in info else None
    i1 = info["1"][0] if info and "1" in info else None
    sam2 = viz.load_sam2_wrist(base)
    frames_idx = [f for f, _ in rows_idx]
    ext1 = decode_video_frames(str(viz.video_path("droid", i1, "primary")), frames_idx) if i1 is not None else None
    wrist = decode_video_frames(str(viz.video_path("droid", meta["episode_index"], "wrist")), frames_idx)

    rows = []
    for k, (f, tag) in enumerate(rows_idx):
        tiles = []
        tiles.append(viz.overlay_mask(ext1[k], gt[f] if gt is not None else None) if ext1 else
                     np.zeros((180, 320, 3), np.uint8))
        if sam2 is not None:
            tiles.append(viz.overlay_mask(wrist[k], sam2[0][min(f, len(sam2[0]) - 1)],
                                          fill=viz.PRED_COLOR, edge=viz.PRED_EDGE, tag_text="old auto SAM2"))
        else:
            tiles.append(viz.overlay_mask(wrist[k], None))
        tiles.append(viz.overlay_mask(wrist[k], prop[f], fill=BLUE, edge=BLUE_E,
                                      tag_text="anchors+propagate"))
        rows.append((f"frame {f}", tag, tiles))
    out = viz.OUT_DIR / f"{base}_pipeline_compare.png"
    build_montage(
        out,
        f"Sparse human anchors -> dense wrist masks · DROID {base}",
        f'"{meta["instruction"]}"  ·  orange = old auto SAM2 · blue = manual-anchor propagation (production pipeline)',
        ["exterior_1 + GT", "wrist + auto SAM2", "wrist + anchor-propagated"],
        rows,
        footer="rows marked 'between anchors' had NO human input - dense quality from ~5 clicks/episode is the payoff",
        tile=(360, 203),
    )
    print("wrote", out)


if __name__ == "__main__":
    for b in (sys.argv[1:] or ["10427", "0", "1000"]):
        try:
            render(b)
        except Exception as e:
            print(b, "failed:", type(e).__name__, e)
