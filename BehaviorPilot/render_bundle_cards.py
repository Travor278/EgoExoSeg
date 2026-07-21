"""Render sim cross-view GT cards from the remote-prepared bundle/ directory.

bundle/{demo}/{view}/: f*.jpg (sampled frames), m*.npy (target masks), meta.json
One card per demo; columns = available views, rows = the 4 sampled steps.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from crossview_viz import build_montage  # noqa: E402

sys.path.insert(0, str(ROOT.parent / "RoboInter-Data"))
from visualize_robointer_masks import overlay_mask  # noqa: E402

DEMO_TASK = {"1550": "turning_on_radio", "50220": "setting_mousetraps",
             "71020": "picking_up_toys", "42750": "can_meat"}
VIEW_ORDER = ["exo0", "exo1", "exo2", "head", "head_gaze", "left_wrist", "right_wrist"]
LABEL = {"exo0": "exo0 (auto)", "exo1": "exo1 (auto)", "exo2": "exo2 (auto)",
         "head": "ego head", "head_gaze": "ego head (gaze)",
         "left_wrist": "left wrist", "right_wrist": "right wrist"}


def render_demo(demo_dir: Path) -> None:
    demo = demo_dir.name
    views = [v for v in VIEW_ORDER if (demo_dir / v / "meta.json").exists()]
    if not views:
        return
    metas = {v: json.loads((demo_dir / v / "meta.json").read_text()) for v in views}
    idxs = metas[views[0]]["idxs"]
    target = metas[views[0]]["target"].get("name", "?")

    cols, names = [], []
    for v in views:
        d = demo_dir / v
        tiles = []
        for t in idxs:
            jpg, npy = d / f"f{t:05d}.jpg", d / f"m{t:05d}.npy"
            if not jpg.exists():
                tiles.append(np.zeros((180, 320, 3), np.uint8))
                continue
            rgb = cv2.cvtColor(cv2.imread(str(jpg)), cv2.COLOR_BGR2RGB)
            small = cv2.resize(rgb, (rgb.shape[1] // 2, rgb.shape[0] // 2), interpolation=cv2.INTER_AREA)
            mask = np.load(npy) if npy.exists() else None
            tiles.append(overlay_mask(small, mask))
        m = metas[v]
        cols.append(tiles)
        names.append(f"{LABEL[v]} · vis {m['visible_frac']:.0%}")
    rows = [(f"step {t}", f"t={t / 60:.1f}s", [cols[c][k] for c in range(len(views))])
            for k, t in enumerate(idxs)]
    task = DEMO_TASK.get(demo, "?")
    out = ROOT / "samples" / f"{demo}_{task}_crossview_gt.png"
    build_montage(
        out,
        f"BEHAVIOR-1K sim · {task} · demo {demo}",
        f"target = {target} · renderer GT instance masks (green), zero annotation · target-aimed ring cams (v2.2)",
        names, rows,
        footer="viewer-camera multi-pass replay (OG#2312 workaround) · per-column vis = fraction of frames with target visible",
        tile=(300, 180),
    )
    print("wrote", out)


def main() -> None:
    for demo_dir in sorted((ROOT / "bundle").iterdir()):
        if demo_dir.is_dir():
            render_demo(demo_dir)


if __name__ == "__main__":
    main()
