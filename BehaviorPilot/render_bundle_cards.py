"""Render sim cross-view GT cards from the remote-prepared bundle/ directory.

bundle/{demo}/{view}/: f*.jpg (sampled frames), m*.npy (target masks), meta.json
One card per demo; columns = available views, rows = the 4 sampled steps.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
# Override to keep a production run's output separate from the historical
# bundle/ + samples/ pair, e.g. PILOT_BUNDLE=autodl_run40/bundle
BUNDLE = Path(os.environ.get("PILOT_BUNDLE") or ROOT / "bundle")
CARDS = Path(os.environ.get("PILOT_CARDS") or ROOT / "samples")
# Columns whose target is visible in fewer than this fraction of frames are
# dropped as dead. Set PILOT_MIN_VIS=0 to keep the full 5-column recipe even when
# a view barely sees the target — the frames are always in the bundle either way.
MIN_VIS = float(os.environ.get("PILOT_MIN_VIS", "0.05"))
sys.path.insert(0, str(ROOT.parent))
from crossview_viz import build_montage  # noqa: E402

sys.path.insert(0, str(ROOT.parent / "RoboInter-Data"))
from visualize_robointer_masks import overlay_mask  # noqa: E402

DEMO_TASK = {"1550": "turning_on_radio", "50220": "setting_mousetraps",
             "71020": "picking_up_toys", "42750": "can_meat",
             "720270": "cook_a_frozen_pie", "822240": "store_batteries",
             "942050": "dispose_of_batteries", "761590": "dispose_of_glass",
             "780": "turning_on_radio", "21060": "putting_away_Halloween_decorations"}


def task_name(demo: str) -> str:
    """Known task name, else the task id the demo belongs to (demo // 10000)."""
    return DEMO_TASK.get(demo, f"task-{int(demo) // 10000:04d}")
VIEW_ORDER = ["exo0", "exo1", "exo2", "exo3", "exo4", "exo5", "exo6",
              "head", "head_gaze", "left_wrist", "right_wrist"]
# RING mode can render up to exo6, so look labels up with a fallback rather
# than indexing — a missing entry would KeyError at card time, after the
# expensive rendering is already done.
LABEL = {"exo0": "exo0 (auto)", "exo1": "exo1 (auto)", "exo2": "exo2 (auto)",
         "exo3": "exo3 (auto)", "exo4": "exo4 (auto)",
         "exo5": "exo5 (auto)", "exo6": "exo6 (auto)",
         "head": "ego head", "head_gaze": "ego head (gaze)",
         "left_wrist": "left wrist", "right_wrist": "right wrist"}


def render_demo(demo_dir: Path) -> None:
    demo = demo_dir.name
    views = [v for v in VIEW_ORDER if (demo_dir / v / "meta.json").exists()]
    if not views:
        return
    metas = {v: json.loads((demo_dir / v / "meta.json").read_text()) for v in views}
    # drop dead columns (target basically never visible) unless that leaves too few
    live = [v for v in views if metas[v]["visible_frac"] >= MIN_VIS]
    if len(live) >= 2:
        views = live
    # RING mode renders 6-7 exo cameras and defers the choice to here, where the
    # only signal that has never mispredicted is available: measured pixels. Keep
    # the two best-covered exos (angle-diverse by construction) and drop the rest,
    # so the card still reads as "two static exo views" rather than a contact sheet.
    # Rank on median target pixels, not visible_frac. 771920's exo0 saw the target
    # in 87% of frames but at 341px — technically "visible", too small to read —
    # while exo1 had 100% at 2296px. Sorting on coverage alone kept the useless one.
    # visible_frac only breaks ties now.
    exos = sorted((v for v in views if v.startswith("exo")),
                  key=lambda v: (-metas[v].get("median_px", 0), -metas[v]["visible_frac"]))
    # Never promote a blind camera just because it ranked second. The ring is placed
    # without a line-of-sight check, so on cramped scenes most of its 8 angles sit
    # inside walls (191980: exo0 had 2268 target px, the other seven had 0). One real
    # exo column beats one real column plus a white wall.
    exos = [v for v in exos if metas[v].get("median_px", 0) > 0][:2]
    views = [v for v in views if not v.startswith("exo") or v in exos]
    views.sort(key=VIEW_ORDER.index)
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
        names.append(f"{LABEL.get(v, v)} · vis {m['visible_frac']:.0%}")
    rows = [(f"step {t}", f"t={t / 60:.1f}s", [cols[c][k] for c in range(len(views))])
            for k, t in enumerate(idxs)]
    task = task_name(demo)
    CARDS.mkdir(parents=True, exist_ok=True)
    out = CARDS / f"{demo}_{task}_crossview_gt.png"
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
    # optional demo ids on the command line; default = every bundled demo
    wanted = set(sys.argv[1:])
    for demo_dir in sorted(BUNDLE.iterdir()):
        if demo_dir.is_dir() and (not wanted or demo_dir.name in wanted):
            render_demo(demo_dir)


if __name__ == "__main__":
    main()
