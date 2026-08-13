"""Prepare a local, frame-by-frame review page for the rendered cards.

For every bundle this writes the ego view plus the two exo views the exporter would
pick, and for each sampled frame a mask overlay: a transparent PNG that is coloured
only where the target is. The browser composites it over the JPEG, so the reviewer
can fade the mask in and out and judge the annotation rather than just the framing.
Overlays are separate files instead of being burned into the JPEG precisely so that
fading is possible, and a two-colour PNG of a blob costs a few KB.

Outputs into OUT_DIR:
    data.js              window.REVIEW = {cards: [...]}  (a .js file, not .json,
                         because file:// pages cannot fetch())
    m/<card>/<view>/<t>.png   mask overlays
    index.html           copied from review_page.html

Images are referenced back into the bundle by relative path, so nothing is copied.

Usage:
    PILOT_BUNDLE=BehaviorPilot/run_v5/bundle python build_review.py OUT_DIR
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

BUNDLE = Path(os.environ.get("PILOT_BUNDLE") or Path(__file__).resolve().parent / "run_v5" / "bundle")
EGO_VIEWS = ("head_gaze", "head")
MIN_VIS = float(os.environ.get("PILOT_MIN_VIS", "0.05"))
MASK_RGB = (255, 64, 160)


def pretty(name: str) -> str:
    stem = re.sub(r"_\d+$", "", name.split(".")[0])
    return stem.replace("__", " ").replace("_", " ").strip()


def pick_views(d: Path) -> dict | None:
    """Ego + the two exos with the most target pixels — the exporter's own choice."""
    views = {}
    for v in sorted(p.name for p in d.iterdir() if p.is_dir()):
        meta = d / v / "meta.json"
        if meta.exists():
            views[v] = json.loads(meta.read_text())
    if not views:
        return None
    ego = next((v for v in EGO_VIEWS if v in views and views[v]["visible_frac"] >= MIN_VIS), None)
    # median_px > 0 drops the blank-wall cameras: the exo ring is placed without a
    # line-of-sight check, so on cramped scenes most angles render a wall.
    exos = sorted((v for v in views
                   if v.startswith("exo") and views[v]["visible_frac"] >= MIN_VIS
                   and views[v].get("median_px", 0) > 0),
                  key=lambda v: (-views[v].get("median_px", 0), -views[v]["visible_frac"]))[:2]
    if not ego or not exos:
        return None
    return {"ego": ego, "exos": exos, "views": views}


def write_overlay(mask_path: Path, out_png: Path) -> tuple[int, list | None]:
    """Transparent PNG coloured only where the mask is.

    Returns (pixel count, bbox) where bbox is [cx, cy, w, h] in FRACTIONS of the
    frame. The page needs it to zoom: targets here are routinely a few dozen pixels
    inside 1280x720 (a drill at 34 px), and at fit-to-width that is a dot no reviewer
    can judge. Fractions rather than pixels so the page can scale without knowing the
    source resolution.
    """
    m = np.load(mask_path)
    n = int(m.sum())
    if n == 0:
        return 0, None
    rgba = np.zeros((*m.shape, 4), np.uint8)
    rgba[m] = (*MASK_RGB, 255)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(out_png, optimize=True)

    ys, xs = np.nonzero(m)
    h, w = m.shape
    bbox = [round(float((xs.min() + xs.max()) / 2 / w), 4),
            round(float((ys.min() + ys.max()) / 2 / h), 4),
            round(float((xs.max() - xs.min() + 1) / w), 4),
            round(float((ys.max() - ys.min() + 1) / h), 4)]
    return n, bbox


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "BehaviorPilot/run_v5/review")
    out.mkdir(parents=True, exist_ok=True)
    bundle_rel = os.path.relpath(BUNDLE.resolve(), out.resolve()).replace("\\", "/")

    cards, skipped = [], 0
    for d in sorted(p for p in BUNDLE.iterdir() if p.is_dir()):
        info = pick_views(d)
        if not info:
            skipped += 1
            continue
        vlist = [info["ego"], *info["exos"]]
        idxs = info["views"][info["ego"]]["idxs"]
        frames = []
        for t in idxs:
            per_view = []
            for v in vlist:
                jpg = d / v / f"f{t:05d}.jpg"
                npy = d / v / f"m{t:05d}.npy"
                png = out / "m" / d.name / v / f"{t}.png"
                px, bbox = write_overlay(npy, png) if npy.exists() else (0, None)
                per_view.append({
                    "img": f"{bundle_rel}/{d.name}/{v}/f{t:05d}.jpg" if jpg.exists() else None,
                    "mask": f"m/{d.name}/{v}/{t}.png" if px else None,
                    "px": px,
                    "bbox": bbox,
                })
            frames.append({"t": t, "views": per_view})

        tgt = info["views"][info["ego"]]["target"]
        areas = [fv["px"] for f in frames for fv in f["views"] if fv["px"] > 0]
        cards.append({
            "id": d.name,
            "target": pretty(tgt.get("name", "?")),
            "target_raw": tgt.get("name", "?"),
            "views": vlist,
            "stats": [{"view": v,
                       "vis": round(info["views"][v]["visible_frac"], 3),
                       "median_px": info["views"][v].get("median_px", 0)} for v in vlist],
            # the weaker side of a pair is what limits its usefulness, so surface the
            # minimum rather than the mean -- a 3000px ego with a 13px exo is unusable
            "min_px": min(areas) if areas else 0,
            "frames": frames,
        })

    (out / "data.js").write_text(
        "window.REVIEW = " + json.dumps({"cards": cards}, ensure_ascii=False) + ";\n",
        encoding="utf-8")
    page = Path(__file__).resolve().parent / "review_page.html"
    if page.exists():
        shutil.copyfile(page, out / "index.html")

    print(f"{len(cards)} cards, {skipped} skipped")
    print(f"frames: {sum(len(c['frames']) for c in cards)}  "
          f"overlays: {sum(1 for c in cards for f in c['frames'] for v in f['views'] if v['mask'])}")
    print(f"-> {out / 'index.html'}")


if __name__ == "__main__":
    main()
