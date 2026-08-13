"""Export rendered bundles into the Ego-Exo4D-Relation layout.

Target format (jaychempan/Ego-Exo4D-Relation-Test):
    data/<video_id>/<camera>/<frame>.jpg
    ego2exo.json   ego view is the prompt, exo view is the target
    exo2ego.json   the other direction

Each record:
    {"<video_id>": {
        "video_id": ...,
        "video_path": ["<vid>/<target_cam>/<frame>.jpg", ...],
        "prompt": {"first_frame_image": "<vid>/<src_cam>/<frame>.jpg",
                   "first_frame_anns": {"0": {"segmentation": {"counts","size"},
                                              "area", "category_id", "text"}}},
        "objects": {"0": {... , "segmentation": {...}}}}}

Masks are COCO RLE at the ORIGINAL render resolution; `size` records the true
[h, w] rather than resizing to the reference dataset's 704x704.

Per demo we keep the ego view plus the two exo views with the most target pixels,
which is the selection the cards already use, and emit every (ego, exo) pair in
both directions.

Usage:
    PILOT_BUNDLE=BehaviorPilot/run_v5/bundle python export_egoexo_relation.py OUT_DIR
"""
from __future__ import annotations

import itertools
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as coco_mask

BUNDLE = Path(os.environ.get("PILOT_BUNDLE") or Path(__file__).resolve().parent / "run_v5" / "bundle")
EGO_VIEWS = ("head_gaze", "head")
MIN_VIS = float(os.environ.get("PILOT_MIN_VIS", "0.05"))


def pretty(name: str) -> str:
    """'digital_camera.n.01' / 'digital_camera_87' -> 'digital camera'."""
    stem = name.split(".")[0]
    stem = re.sub(r"_\d+$", "", stem)
    return stem.replace("__", " ").replace("_", " ").strip()


def rle_of(mask_path: Path) -> tuple[dict, float] | None:
    """COCO RLE + area for a boolean mask, or None if the object is absent."""
    if not mask_path.exists():
        return None
    m = np.load(mask_path)
    if not m.any():
        return None
    # pycocotools wants Fortran-order uint8; counts come back as bytes
    rle = coco_mask.encode(np.asfortranarray(m.astype(np.uint8)))
    return ({"counts": rle["counts"].decode("ascii"), "size": [int(m.shape[0]), int(m.shape[1])]},
            float(coco_mask.area(rle)))


def load_demo(d: Path) -> dict | None:
    views = {}
    for v in sorted(p.name for p in d.iterdir() if p.is_dir()):
        meta_p = d / v / "meta.json"
        if meta_p.exists():
            views[v] = json.loads(meta_p.read_text())
    if not views:
        return None
    ego = next((v for v in EGO_VIEWS if v in views and views[v]["visible_frac"] >= MIN_VIS), None)
    # median_px > 0 is required, not just some visibility. A camera can see the target
    # in 43-47% of frames and still have a median of ZERO pixels -- catching a few
    # stray pixels past a counter edge rather than the object. Four exo layouts were
    # measured (blind ring, room-clamped ring, room corners, raised ring) and all land
    # on ~2 usable of 8: on these scenes a small object on a counter simply has two or
    # three directions that are close, unoccluded and not grazing.
    exos = sorted((v for v in views
                   if v.startswith("exo") and views[v]["visible_frac"] >= MIN_VIS
                   and views[v].get("median_px", 0) > 0),
                  key=lambda v: (-views[v].get("median_px", 0), -views[v]["visible_frac"]))[:2]
    # An exo-exo pair needs two exos and no ego at all, so a card with no usable ego
    # is still worth exporting -- which the old "needs one of each" rejected outright.
    if not exos or (not ego and len(exos) < 2):
        return None
    idx_src = ego or exos[0]
    return {"ego": ego, "exos": exos, "views": views, "idxs": views[idx_src]["idxs"],
            "target": views[idx_src]["target"].get("name", "?")}


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "BehaviorPilot/run_v5/export")
    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    ego2exo: dict = {}
    exo2ego: dict = {}
    # exo-exo is the larger half of this dataset. Counted over 177 cards there are 639
    # exo-exo pairs against 155 ego-exo, because a card typically has two usable exos
    # and an ego that sees the target far less often (the head camera is level and only
    # steered within a 45 degree cone). It also needs no new rendering: the exo views
    # are already on disk.
    exo2exo: dict = {}
    n_pairs = n_skip = 0

    for d in sorted(BUNDLE.iterdir()):
        if not d.is_dir():
            continue
        info = load_demo(d)
        if not info:
            n_skip += 1
            continue
        vid = d.name
        text = pretty(info["target"])

        for cam in [c for c in [info["ego"], *info["exos"]] if c]:
            dst = data_dir / vid / cam
            dst.mkdir(parents=True, exist_ok=True)
            for t in info["idxs"]:
                src = d / cam / f"f{t:05d}.jpg"
                if src.exists():
                    shutil.copyfile(src, dst / f"{t}.jpg")

        # Every directed pair this card supports: ego<->each exo, and exo<->exo.
        directions = []
        if info["ego"]:
            for exo in info["exos"]:
                directions.append((info["ego"], exo, ego2exo, "ego2exo"))
                directions.append((exo, info["ego"], exo2ego, "exo2ego"))
        for a, b in itertools.combinations(info["exos"], 2):
            directions.append((a, b, exo2exo, "exo2exo"))
            directions.append((b, a, exo2exo, "exo2exo"))

        for src_cam, tgt_cam, table, tag in directions:
            # the prompt frame is the first sampled step where BOTH views see it
            pair_id = f"{vid}_{src_cam}__{tgt_cam}"
            prompt_t = tgt_t = None
            for t in info["idxs"]:
                s = rle_of(d / src_cam / f"m{t:05d}.npy")
                g = rle_of(d / tgt_cam / f"m{t:05d}.npy")
                if s and g:
                    prompt_t, tgt_t, s_rle, g_rle = t, t, s, g
                    break
            if prompt_t is None:
                n_skip += 1
                continue
            table[pair_id] = {
                "video_id": pair_id,
                "video_path": [f"{vid}/{tgt_cam}/{t}.jpg" for t in info["idxs"]],
                "prompt": {
                    "first_frame_image": f"{vid}/{src_cam}/{prompt_t}.jpg",
                    "first_frame_anns": {"0": {"segmentation": s_rle[0], "area": s_rle[1],
                                               "category_id": 1.0, "text": text}},
                },
                "objects": {"0": {
                    "video_id": pair_id, "obj_id": 0,
                    "crop_caption": text, "crop_category": text, "image_caption": text,
                    "video_caption": text, "final_caption": text,
                    "video_path": [f"{vid}/{tgt_cam}/{tgt_t}.jpg"],
                    "segmentation": g_rle[0], "area": g_rle[1],
                }},
            }
            n_pairs += 1

    (out / "ego2exo.json").write_text(json.dumps(ego2exo, indent=1))
    (out / "exo2ego.json").write_text(json.dumps(exo2ego, indent=1))
    (out / "exo2exo.json").write_text(json.dumps(exo2exo, indent=1))
    print(f"{n_pairs} pairs ({len(ego2exo)} ego2exo, {len(exo2ego)} exo2ego, "
          f"{len(exo2exo)} exo2exo), {n_skip} skipped")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
