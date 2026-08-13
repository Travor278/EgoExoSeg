"""Export only the cards and frames a reviewer kept, in Ego-Exo4D-Relation layout.

The review page stores verdicts per (card, frame) in the browser, so this takes the
JSON its export button produces:

    {"cards": [{"id": ..., "card": "keep"|"drop",
                "drop_frames": [t, ...], "add_frames": [t, ...]}, ...]}

Two review styles are supported, because the actual one turned out to be the
opposite of what was first assumed:

  * POSITIVE selection (what was used): keep_frames lists the frames the reviewer
    marked good. When a card has any, ONLY those frames are exported -- the rest of
    the card was seen and not chosen.
  * NEGATIVE selection: no keep_frames, drop_frames lists the rejects, everything
    else on the card is kept.

Other rules:
  * card == "drop"   -> the whole card is excluded
  * a card with no verdict at all is EXCLUDED by default; "not yet looked at" is not
    "approved". Pass --include-unreviewed to change that.
  * add_frames are frames picked out of the scrub video that do not exist in the
    bundle yet. They are reported and skipped here; materialising them needs the
    source videos, which is a render-side job.

Usage:
    python apply_review.py DECISIONS.json BUNDLE_DIR OUT_DIR [--include-unreviewed]
"""
from __future__ import annotations

import itertools
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as coco_mask

EGO_VIEWS = ("head_gaze", "head")
MIN_VIS = 0.05


def pretty(name: str) -> str:
    return re.sub(r"_\d+$", "", name.split(".")[0]).replace("__", " ").replace("_", " ").strip()


def rle_of(p: Path):
    if not p.exists():
        return None
    m = np.load(p)
    if not m.any():
        return None
    r = coco_mask.encode(np.asfortranarray(m.astype(np.uint8)))
    return ({"counts": r["counts"].decode("ascii"), "size": [int(m.shape[0]), int(m.shape[1])]},
            float(coco_mask.area(r)))


def pick(d: Path):
    views = {}
    for v in sorted(p.name for p in d.iterdir() if p.is_dir()):
        m = d / v / "meta.json"
        if m.exists():
            views[v] = json.loads(m.read_text())
    if not views:
        return None
    ego = next((v for v in EGO_VIEWS if v in views and views[v]["visible_frac"] >= MIN_VIS), None)
    exos = sorted((v for v in views if v.startswith("exo")
                   and views[v]["visible_frac"] >= MIN_VIS and views[v].get("median_px", 0) > 0),
                  key=lambda v: (-views[v].get("median_px", 0), -views[v]["visible_frac"]))[:2]
    if not exos or (not ego and len(exos) < 2):
        return None
    ref = ego or exos[0]
    return {"ego": ego, "exos": exos, "views": views,
            "idxs": views[ref]["idxs"], "target": views[ref]["target"].get("name", "?")}


def main() -> None:
    dec_p, bundle_p, out_p = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    include_unreviewed = "--include-unreviewed" in sys.argv
    # Cards live on two hosts and the page distinguishes them by a z_ prefix, so the
    # SAME demo can appear twice as two different cards with different verdicts.
    # Each host exports only the ids carrying its own prefix, and keeps that prefix in
    # the output so the two halves can be merged without colliding.
    prefix = ""
    if "--prefix" in sys.argv:
        prefix = sys.argv[sys.argv.index("--prefix") + 1]

    dec = json.loads(dec_p.read_text(encoding="utf-8"))
    verdict = {c["id"]: c for c in dec.get("cards", [])}
    print(f"decisions: {len(verdict)} cards carry a verdict")

    data = out_p / "data"
    data.mkdir(parents=True, exist_ok=True)
    tables = {"ego2exo": {}, "exo2ego": {}, "exo2exo": {}}
    n_pairs = 0
    kept_cards = dropped_cards = skipped = 0
    dropped_frames = n_selected = pending_add = 0

    for d in sorted(p for p in bundle_p.iterdir() if p.is_dir()):
        card_id = f"{prefix}{d.name}"
        v = verdict.get(card_id)
        if v is None and not include_unreviewed:
            skipped += 1
            continue
        if v and v.get("card") == "drop":
            dropped_cards += 1
            continue
        info = pick(d)
        if not info:
            skipped += 1
            continue
        keep = set(v.get("keep_frames", [])) if v else set()
        drop = set(v.get("drop_frames", [])) if v else set()
        if keep:
            # Positive selection: the marked frames ARE the card.
            idxs = [t for t in info["idxs"] if t in keep]
            n_selected += len(idxs)
        else:
            idxs = [t for t in info["idxs"] if t not in drop]
        dropped_frames += len(info["idxs"]) - len(idxs)
        pending_add += len(v.get("add_frames", [])) if v else 0
        if not idxs:
            dropped_cards += 1
            continue
        kept_cards += 1

        vid, text = card_id, pretty(info["target"])
        for cam in [c for c in [info["ego"], *info["exos"]] if c]:
            dst = data / vid / cam
            dst.mkdir(parents=True, exist_ok=True)
            for t in idxs:
                src = d / cam / f"f{t:05d}.jpg"
                if src.exists():
                    shutil.copyfile(src, dst / f"{t}.jpg")

        directions = []
        if info["ego"]:
            for exo in info["exos"]:
                directions += [(info["ego"], exo, "ego2exo"), (exo, info["ego"], "exo2ego")]
        for a, b in itertools.combinations(info["exos"], 2):
            directions += [(a, b, "exo2exo"), (b, a, "exo2exo")]

        for src_cam, tgt_cam, tag in directions:
            pair_id = f"{vid}_{src_cam}__{tgt_cam}"
            hit = None
            for t in idxs:
                s = rle_of(d / src_cam / f"m{t:05d}.npy")
                g = rle_of(d / tgt_cam / f"m{t:05d}.npy")
                if s and g:
                    hit = (t, s, g)
                    break
            if not hit:
                continue
            t, (s_rle, s_area), (g_rle, g_area) = hit
            tables[tag][pair_id] = {
                "video_id": pair_id,
                "video_path": [f"{vid}/{tgt_cam}/{x}.jpg" for x in idxs],
                "prompt": {"first_frame_image": f"{vid}/{src_cam}/{t}.jpg",
                           "first_frame_anns": {"0": {"segmentation": s_rle, "area": s_area,
                                                      "category_id": 1.0, "text": text}}},
                "objects": {"0": {"video_id": pair_id, "obj_id": 0,
                                  "crop_caption": text, "crop_category": text,
                                  "image_caption": text, "video_caption": text,
                                  "final_caption": text,
                                  "video_path": [f"{vid}/{tgt_cam}/{t}.jpg"],
                                  "segmentation": g_rle, "area": g_area}},
            }
            n_pairs += 1

    for tag, tbl in tables.items():
        (out_p / f"{tag}.json").write_text(json.dumps(tbl, indent=1))

    print(f"cards kept {kept_cards}, dropped {dropped_cards}, "
          f"skipped (no verdict / unusable) {skipped}")
    print(f"frames selected by the reviewer: {n_selected}")
    print(f"frames not selected / dropped   : {dropped_frames}")
    if pending_add:
        print(f"frames picked from video but NOT yet rendered: {pending_add}"
              f"  (run the add-frame extractor to include them)")
    for tag, tbl in tables.items():
        print(f"  {tag:9} {len(tbl)} pairs")
    print(f"TOTAL PAIRS: {n_pairs}")
    print(f"-> {out_p}")


if __name__ == "__main__":
    main()
