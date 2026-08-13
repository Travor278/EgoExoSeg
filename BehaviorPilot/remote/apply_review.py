"""Apply review_decisions.json back onto the bundles.

The page can only record intent: it shows downscaled video with the mask burned in,
which is fine for choosing moments but is not data. This script does the real work on
the host, where rgb.mp4 and seg.h5 still are:

  card=drop      remove the whole bundle directory
  drop_frames    delete that step's jpg/npy in every view and drop it from meta idxs
  add_frames     decode that step at FULL resolution and rebuild its mask from seg.h5,
                 then insert it into meta idxs

Usage (on the host, in the behavior env):
    python apply_review.py review_decisions.json [PILOT_ROOT] [--dry-run]
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

args = [a for a in sys.argv[1:] if not a.startswith("--")]
DRY = "--dry-run" in sys.argv
DEC = Path(args[0])
ROOT = Path(args[1]) if len(args) > 1 else Path.home() / "pilot_v5"
BUNDLE, VIEWS = ROOT / "bundle", ROOT / "views"


def decode_at(mp4: Path, want: set[int]) -> dict[int, np.ndarray]:
    import av
    out = {}
    if not want or not mp4.exists():
        return out
    hi = max(want)
    with av.open(str(mp4)) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i in want:
                out[i] = fr.to_ndarray(format="rgb24")
            if i >= hi:
                break
    return out


def main() -> None:
    dec = json.loads(DEC.read_text(encoding="utf-8"))
    n_drop_card = n_drop_frame = n_add = n_miss = 0

    for entry in dec.get("cards", []):
        cid = entry["id"]
        d = BUNDLE / cid
        if not d.is_dir():
            print(f"  MISSING {cid}")
            n_miss += 1
            continue

        if entry.get("card") == "drop":
            print(f"  DROP CARD {cid}")
            if not DRY:
                shutil.rmtree(d)
            n_drop_card += 1
            continue

        views = [p.name for p in d.iterdir() if p.is_dir()]
        drop = set(entry.get("drop_frames") or [])
        add = sorted(set(entry.get("add_frames") or []))
        base = cid.split("__")[0]

        for v in views:
            meta_p = d / v / "meta.json"
            if not meta_p.exists():
                continue
            meta = json.loads(meta_p.read_text())
            idxs = [t for t in meta["idxs"] if t not in drop]

            for t in drop:
                for f in (d / v / f"f{t:05d}.jpg", d / v / f"m{t:05d}.npy"):
                    if f.exists() and not DRY:
                        f.unlink()

            if add:
                src = VIEWS / base / v
                h5, mp4 = src / "seg.h5", src / "rgb.mp4"
                if not (h5.exists() and mp4.exists()):
                    print(f"  {cid}/{v}: views gone, cannot add {add}")
                else:
                    frames = decode_at(mp4, set(add))
                    with h5py.File(h5, "r") as f:
                        seg = f["seg"]
                        ids = meta.get("target_ids") or []
                        for t in add:
                            if t >= seg.shape[0]:
                                continue
                            if t in frames and not DRY:
                                Image.fromarray(frames[t]).save(d / v / f"f{t:05d}.jpg", quality=92)
                            m = (np.isin(seg[t][()], ids) if ids
                                 else np.zeros(seg[t][()].shape, bool))
                            if not DRY:
                                np.save(d / v / f"m{t:05d}.npy", m)
                            if t not in idxs:
                                idxs.append(t)
                    idxs.sort()

            meta["idxs"] = sorted(set(idxs))
            if not DRY:
                meta_p.write_text(json.dumps(meta))

        n_drop_frame += len(drop)
        n_add += len(add)
        print(f"  {cid}: -{len(drop)} frames, +{len(add)} frames -> {len(views)} views")

    tag = "(dry run) " if DRY else ""
    print(f"{tag}cards dropped {n_drop_card}, frames dropped {n_drop_frame}, "
          f"frames added {n_add}, missing {n_miss}")


if __name__ == "__main__":
    main()
