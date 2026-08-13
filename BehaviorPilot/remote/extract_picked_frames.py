"""Materialise the frames a reviewer picked out of the scrub videos.

The review page lets a reviewer scrub the full-length video and mark a moment worth
keeping. That records a frame NUMBER; the JPEG and the mask for it do not exist,
because the bundle only ever held the 8 frames the sampler chose. 272 of 394 picked
frames were in that state -- exporting without this step would have dropped 69% of
the review.

For each picked frame this writes the same files the bundler writes, so the card
gains rows indistinguishable from its original ones:

    bundle/<card>/<view>/f{t:05d}.jpg
    bundle/<card>/<view>/m{t:05d}.npy

Frame numbers are in the card's REFERENCE base. Views have different lengths, so
each one's own index is derived from the time fraction -- the same mapping the
bundler uses, and the reason a raw index cannot be reused across views.

Usage:
    python extract_picked_frames.py DECISIONS.json PILOT_ROOT [--dry-run]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def frames_from(mp4: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    """Decode just the requested frame indices."""
    import av

    out: dict[int, np.ndarray] = {}
    if not wanted:
        return out
    hi = max(wanted)
    with av.open(str(mp4)) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i in wanted:
                out[i] = fr.to_ndarray(format="rgb24")
            if i >= hi:
                break
    return out


def main() -> None:
    dec_p, root = Path(sys.argv[1]), Path(sys.argv[2])
    dry = "--dry-run" in sys.argv
    bundle, views = root / "bundle", root / "views"
    dec = json.loads(dec_p.read_text(encoding="utf-8"))

    todo: dict[str, list[int]] = {}
    for v in dec.get("cards", []):
        picked = sorted(set(v.get("keep_frames", [])) | set(v.get("add_frames", [])))
        if picked:
            # The page prefixes cards from the second host with z_; bundles do not.
            todo[v["id"].removeprefix("z_")] = picked

    n_card = n_new = n_skip = n_fail = 0
    for card, picked in sorted(todo.items()):
        d = bundle / card
        if not d.is_dir():
            continue
        base = card.split("__")[0]
        metas = {}
        for m in d.glob("*/meta.json"):
            metas[m.parent.name] = json.loads(m.read_text())
        if not metas:
            continue
        missing = [t for t in picked
                   if not all((d / v / f"f{t:05d}.jpg").exists() for v in metas)]
        if not missing:
            n_skip += 1
            continue
        n_card += 1
        if dry:
            print(f"  {card}: would extract {missing}")
            n_new += len(missing) * len(metas)
            continue

        tmin = min(m["T"] for m in metas.values())
        span = max(1, tmin - 1)
        for view, meta in metas.items():
            vd = views / base / view
            mp4, h5 = vd / "rgb.mp4", vd / "seg.h5"
            if not (mp4.exists() and h5.exists()):
                n_fail += 1
                continue
            ids = meta.get("target_ids") or []
            try:
                with h5py.File(h5, "r") as f:
                    seg = f["seg"]
                    n = seg.shape[0]
                    local = {t: min(n - 1, int(round(t / span * (n - 1)))) for t in missing}
                    imgs = frames_from(mp4, set(local.values()))
                    for t in missing:
                        lt = local[t]
                        if lt in imgs:
                            Image.fromarray(imgs[lt]).save(d / view / f"f{t:05d}.jpg", quality=92)
                        a = seg[lt][()]
                        np.save(d / view / f"m{t:05d}.npy",
                                np.isin(a, ids) if ids else np.zeros(a.shape, bool))
                        n_new += 1
            except Exception as e:  # noqa: BLE001
                print(f"  {card}/{view} failed: {type(e).__name__}: {e}", flush=True)
                n_fail += 1
        print(f"  {card}: +{len(missing)} frames x {len(metas)} views", flush=True)

        # Record the new frame list so the exporter and any rebuild see them.
        for view, meta in metas.items():
            meta["idxs"] = sorted(set(meta["idxs"]) | set(missing))
            (d / view / "meta.json").write_text(json.dumps(meta))

    print(f"cards touched {n_card}, already complete {n_skip}, "
          f"files written {n_new}, view failures {n_fail}")


if __name__ == "__main__":
    main()
