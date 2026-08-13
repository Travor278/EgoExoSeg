"""Build a self-contained review package ON a render host.

Runs where the data already lives, so nothing has to be copied twice: the .npy masks
in bundle/ are 900 KB each and dominate its size, while what a reviewer actually
needs is a JPEG plus a few-KB overlay PNG.

Per card it emits:
  * the 8 sampled frames at full resolution, each with a transparent mask overlay,
    for judging whether the annotation is tight
  * a scrub video per view -- EVERY rendered frame, mask tinted in, downscaled --
    so the reviewer can pick their own moments instead of being limited to the 8
    the sampler chose

Videos come from views/<demo>/<view>/{rgb.mp4,seg.h5} and are only available for
demos whose views were kept; cards without them still get the 8 frames.

Usage (on the host, inside the behavior env):
    python build_review_host.py OUT_DIR [PILOT_ROOT]
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

ROOT = Path(sys.argv[2] if len(sys.argv) > 2 else os.environ.get("PILOT_ROOT", Path.home() / "pilot_v5"))
BUNDLE, VIEWS = ROOT / "bundle", ROOT / "views"
EGO_VIEWS = ("head_gaze", "head")
MIN_VIS = float(os.environ.get("PILOT_MIN_VIS", "0.05"))
MASK_RGB = (255, 64, 160)
VID_W = int(os.environ.get("REVIEW_VID_W", "640"))   # scrubbing needs speed, not detail
VID_FPS = int(os.environ.get("REVIEW_VID_FPS", "10"))


def pretty(name: str) -> str:
    return re.sub(r"_\d+$", "", name.split(".")[0]).replace("__", " ").replace("_", " ").strip()


def pick_views(d: Path):
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
    return {"ego": ego, "exos": exos, "views": views} if ego and exos else None


def overlay_png(mask: np.ndarray, out: Path):
    """Transparent PNG coloured only where the mask is; returns (px, bbox-fractions)."""
    n = int(mask.sum())
    if n == 0:
        return 0, None
    rgba = np.zeros((*mask.shape, 4), np.uint8)
    rgba[mask] = (*MASK_RGB, 255)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(out, optimize=True)
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    return n, [round(float((xs.min() + xs.max()) / 2 / w), 4),
               round(float((ys.min() + ys.max()) / 2 / h), 4),
               round(float((xs.max() - xs.min() + 1) / w), 4),
               round(float((ys.max() - ys.min() + 1) / h), 4)]


def is_black(img: Image.Image) -> bool:
    """A failed render comes out uniformly black; a real indoor scene never does.

    These are not rare -- 32 of 704 sampled cells in one review set were black, all
    on the moving ego camera -- and they read as "the mask is missing" unless the
    page says otherwise.
    """
    return float(np.asarray(img.convert("L").resize((64, 36))).mean()) < 3.0


def scrub_video(view_dir: Path, ids: list[int], out_mp4: Path) -> bool:
    """Re-encode rgb.mp4 with the target tinted, so the reviewer can scrub with mask."""
    mp4, h5 = view_dir / "rgb.mp4", view_dir / "seg.h5"
    if not (mp4.exists() and h5.exists()) or not ids:
        return False
    try:
        import av
    except ImportError:
        return False
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    idset = np.array(sorted(ids))
    try:
        # Encode with PyAV rather than piping to an ffmpeg binary. The b1k container has
        # PyAV but no ffmpeg on PATH, so the subprocess version silently produced zero
        # videos there ("No such file or directory: 'ffmpeg'") while working fine on the
        # native host. PyAV carries its own libav, so this needs nothing installed.
        with h5py.File(h5, "r") as f:
            seg = f["seg"]
            n = seg.shape[0]
            with av.open(str(mp4)) as probe:
                first = next(probe.decode(video=0)).to_ndarray(format="rgb24")
            h, w = first.shape[:2]
            vw = VID_W
            vh = int(round(h * vw / w)) // 2 * 2

            with av.open(str(out_mp4), "w") as dst:
                st = dst.add_stream("libx264", rate=VID_FPS)
                st.width, st.height, st.pix_fmt = vw, vh, "yuv420p"
                st.options = {"crf": "28", "preset": "veryfast"}
                with av.open(str(mp4)) as src:
                    for i, fr in enumerate(src.decode(video=0)):
                        if i >= n:
                            break
                        a = fr.to_ndarray(format="rgb24")
                        m = np.isin(seg[i][()], idset)
                        if m.any():
                            a = a.copy()
                            a[m] = (0.45 * a[m] + 0.55 * np.array(MASK_RGB)).astype(np.uint8)
                        vf = av.VideoFrame.from_ndarray(a, format="rgb24")
                        for pkt in st.encode(vf.reformat(width=vw, height=vh)):
                            dst.mux(pkt)
                for pkt in st.encode():
                    dst.mux(pkt)
        return out_mp4.exists() and out_mp4.stat().st_size > 0
    except Exception as e:
        print(f"  video failed for {view_dir}: {e}", flush=True)
        return False


def main() -> None:
    out = Path(sys.argv[1])
    (out / "f").mkdir(parents=True, exist_ok=True)
    cards, n_vid = [], 0

    for d in sorted(p for p in BUNDLE.iterdir() if p.is_dir()):
        info = pick_views(d)
        if not info:
            continue
        vlist = [info["ego"], *info["exos"]]
        idxs = info["views"][info["ego"]]["idxs"]
        base = d.name.split("__")[0]

        frames = []
        for t in idxs:
            per_view = []
            for v in vlist:
                jpg = d / v / f"f{t:05d}.jpg"
                npy = d / v / f"m{t:05d}.npy"
                rel = f"f/{d.name}/{v}/{t}.jpg"
                black = False
                if jpg.exists():
                    im = Image.open(jpg)
                    black = is_black(im)
                    (out / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(jpg, out / rel)
                px, bbox = (overlay_png(np.load(npy), out / "m" / d.name / v / f"{t}.png")
                            if npy.exists() else (0, None))
                per_view.append({"img": rel if jpg.exists() else None,
                                 "mask": f"m/{d.name}/{v}/{t}.png" if px else None,
                                 "px": px, "bbox": bbox, "black": black})
            frames.append({"t": t, "views": per_view})

        vids = []
        for v in vlist:
            rel = f"v/{d.name}/{v}.mp4"
            ok = scrub_video(VIEWS / base / v, info["views"][v].get("target_ids") or [], out / rel)
            vids.append(rel if ok else None)
            n_vid += 1 if ok else 0

        tgt = info["views"][info["ego"]]["target"]
        areas = [fv["px"] for f in frames for fv in f["views"] if fv["px"] > 0]
        cards.append({
            "id": d.name, "target": pretty(tgt.get("name", "?")), "target_raw": tgt.get("name", "?"),
            "views": vlist, "videos": vids,
            "stats": [{"view": v, "vis": round(info["views"][v]["visible_frac"], 3),
                       "median_px": info["views"][v].get("median_px", 0)} for v in vlist],
            "min_px": min(areas) if areas else 0,
            # Reference frame count: the base the card's t labels live in. The page
            # converts a scrub position to a frame number with it, and the views
            # themselves have different lengths so no single clip can supply it.
            "n_ref": int(info["views"][info["ego"]].get("Tmin") or 0),
            "n_black": sum(1 for f in frames for fv in f["views"] if fv["black"]),
            "frames": frames,
        })
        print(f"  {d.name}: {len(frames)} frames, videos={sum(1 for x in vids if x)}", flush=True)

    (out / "data.js").write_text(
        "window.REVIEW = " + json.dumps({"cards": cards}, ensure_ascii=False) + ";\n",
        encoding="utf-8")
    print(f"{len(cards)} cards, {n_vid} scrub videos -> {out}")


if __name__ == "__main__":
    main()
