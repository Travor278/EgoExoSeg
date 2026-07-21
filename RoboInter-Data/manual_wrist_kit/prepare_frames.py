"""Prepare a labelme manual-annotation test kit for RoboInter WRIST frames.

Picks a handful of diverse DROID episodes (wrist mp4s already local), extracts
K uniformly-sampled frames as 2x-upscaled PNGs (easier to polygon precisely),
and writes per-episode meta. Convert back + compare with convert_labelme.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
RID = ROOT.parent
sys.path.insert(0, str(RID.parent))
sys.path.insert(0, str(RID))
from crossview_viz import decode_video_frames, sample_indices  # noqa: E402

MANIFEST = RID / "samples" / "gallery_manifest.json"
SCALE = 2  # 320x180 -> 640x360 for annotation comfort
K = 6      # frames per episode

# 6 diverse DROID picks: 2 where SAM2-auto struggled + 4 ordinary (edit freely)
BASES = ["10427", "10020", "10010", "0", "1000", "10082"]


FALLBACK = {  # hand-demo episodes not in the current manifest (SAM2 behavior well known)
    "10020": {"eps": [2], "frames": 113, "instr": "take the black lid from the top of the can and put it on the table"},
    "10010": {"eps": [3], "frames": 104, "instr": "move the bowl to the right"},
    "0": {"eps": [0], "frames": 240, "instr": "pulling the fabric to the left"},
}


def main() -> None:
    entries = {e["base"]: e for e in json.loads(MANIFEST.read_text(encoding="utf-8")) if e["ds"] == "droid"}
    for b, e in FALLBACK.items():
        entries.setdefault(b, {"base": b, **e})
    out_frames = ROOT / "frames"
    for base in BASES:
        e = entries.get(base)
        if e is None:
            print(f"!! {base} not in manifest, skip")
            continue
        ep = e["eps"][0]
        mp4 = RID / "videos" / f"chunk-000/observation.images.wrist/episode_{ep:06d}.mp4"
        if not mp4.exists():
            print(f"!! wrist mp4 missing for {base} (ep {ep}), skip")
            continue
        idxs = sample_indices(e["frames"], K)
        frames = decode_video_frames(str(mp4), idxs)
        d = out_frames / base
        d.mkdir(parents=True, exist_ok=True)
        for f_idx, fr in zip(idxs, frames):
            up = cv2.resize(fr, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(str(d / f"f{f_idx:04d}.png"), cv2.cvtColor(up, cv2.COLOR_RGB2BGR))
        (d / "meta.json").write_text(
            json.dumps({"base": base, "episode_index": ep, "frames": idxs, "scale": SCALE,
                        "instruction": e["instr"], "total_frames": e["frames"]}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"{base}: {len(idxs)} frames -> {d}  ({e['instr']})")
    print("\nnext: labelme 标注(见 README_标注指南.md), 然后 python convert_labelme.py")


if __name__ == "__main__":
    main()
