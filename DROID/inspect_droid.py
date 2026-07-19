"""Inspect DROID (lerobot/droid_100) samples and render cross-view montages.

DROID's standardized Franka rig records 3 synchronized stereo cameras.
The LeRobot mirror exposes the LEFT eye of each as one concatenated MP4:
    observation.images.exterior_image_1_left  -> external ZED 2 #1  (exo)
    observation.images.exterior_image_2_left  -> external ZED 2 #2  (exo)
    observation.images.wrist_image_left       -> wrist ZED Mini     (wrist)
This is the canonical "2 exo + 1 wrist" (i.e. 1-ego : 2-exo) layout.

Frames of all three cameras are frame-synchronized, and the single per-camera
MP4 concatenates episodes in the order given by the data parquet's row index,
so a global row index maps to the same instant in every camera video.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crossview_viz import build_montage, decode_video_frames, sample_indices, video_len  # noqa: E402

CAMS = [
    ("exterior_1", "exterior_image_1_left"),
    ("exterior_2", "exterior_image_2_left"),
    ("wrist", "wrist_image_left"),
]


def episode_ranges(parquet_path: Path):
    """Return [(episode_index, start_row, length), ...] from the data parquet."""
    tbl = pq.read_table(parquet_path, columns=["episode_index"])
    epi = tbl.column("episode_index").to_numpy()
    ranges = []
    start = 0
    for i in range(1, len(epi) + 1):
        if i == len(epi) or epi[i] != epi[start]:
            ranges.append((int(epi[start]), start, i - start))
            start = i
    return ranges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--frames", type=int, default=4)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    raw = root / "raw"
    out_dir = root / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    vids = {name: raw / key / "file-000.mp4" for name, key in CAMS}
    for name, p in vids.items():
        if not p.exists():
            raise SystemExit(f"missing video {p} -- run the download first")

    data_parquet = raw / "data" / "file-000.parquet"
    ranges = episode_ranges(data_parquet)
    vlen = min(video_len(str(p)) for p in vids.values())
    print("video frames:", vlen, "| episodes in parquet:", len(ranges))

    results = []
    for ep in args.episodes:
        epi_idx, start, length = ranges[ep]
        local = sample_indices(length, args.frames)
        global_idx = [start + li for li in local]
        # decode the same global indices from each camera (synchronized)
        frames_per_cam = {name: decode_video_frames(str(vids[name]), global_idx) for name, _ in CAMS}

        rows = []
        for k, gi in enumerate(global_idx):
            arrs = [frames_per_cam[name][k] for name, _ in CAMS]
            rows.append((f"frame {gi}", f"ep {epi_idx}  t{k}", arrs))

        out = out_dir / f"episode_{epi_idx:03d}_crossview.png"
        build_montage(
            out_path=out,
            title="DROID  ·  same object, 2 external + 1 wrist view",
            subtitle=f"lerobot/droid_100  —  episode {epi_idx}  ({length} frames)  —  Franka Panda, ZED stereo, 15 fps",
            cam_names=[name for name, _ in CAMS],
            rows=rows,
            footer="EXO = external ZED 2 (adjustable)   |   WRIST = ZED Mini on the wrist   |   source: lerobot/droid_100 (left eye, 180x320)",
            tile=(340, 191),  # 320x180 native aspect
        )
        results.append({
            "episode": epi_idx,
            "length": length,
            "global_indices": global_idx,
            "cameras": [name for name, _ in CAMS],
            "montage": str(out.relative_to(root)),
        })
        print(f"episode {epi_idx}: {length} frames -> {out.name}")

    manifest = {
        "dataset": "DROID",
        "source": "https://huggingface.co/datasets/lerobot/droid_100",
        "camera_layout": "2 external ZED 2 (exterior_1/2) + 1 wrist ZED Mini",
        "note": "LeRobot mirror = left eye of each stereo cam, 180x320, 15 fps. Full DROID is stereo + higher res. No mask.",
        "episodes": results,
    }
    (root / "crossview_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote", root / "crossview_manifest.json")


if __name__ == "__main__":
    main()
