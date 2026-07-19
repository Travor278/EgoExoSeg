"""Inspect RoboSet (RoboAgent) HDF5 samples and render cross-view montages.

RoboSet packs many trajectories per .h5 file. Each Trial group has a `data`
subgroup holding 4 synchronized RealSense D455 RGB streams:
    rgb_top, rgb_left, rgb_right   -> 3 fixed external cameras
    rgb_wrist                      -> 1 wrist camera (above the end-effector)
plus matching depth (d_*). This confirms the survey's "3 fixed + 1 wrist" claim.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crossview_viz import build_montage, sample_indices  # noqa: E402

# display name -> hdf5 key.  Order = external views first, wrist last.
CAMS = [("top", "rgb_top"), ("left", "rgb_left"), ("right", "rgb_right"), ("wrist", "rgb_wrist")]


def inspect(h5_path: Path, out_dir: Path, n_trials: int, n_frames: int):
    results = []
    with h5py.File(h5_path, "r") as f:
        trials = list(f.keys())[:n_trials]
        for trial in trials:
            data = f[trial]["data"]
            keys = set(data.keys())
            present = [(name, key) for name, key in CAMS if key in keys]
            n = int(data[present[0][1]].shape[0])
            idxs = sample_indices(n, n_frames)

            rows = []
            for idx in idxs:
                arrs = [np.asarray(data[key][idx]) for _, key in present]
                rows.append((f"frame {idx}", f"of {n}", arrs))

            out = out_dir / f"{trial}_crossview.png"
            build_montage(
                out_path=out,
                title="RoboSet  ·  same object, 4 synchronized views",
                subtitle=f"{h5_path.stem}  —  {trial}  —  Franka Panda, RealSense D455, 5 Hz",
                cam_names=[name for name, _ in present],
                rows=rows,
                footer="EXO = fixed external camera   |   WRIST = end-effector camera   |   source: jdvakil/RoboSet-Teleoperation (HF mirror, 240x424)",
            )
            results.append({
                "trial": trial,
                "frames": n,
                "cameras": [name for name, _ in present],
                "hdf5_keys": [key for _, key in present],
                "sampled_indices": idxs,
                "montage": str(out.relative_to(out_dir.parent)),
            })
            print(f"{trial}: cams={[name for name,_ in present]} frames={n} -> {out.name}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=None, help="path to a RoboSet .h5 file")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--frames", type=int, default=4)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    h5_path = Path(args.h5) if args.h5 else next((root / "raw").glob("*.h5"))
    out_dir = Path(args.out_dir) if args.out_dir else root / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = inspect(h5_path, out_dir, args.trials, args.frames)
    manifest = {
        "dataset": "RoboSet (RoboAgent)",
        "source": "https://huggingface.co/datasets/jdvakil/RoboSet-Teleoperation",
        "h5_file": h5_path.name,
        "camera_layout": "3 fixed external (top/left/right) + 1 wrist, RealSense D455",
        "note": "HF mirror resolution 240x424; original RoboSet is higher. No segmentation mask in release.",
        "trials": results,
    }
    (root / "crossview_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote", root / "crossview_manifest.json")


if __name__ == "__main__":
    main()
