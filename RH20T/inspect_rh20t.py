"""Inspect RH20T (hainh22/rh20t LeRobot mirror) and render a multi-view montage.

RH20T's selling point is view density: each platform films the workspace with
8-10 global RGBD cameras + 1-2 in-hand cameras, all calibrated to the robot
base frame and time-synchronized. This mirror stores one MP4 per camera per
episode, so a single episode gives every view at once:
    cam_<serial>       -> global external cameras   (exo)
    cam_eye_in_hand    -> wrist / in-hand camera     (wrist)
    cam_front_view / cam_side_view -> derived named views (derived)
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crossview_viz import build_montage, decode_video_frames, sample_indices, video_len  # noqa: E402


# The hainh22/rh20t mirror is internally inconsistent: the serial-numbered
# camera videos (360x640, matching RH20T's real downsampled resolution) are one
# genuine RH20T scene shot from many global angles, but cam_front_view /
# cam_side_view / cam_eye_in_hand (256x256) come from a DIFFERENT source and do
# NOT depict the same take. We therefore montage only the consistent global set.
NAMED_MISMATCH = {"cam_front_view", "cam_side_view", "cam_eye_in_hand"}


def is_global_serial(cam: str) -> bool:
    return cam not in NAMED_MISMATCH


def collect(raw: Path, ep: int, keep):
    cams = []
    for d in sorted(raw.glob("cam_*")):
        if not keep(d.name):
            continue
        mp4 = d / f"episode_{ep:06d}.mp4"
        if mp4.exists() and video_len(str(mp4)) > 0:
            cams.append((d.name, mp4))
    return cams


def render(cams, ep, n_frames, out, title, subtitle, footer, tile):
    vlen = min(video_len(str(mp4)) for _, mp4 in cams)
    idxs = sample_indices(vlen, n_frames)
    frames_per_cam = {name: decode_video_frames(str(mp4), idxs) for name, mp4 in cams}
    rows = []
    for k, gi in enumerate(idxs):
        rows.append((f"frame {gi}", f"of {vlen}", [frames_per_cam[name][k] for name, _ in cams]))
    disp = [name.replace("cam_", "") for name, _ in cams]
    build_montage(out_path=out, title=title, subtitle=subtitle, cam_names=disp,
                  rows=rows, footer=footer, tile=tile)
    return vlen, idxs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frames", type=int, default=3)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    raw = root / "raw"
    out_dir = root / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    ep = args.episode

    # --- main deliverable: the consistent global-camera set ---
    glob_cams = collect(raw, ep, is_global_serial)
    if not glob_cams:
        raise SystemExit(f"no global cam videos found for episode {ep} in {raw}")
    vlen, idxs = render(
        glob_cams, ep, args.frames,
        out=out_dir / f"episode_{ep:06d}_global{len(glob_cams)}views.png",
        title=f"RH20T  ·  same object across {len(glob_cams)} global camera views",
        subtitle=f"hainh22/rh20t  —  episode {ep}  —  UR5, all cameras calibrated to the robot base frame, time-synchronized, 10 fps",
        footer="EXO = global RGBD camera (serial number)   |   360x640   |   source: hainh22/rh20t (LeRobot mirror of RH20T)",
        tile=(250, 145),
    )
    print(f"episode {ep}: {len(glob_cams)} global cameras, {vlen} frames -> global montage")

    # --- transparency: the mismatched named views (a DIFFERENT scene) ---
    named_cams = collect(raw, ep, lambda c: c in NAMED_MISMATCH)
    if named_cams:
        render(
            named_cams, ep, args.frames,
            out=out_dir / f"episode_{ep:06d}_mirror_mismatch.png",
            title="RH20T mirror caveat  ·  these 3 named views are a DIFFERENT scene",
            subtitle=f"hainh22/rh20t  —  episode {ep}  —  cam_front/side/eye_in_hand (256x256) do NOT match the global cameras above",
            footer="Evidence that the community mirror splices heterogeneous sources under one episode index. Do NOT treat as the same take.",
            tile=(250, 145),
        )
        print(f"episode {ep}: {len(named_cams)} mismatched named views -> caveat montage")

    manifest = {
        "dataset": "RH20T",
        "source": "https://huggingface.co/datasets/hainh22/rh20t",
        "official_site": "https://rh20t.github.io/",
        "episode": ep,
        "global_cameras_used": [name for name, _ in glob_cams],
        "n_global": len(glob_cams),
        "frames": vlen,
        "mirror_caveat": (
            "This mirror is internally inconsistent. The serial-numbered cameras (360x640) are one "
            "genuine RH20T scene from many global angles; cam_front_view/side_view/eye_in_hand (256x256) "
            "are a DIFFERENT source and are NOT the same take. Paper-verified config = 8-10 global RGBD "
            "+ 1-2 in-hand cameras, all calibrated to the robot base frame + time-synchronized. For a "
            "benchmark, pull properly-aligned global+in-hand streams from the official RH20T release."
        ),
        "mask": "none in release",
    }
    (root / "crossview_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote", root / "crossview_manifest.json")


if __name__ == "__main__":
    main()
