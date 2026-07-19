"""Inspect Galaxea R1 samples and render wrist<->head cross-view montages.

The OFFICIAL Galaxea Open-World Dataset (720x1280, head + head_right + 2 wrist)
is HF-gated + ModelScope login-walled. This script instead uses an UNGATED
COMMUNITY recording on the SAME Galaxea R1 robot and the SAME camera rig
(1 head external + left/right wrist in-hand), so the wrist<->head cross-view
structure — and the key question of whether the manipulated object is co-visible
in a wrist view and the head view at the same instant — is representative.
Resolution here is small (community/normalized dump); treat visuals as
structure-only, not final quality.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crossview_viz import build_montage, decode_video_frames_robust, robust_video_len, sample_indices  # noqa: E402

# display name -> camera folder.  head = external/exo, wrists = in-hand.
CAMS = ["head", "left_wrist", "right_wrist"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*", default=[0, 9, 18])
    ap.add_argument("--frames", type=int, default=4)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    raw = root / "raw"
    out_dir = root / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for ep in args.episodes:
        vids = [(c, raw / c / f"episode_{ep:06d}.mp4") for c in CAMS]
        vids = [(c, p) for c, p in vids if p.exists()]
        if not vids:
            continue
        vlen = min(robust_video_len(str(p)) for _, p in vids)
        idxs = sample_indices(vlen, args.frames)
        frames_per_cam = {c: decode_video_frames_robust(str(p), idxs) for c, p in vids}
        h, w = frames_per_cam[vids[0][0]][0].shape[:2]

        rows = []
        for k, gi in enumerate(idxs):
            rows.append((f"frame {gi}", f"of {vlen}", [frames_per_cam[c][k] for c, _ in vids]))

        out = out_dir / f"episode_{ep:06d}_crossview.png"
        build_montage(
            out_path=out,
            title="Galaxea R1  ·  same object, head + dual-wrist views",
            subtitle=f"community R1 'cup' recording (stand-in for gated Open-World)  —  episode {ep}  —  {w}x{h}, 30 fps",
            cam_names=[c for c, _ in vids],
            rows=rows,
            footer="EXO = head camera  |  WRIST = in-hand camera  |  source: s-tian/galaxea_r1_cup (ungated community R1; NOT the official 720x1280 Open-World)",
            tile=(300, int(300 * h / w)),
        )
        results.append({"episode": ep, "frames": vlen, "cameras": [c for c, _ in vids],
                        "resolution": f"{w}x{h}", "montage": str(out.relative_to(root))})
        print(f"episode {ep}: {[c for c,_ in vids]} {w}x{h} {vlen}f -> {out.name}")

    manifest = {
        "dataset": "Galaxea R1 (community stand-in for Galaxea Open-World)",
        "official_dataset": "https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset (HF gated + ModelScope login-walled)",
        "sample_source": "https://huggingface.co/datasets/s-tian/galaxea_r1_cup (ungated community R1 recording)",
        "official_camera_layout": "head + head_right (external) + left_wrist + right_wrist (in-hand), 720x1280 AV1 @15fps, strict temporal sync",
        "sample_camera_layout": "head (external) + left_wrist + right_wrist (in-hand), small-res, 30fps",
        "caveat": "This is NOT the official Open-World Dataset; it is a community recording on the same R1 robot/rig used to VERIFY the wrist<->head cross-view structure and object co-visibility. No object masks in either.",
        "episodes": results,
    }
    (root / "crossview_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote", root / "crossview_manifest.json")


if __name__ == "__main__":
    main()
