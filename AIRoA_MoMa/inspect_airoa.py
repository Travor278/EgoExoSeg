"""Inspect AIRoA MoMa samples and render hand<->head cross-view montages.

The OFFICIAL AIRoA MoMa dataset (airoa-org/airoa-moma) is HF-gated. This script
uses the UNGATED demo release from the SAME org (airoa-org/corl2025_demo), which
is the SAME Toyota HSR robot + SAME 2-camera rig at FULL resolution:
    observation.image.head -> head-mounted external/global camera (exo)
    observation.image.hand -> hand-mounted in-hand camera         (wrist)
480x640 AV1 @10fps, time-synchronized. Lets us verify the wrist<->head cross-view
structure and — the key question — whether the manipulated object is co-visible
in the hand and head views at the same instant. No object masks in either.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crossview_viz import build_montage, decode_video_frames_robust, robust_video_len, sample_indices  # noqa: E402

CAMS = ["head", "hand"]  # display order: external first, in-hand second


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=["001", "010"])
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frames", type=int, default=4)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    raw = root / "raw"
    out_dir = root / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for task in args.tasks:
        vids = [(c, raw / f"task{task}" / c / f"episode_{args.episode:06d}.mp4") for c in CAMS]
        vids = [(c, p) for c, p in vids if p.exists()]
        if not vids:
            continue
        vlen = min(robust_video_len(str(p)) for _, p in vids)
        idxs = sample_indices(vlen, args.frames)
        fpc = {c: decode_video_frames_robust(str(p), idxs) for c, p in vids}
        h, w = fpc[vids[0][0]][0].shape[:2]

        rows = []
        for k, gi in enumerate(idxs):
            rows.append((f"frame {gi}", f"of {vlen}", [fpc[c][k] for c, _ in vids]))

        out = out_dir / f"task{task}_ep{args.episode:06d}_crossview.png"
        build_montage(
            out_path=out,
            title="AIRoA MoMa  ·  same object, head + hand views",
            subtitle=f"airoa-org/corl2025_demo (ungated, same HSR rig as gated MoMa)  —  task {task}  —  {w}x{h}, 10 fps",
            cam_names=[c for c, _ in vids],
            rows=rows,
            footer="EXO = head camera  |  WRIST = in-hand camera  |  source: airoa-org/corl2025_demo (Toyota HSR; official-org ungated demo)",
            tile=(340, int(340 * h / w)),
        )
        results.append({"task": task, "episode": args.episode, "frames": vlen,
                        "cameras": [c for c, _ in vids], "resolution": f"{w}x{h}",
                        "montage": str(out.relative_to(root))})
        print(f"task {task} ep{args.episode}: {[c for c,_ in vids]} {w}x{h} {vlen}f -> {out.name}")

    manifest = {
        "dataset": "AIRoA MoMa (via airoa-org ungated demo)",
        "official_dataset": "https://huggingface.co/datasets/airoa-org/airoa-moma (HF gated)",
        "sample_source": "https://huggingface.co/datasets/airoa-org/corl2025_demo (ungated, same org + same HSR rig)",
        "camera_layout": "head (external/head-mounted) + hand (in-hand), 480x640 AV1 @10fps, time-synchronized",
        "mask": "none in either release",
        "note": "Demo is full-resolution and from the official org, so a strong stand-in for the gated MoMa. head is robot-head-mounted (egocentric-to-robot), not a fixed third-person rig.",
        "samples": results,
    }
    (root / "crossview_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote", root / "crossview_manifest.json")


if __name__ == "__main__":
    main()
