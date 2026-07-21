"""Multi-anchor SAM2 propagation from the hand-labeled keyframes.

For each episode with manual_masks/{base}_wrist_manual.npz:
  every non-empty manual keyframe becomes an `add_new_mask` anchor for the SAM2
  video predictor; a forward + reverse propagation then yields dense masks whose
  drift is bounded by the anchor spacing (this is the production wrist-GT pass).

Outputs: propagated/{base}_wrist_prop.npz (masks (T,H,W) bool, tag)
Sanity report per episode: IoU at the anchors (should be ~1), per-frame area curve
smoothness, and agreement with the old single/auto-prompt SAM2 masks if present.

Run with the CUDA env:
  & D:\\Dev\\conda-envs\\py312\\python.exe propagate_from_manual.py
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
RID = ROOT.parent
V2SAM = RID.parent / "V2sam"
sys.path.insert(0, str(V2SAM))
sys.path.insert(0, str(RID.parent))

import torch  # noqa: E402

assert torch.cuda.is_available(), "run with py312 CUDA env"
from third_parts.sam2.build_sam import build_sam2_video_predictor  # noqa: E402

from crossview_viz import decode_video_frames, robust_video_len  # noqa: E402

CKPT = V2SAM / "weights/sam2/sam2_hiera_base_plus.pt"
OUT = ROOT / "propagated"
TMP = ROOT / "_frames_tmp_prop"


def wrist_mp4(ep_idx: int) -> Path:
    return RID / "videos" / f"chunk-000/observation.images.wrist/episode_{ep_idx:06d}.mp4"


def run(base: str, predictor) -> None:
    meta = json.loads((ROOT / "frames" / base / "meta.json").read_text(encoding="utf-8"))
    manual = np.load(ROOT / "manual_masks" / f"{base}_wrist_manual.npz")["masks"]
    T = min(meta["total_frames"], robust_video_len(str(wrist_mp4(meta["episode_index"]))) or meta["total_frames"])
    anchors = [f for f in meta["frames"] if f < T and manual[f].any()]
    if not anchors:
        print(f"{base}: no non-empty anchors, skip")
        return

    frames = decode_video_frames(str(wrist_mp4(meta["episode_index"])), list(range(T)))
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    for i, fr in enumerate(frames):
        cv2.imwrite(str(TMP / f"{i:05d}.jpg"), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))

    state = predictor.init_state(str(TMP), offload_video_to_cpu=True, offload_state_to_cpu=True)
    for f in anchors:
        predictor.add_new_mask(state, frame_idx=f, obj_id=1, mask=torch.from_numpy(manual[f]))
    got = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for reverse in (False, True):
            for f_idx, _ids, logits in predictor.propagate_in_video(state, reverse=reverse):
                got[f_idx] = (logits[0, 0] > 0).float().cpu().numpy().astype(bool)

    full = np.zeros((meta["total_frames"], 180, 320), bool)
    for f, m in got.items():
        if m.shape != (180, 320):
            m = cv2.resize(m.astype(np.uint8), (320, 180), interpolation=cv2.INTER_NEAREST).astype(bool)
        if f < full.shape[0]:
            full[f] = m

    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / f"{base}_wrist_prop.npz", masks=full,
                        tag=np.array(f"SAM2-b+ {len(anchors)}anchors(manual)"))

    a_iou = []
    for f in anchors:
        inter = (full[f] & manual[f]).sum()
        union = (full[f] | manual[f]).sum()
        a_iou.append(inter / union if union else 1.0)
    areas = full.sum((1, 2))
    jumps = np.abs(np.diff(areas.astype(int)))
    big_jumps = int((jumps > np.maximum(200, areas[:-1] * 0.6)).sum())
    old = RID / "sam2_wrist" / f"{base}_wrist_sam2.npz"
    agree = ""
    if old.exists():
        o = np.load(old)["masks"]
        L = min(len(o), len(full))
        inter = (full[:L] & o[:L]).sum()
        union = (full[:L] | o[:L]).sum()
        agree = f" agree_with_old={inter / union:.2f}" if union else ""
    print(f"{base}: anchors={anchors} anchorIoU={np.mean(a_iou):.2f} "
          f"coverage={(areas > 0).mean():.0%} area_jumps={big_jumps}{agree}")


def main() -> None:
    print("loading SAM2 b+ (video) on cuda ...")
    predictor = build_sam2_video_predictor("sam2_hiera_b+.yaml", str(CKPT), device="cuda",
                                           apply_postprocessing=False)
    bases = sorted(p.stem.replace("_wrist_manual", "") for p in (ROOT / "manual_masks").glob("*_wrist_manual.npz"))
    for base in bases:
        try:
            run(base, predictor)
        except Exception as e:
            print(f"{base}: FAILED {type(e).__name__}: {e}")
    if TMP.exists():
        shutil.rmtree(TMP)
    print("PROPAGATE-DONE")


if __name__ == "__main__":
    main()
