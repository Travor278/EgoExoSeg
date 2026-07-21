"""Automatic SAM2 wrist-mask pass for every gallery entry (no manual clicks).

Prompting heuristic (per episode):
  1. prompt frame = RoboInter `annotation.contact_frame` (the gripper touches the
     manipulated object there, so the object occupies the wrist view's working
     area); fallback = mid-episode.
  2. SAM2 image predictor on that frame with a 3x3 grid of candidate points over
     the center/lower working area, multimask on -> up to 27 candidate masks.
  3. keep candidates with sane area (0.3%..55% of frame), pick the highest
     predicted-IoU score; that mask becomes an `add_new_mask` prompt.
  4. SAM2 video predictor propagates it forward+reverse over the FULL episode.

Output: sam2_wrist/{base}_wrist_sam2.npz (same convention as sam2_wrist_preview,
picked up by visualize_robointer_masks / build_gallery). Existing npz are kept
(skip = resume), so the 4 hand-prompted demos stay as-is.

Masks are AUTO prompts + SAM2, tagged as such — NOT ground truth, not reviewed.
Run with the CUDA env:  & D:\\Dev\\conda-envs\\py312\\python.exe sam2_wrist_auto.py
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
V2SAM = ROOT.parent / "V2sam"
sys.path.insert(0, str(V2SAM))
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

assert torch.cuda.is_available(), "run this with the py312 CUDA env"

from third_parts.sam2.build_sam import build_sam2, build_sam2_video_predictor  # noqa: E402
from third_parts.sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402

import visualize_robointer_masks as viz  # noqa: E402
from crossview_viz import decode_video_frames, robust_video_len  # noqa: E402

CKPT = V2SAM / "weights/sam2/sam2_hiera_base_plus.pt"
CONFIG = "sam2_hiera_b+.yaml"
OUT_DIR = ROOT / "sam2_wrist"
FRAME_TMP = OUT_DIR / "_frames_tmp_auto"
MANIFEST = ROOT / "samples" / "gallery_manifest.json"

GRID_X = (120, 160, 200)
GRID_Y = (70, 95, 120)
AREA_MIN, AREA_MAX = 0.003, 0.55


def contact_frame(ds: str, ep_idx: int, total: int) -> int:
    p = viz.DATASETS[ds]["chunk_dir"] / f"episode_{ep_idx:06d}.parquet"
    try:
        col = pq.read_table(p, columns=["annotation.contact_frame"]).to_pandas().iloc[:, 0]
        vals = [int(v) for v in (json.loads(str(x)) for x in col if str(x).strip()) if isinstance(v, int) and v >= 0]
        if vals:
            return int(np.clip(max(set(vals), key=vals.count), 0, total - 1))
    except Exception:
        pass
    return total // 2


def pick_prompt_mask(img_pred: SAM2ImagePredictor, frame: np.ndarray) -> tuple[np.ndarray | None, float]:
    h, w = frame.shape[:2]
    img_pred.set_image(frame)
    best, best_score = None, -1.0
    for gx in GRID_X:
        for gy in GRID_Y:
            masks, scores, _ = img_pred.predict(
                point_coords=np.array([[gx, gy]], dtype=np.float32),
                point_labels=np.array([1], dtype=np.int32),
                multimask_output=True,
            )
            for m, s in zip(masks, scores):
                frac = float(m.sum()) / (h * w)
                if AREA_MIN <= frac <= AREA_MAX and float(s) > best_score:
                    best, best_score = m.astype(bool), float(s)
    return best, best_score


def run_entry(e: dict, video_pred, img_pred) -> str:
    base = e["base"]
    out = OUT_DIR / f"{base}_wrist_sam2.npz"
    if out.exists():
        return "skip(existing)"
    ds, ep = e["ds"], e["eps"][0]
    vp = viz.video_path(ds, ep, "wrist")
    if not vp.exists():
        return "skip(no-video)"
    total = min(e["frames"], robust_video_len(str(vp)) or e["frames"])
    pf = contact_frame(ds, ep, total)

    frames = decode_video_frames(str(vp), list(range(total)))
    h, w = frames[0].shape[:2]
    prompt_mask, score = pick_prompt_mask(img_pred, frames[pf])
    if prompt_mask is None:
        return "fail(no-prompt-candidate)"

    if FRAME_TMP.exists():
        shutil.rmtree(FRAME_TMP)
    FRAME_TMP.mkdir(parents=True)
    for i, fr in enumerate(frames):
        cv2.imwrite(str(FRAME_TMP / f"{i:05d}.jpg"), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))

    state = video_pred.init_state(str(FRAME_TMP), offload_video_to_cpu=True, offload_state_to_cpu=True)
    video_pred.add_new_mask(state, frame_idx=pf, obj_id=1, mask=torch.from_numpy(prompt_mask))
    got = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for reverse in (False, True):
            for f_idx, _ids, logits in video_pred.propagate_in_video(state, start_frame_idx=pf, reverse=reverse):
                got[f_idx] = (logits[0, 0] > 0).float().cpu().numpy().astype(bool)

    full = np.zeros((e["frames"], h, w), dtype=bool)
    for f, m in got.items():
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        if f < full.shape[0]:
            full[f] = m
    OUT_DIR.mkdir(exist_ok=True)
    np.savez_compressed(
        out,
        masks=full,
        tag=np.array(f"SAM2-b+ auto@f{pf}"),
        prompt_frame=np.array(pf, dtype=np.int32),
        prompt_score=np.array(score, dtype=np.float32),
    )
    fracs = full.reshape(full.shape[0], -1).mean(1)
    flag = ""
    if fracs.max() > 0.7:
        flag = " FLAG:blowup"
    elif (fracs > 0).sum() < total * 0.2:
        flag = " FLAG:mostly-empty"
    return f"ok score={score:.2f} pf={pf} px_med={int(np.median(full.sum((1, 2)))):d}{flag}"


def main() -> None:
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    print(f"loading SAM2 b+ (video+image) on cuda ...")
    video_pred = build_sam2_video_predictor(CONFIG, str(CKPT), device="cuda", apply_postprocessing=False)
    img_pred = SAM2ImagePredictor(build_sam2(CONFIG, str(CKPT), device="cuda", apply_postprocessing=False))
    t0 = time.time()
    for i, e in enumerate(entries, 1):
        try:
            msg = run_entry(e, video_pred, img_pred)
        except Exception as exc:
            msg = f"fail({type(exc).__name__}: {exc})"
        print(f"[{i}/{len(entries)}] {e['ds']}:{e['base']}: {msg}", flush=True)
    if FRAME_TMP.exists():
        shutil.rmtree(FRAME_TMP)
    print(f"AUTO-WRIST-DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
