"""Preview of the planned wrist-mask pipeline: SAM2 with a few human clicks,
run on the WRIST stream of RoboInter episodes (DROID + RH20T).

RoboInter ships no wrist masks (annotations are anchored to the primary
exterior camera), so wrist ground truth for wrist<->exo must be built with
"SAM2 propagation + human review" (survey §5 path B / §6). This script
rehearses exactly that:

  clicks on the manipulated object at one clear wrist frame  ->  SAM2
  propagates forward+reverse  ->  npz of predicted wrist masks.

Device auto-select:
  - CUDA available (e.g. py312 env, torch+cu128, RTX 5070): hiera-base+ and
    FULL-video propagation over every frame.
  - CPU only: hiera-tiny and a sparse pseudo-video (the 4 montage frames) to
    stay tractable.

Output: sam2_wrist/{base}_wrist_sam2.npz (masks (T,H,W) bool; tag string).
visualize_robointer_masks.py picks these up and draws them in ORANGE,
tagged as predictions. NOT ground truth — the real GT pass adds human review.

Run (GPU): & D:\\Dev\\conda-envs\\py312\\python.exe sam2_wrist_preview.py
Run (CPU): python sam2_wrist_preview.py
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # duplicate libiomp5md on this box

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
V2SAM = ROOT.parent / "V2sam"
sys.path.insert(0, str(V2SAM))        # makes `third_parts.sam2` importable (hydra configs live there)
sys.path.insert(0, str(ROOT.parent))  # crossview_viz

import torch  # noqa: E402

USE_CUDA = torch.cuda.is_available()
if not USE_CUDA:
    # The vendored SAM2 (v2.0) hardcodes CUDA in a few places; neuter them on CPU boxes.
    torch.Tensor.cuda = lambda self, *a, **k: self

from third_parts.sam2.build_sam import build_sam2_video_predictor  # noqa: E402

from crossview_viz import decode_video_frames, robust_video_len  # noqa: E402

if USE_CUDA:
    CKPT, CONFIG, MODEL_TAG = V2SAM / "weights/sam2/sam2_hiera_base_plus.pt", "sam2_hiera_b+.yaml", "b+"
else:
    CKPT, CONFIG, MODEL_TAG = V2SAM / "weights/sam2/sam2_hiera_tiny.pt", "sam2_hiera_t.yaml", "tiny"
OUT_DIR = ROOT / "sam2_wrist"
FRAME_TMP = ROOT / "sam2_wrist" / "_frames_tmp"

# base: (dataset, wrist episode_index, episode length, montage frames, prompt frame,
#        prompt points [(x, y), ...] in video coords, labels [1=positive, 0=negative])
CASES: dict[str, tuple] = {
    "10020": ("droid", 2, 113, [0, 37, 75, 112], 112, [(218, 72)], [1]),  # black lid, isolated at f112
    # first try (single click (235, 85)) landed on the red stick and SAM2 tracked the stick,
    # not the bowl; second try mis-placed the negative click ON the bowl rim. Fine-grid
    # inspection of the frame fixed the geometry: bowl interior positives left/right of the
    # stick + negatives on the stick and on the occluding gripper finger.
    "10010": ("droid", 3, 104, [0, 34, 69, 103], 34,
              [(205, 85), (256, 50), (240, 60), (258, 130)], [1, 1, 0, 0]),
    "0": ("droid", 0, 240, [0, 80, 159, 239], 80, [(120, 80)], [1]),      # fleece fills the frame at f80
    # RH20T in-hand camera stares at the GT object (red button on white plate) all episode.
    # v1 second positive (138,118) fell on the tablecloth just outside the plate edge and
    # the mask exploded to full frame at f103; fixed: positives strictly inside the device,
    # negatives on tablecloth + robot base.
    "task_0001_user_0001_scene_0004_cfg_0001": (
        "rh20t", 2, 156, [0, 52, 103, 155], 100,
        [(163, 140), (185, 152), (160, 60), (80, 170)], [1, 1, 0, 0]),
}


def wrist_video(ds: str, idx: int) -> Path:
    root = ROOT / "videos" if ds == "droid" else ROOT / "videos" / "rh20t"
    return root / f"chunk-000/observation.images.wrist/episode_{idx:06d}.mp4"


def run_case(base: str, predictor) -> None:
    ds, ep, total, samples, prompt_frame, pts, labels = CASES[base]
    vp = wrist_video(ds, ep)
    if not vp.exists():
        print(f"!! {base}: wrist video missing ({vp}), skip")
        return
    total = min(total, robust_video_len(str(vp)) or total)
    indices = list(range(total)) if USE_CUDA else sorted(set(samples + [prompt_frame]))
    frames = decode_video_frames(str(vp), indices)
    h, w = frames[0].shape[:2]
    k_prompt = indices.index(prompt_frame)

    if FRAME_TMP.exists():
        shutil.rmtree(FRAME_TMP)
    FRAME_TMP.mkdir(parents=True)
    for i, fr in enumerate(frames):
        cv2.imwrite(str(FRAME_TMP / f"{i:05d}.jpg"), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))

    state = predictor.init_state(
        video_path=str(FRAME_TMP), offload_video_to_cpu=True, offload_state_to_cpu=USE_CUDA
    )
    if not USE_CUDA:
        state["device"] = torch.device("cpu")
        state["storage_device"] = torch.device("cpu")
    predictor.add_new_points(
        state,
        frame_idx=k_prompt,
        obj_id=1,
        points=np.array(pts, dtype=np.float32),
        labels=np.array(labels, dtype=np.int32),
    )
    got: dict[int, np.ndarray] = {}
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16) if USE_CUDA else torch.inference_mode()
    )
    with torch.inference_mode(), autocast:
        for reverse in (False, True):
            for f_idx, _obj_ids, logits in predictor.propagate_in_video(
                state, start_frame_idx=k_prompt, reverse=reverse
            ):
                got[f_idx] = (logits[0, 0] > 0).float().cpu().numpy().astype(bool)

    full = np.zeros((total, h, w), dtype=bool)
    for k, f in enumerate(indices):
        m = got.get(k)
        if m is None:
            continue
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        full[f] = m
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"{base}_wrist_sam2.npz"
    mode = "full" if USE_CUDA else "sparse"
    np.savez_compressed(
        out,
        masks=full,
        tag=np.array(f"SAM2-{MODEL_TAG} {len(pts)}pt@f{prompt_frame} {mode}"),
        prompt_frame=np.array(prompt_frame, dtype=np.int32),
        prompt_points=np.array(pts, dtype=np.int32),
        prompt_labels=np.array(labels, dtype=np.int32),
    )
    px_counts = [int(full[f].sum()) for f in samples]
    print(f"{base}: wrote {out.name} [{MODEL_TAG}/{mode}] mask px @montage frames {dict(zip(samples, px_counts))}")


def main(bases: list[str]) -> None:
    dev = "cuda" if USE_CUDA else "cpu"
    print(f"loading SAM2 ({CONFIG}, {CKPT.name}) on {dev} ...")
    # apply_postprocessing=False: the hole-filling postprocessor JIT-compiles a CUDA
    # extension (csrc/connected_components.cu) — unavailable here (no build toolchain).
    predictor = build_sam2_video_predictor(CONFIG, str(CKPT), device=dev, apply_postprocessing=False)
    for base in bases or list(CASES):
        run_case(base, predictor)
    if FRAME_TMP.exists():
        shutil.rmtree(FRAME_TMP)


if __name__ == "__main__":
    main(sys.argv[1:])
