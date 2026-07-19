"""Shared cross-view montage builder for the robotic dataset survey.

Renders a polished grid: columns = camera views, rows = sampled time steps.
Each camera gets a colour-coded role tag so a reader can tell at a glance
whether the same object is seen from external (exo), wrist/in-hand, or
derived views. Used by inspect_*.py in DROID/ RH20T/ RoboSet/.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---- palette (dark theme banner + light grid) -----------------------------
BG = (247, 248, 250)
PANEL = (255, 255, 255)
BANNER = (24, 28, 38)
BANNER_SUB = (150, 158, 176)
INK = (28, 32, 42)
MUTE = (120, 128, 144)
SEP = (226, 230, 236)
TILE_BG = (18, 20, 26)

ROLE_COLORS = {
    "EXO": (37, 99, 235),      # blue  -- external / global fixed camera
    "WRIST": (234, 88, 12),    # orange-- wrist / in-hand / eye-in-hand
    "DERIVED": (139, 92, 246),  # purple-- renamed / derived training view
}

_FONT_DIR = Path("C:/Windows/Fonts")


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(_FONT_DIR / name), size)
    except Exception:
        return ImageFont.load_default()


FONTS = {
    "title": _font("segoeuib.ttf", 34),
    "sub": _font("segoeui.ttf", 18),
    "cam": _font("segoeuib.ttf", 17),
    "tag": _font("segoeuib.ttf", 13),
    "row": _font("segoeuib.ttf", 16),
    "rowsub": _font("segoeui.ttf", 13),
    "foot": _font("segoeui.ttf", 14),
}


def role_of(cam: str) -> str:
    """Classify a camera name into EXO / WRIST / DERIVED for colour coding."""
    c = cam.lower()
    if any(k in c for k in ("wrist", "hand", "eye_in_hand", "handeye", "in_hand")):
        return "WRIST"
    if any(k in c for k in ("front_view", "side_view", "top_view", "primary", "secondary")):
        return "DERIVED"
    return "EXO"


def _text_w(draw, text, font):
    return draw.textbbox((0, 0), text, font=font)[2]


def letterbox(arr: np.ndarray, w: int, h: int) -> Image.Image:
    """Fit an RGB array into a w*h tile on a dark background, keeping aspect."""
    pil = Image.fromarray(arr).convert("RGB")
    pil.thumbnail((w, h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (w, h), TILE_BG)
    canvas.paste(pil, ((w - pil.width) // 2, (h - pil.height) // 2))
    return canvas


def _rounded_tag(draw, xy, text, font, color):
    x, y = xy
    pad_x, pad_y = 7, 3
    tw = _text_w(draw, text, font)
    box = (x, y, x + tw + 2 * pad_x, y + font.size + 2 * pad_y)
    draw.rounded_rectangle(box, radius=6, fill=color)
    draw.text((x + pad_x, y + pad_y - 1), text, font=font, fill=(255, 255, 255))
    return box[2]


def build_montage(
    out_path: Path,
    title: str,
    subtitle: str,
    cam_names: list[str],
    rows: list[tuple[str, str, list[np.ndarray]]],
    footer: str = "",
    tile: tuple[int, int] = (300, 225),
) -> Path:
    """rows: list of (row_label, row_sublabel, [rgb array per camera])."""
    tw, th = tile
    n_cam = len(cam_names)

    pad = 26
    banner_h = 92
    colhead_h = 62
    rowlabel_w = 118
    gap = 10
    foot_h = 40 if footer else 0

    grid_w = n_cam * tw + (n_cam - 1) * gap
    grid_h = len(rows) * th + (len(rows) - 1) * gap
    W = pad + rowlabel_w + grid_w + pad
    H = banner_h + colhead_h + grid_h + foot_h + pad

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # banner
    d.rectangle((0, 0, W, banner_h), fill=BANNER)
    d.text((pad, 20), title, font=FONTS["title"], fill=(255, 255, 255))
    d.text((pad, 60), subtitle, font=FONTS["sub"], fill=BANNER_SUB)

    grid_x0 = pad + rowlabel_w
    colhead_y = banner_h + 8

    # column headers (camera name + role tag)
    for j, cam in enumerate(cam_names):
        cx = grid_x0 + j * (tw + gap)
        role = role_of(cam)
        _rounded_tag(d, (cx, colhead_y), role, FONTS["tag"], ROLE_COLORS[role])
        # camera short name, truncated to fit tile width
        name = cam
        while _text_w(d, name, FONTS["cam"]) > tw - 6 and len(name) > 6:
            name = name[:-2]
        d.text((cx, colhead_y + 24), name, font=FONTS["cam"], fill=INK)

    grid_y0 = banner_h + colhead_h

    # rows
    for i, (label, sub, arrs) in enumerate(rows):
        ry = grid_y0 + i * (th + gap)
        # row label block
        d.text((pad, ry + th // 2 - 20), label, font=FONTS["row"], fill=INK)
        if sub:
            d.text((pad, ry + th // 2 + 2), sub, font=FONTS["rowsub"], fill=MUTE)
        for j, arr in enumerate(arrs):
            cx = grid_x0 + j * (tw + gap)
            tile_img = letterbox(arr, tw, th)
            img.paste(tile_img, (cx, ry))
            d.rectangle((cx, ry, cx + tw - 1, ry + th - 1), outline=SEP, width=1)

    if footer:
        d.text((pad, H - foot_h + 6), footer, font=FONTS["foot"], fill=MUTE)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=95)
    return out_path


def sample_indices(n: int, k: int = 4) -> list[int]:
    """Pick k roughly-even frame indices across a clip of length n."""
    if n <= 1:
        return [0]
    if n <= k:
        return list(range(n))
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def decode_video_frames(path: str, indices: list[int]) -> list[np.ndarray]:
    """Decode specific frame indices from a video file as RGB arrays (H264 path)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    out = []
    for idx in indices:
        idx = min(idx, max(total - 1, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            out.append(np.zeros((th_default(), tw_default(), 3), np.uint8))
            continue
        out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def decode_video_frames_robust(path: str, indices: list[int]) -> list[np.ndarray]:
    """Codec-robust frame decode (handles AV1 via decord/PyAV). Returns RGB arrays."""
    # 1) decord — fast random access, RGB output
    try:
        import decord

        vr = decord.VideoReader(str(path))
        n = len(vr)
        idx = [min(i, n - 1) for i in indices]
        return [vr[i].asnumpy() for i in idx]
    except Exception:
        pass
    # 2) PyAV — bundles dav1d, decode sequentially and grab target indices
    try:
        import av

        want = set(indices)
        maxi = max(indices)
        out = {}
        container = av.open(str(path))
        for i, frame in enumerate(container.decode(video=0)):
            if i in want:
                out[i] = frame.to_ndarray(format="rgb24")
            if i >= maxi:
                break
        container.close()
        last = max(out) if out else 0
        return [out.get(i, out.get(last)) for i in indices]
    except Exception:
        pass
    # 3) fallback to cv2
    return decode_video_frames(path, indices)


def robust_video_len(path: str) -> int:
    try:
        import decord

        return len(decord.VideoReader(str(path)))
    except Exception:
        return video_len(path)


def video_len(path: str) -> int:
    import cv2

    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    return n


def th_default():
    return 225


def tw_default():
    return 300
