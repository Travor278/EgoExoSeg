"""Remote-side card bundle v2: per demo, pick the 4 MOST USEFUL steps
(joint-visibility scored: prefer steps where the target is visible in the most
views, tiebreak by wrist then head pixel count, enforce min separation), then
extract those frames (jpg) + target masks (npy) + stats into ~/behavior_pilot/bundle/.

Handles both seg.h5 (v2.2 streaming) and seg.npz (v1). Corrupt files are skipped.
"""
import json
import os
from pathlib import Path

import av
import h5py
import numpy as np
import PIL.Image as I

ROOT = Path(os.path.expanduser("~/behavior_pilot/views"))
OUT = Path(os.path.expanduser("~/behavior_pilot/bundle"))
K = 4
VIEW_ORDER = ["exo0", "exo1", "exo2", "head", "head_gaze", "left_wrist", "right_wrist"]


def open_seg(d: Path):
    h5p, npzp = d / "seg.h5", d / "seg.npz"
    if h5p.exists():
        f = h5py.File(h5p, "r")
        ds = f["seg"]
        return (lambda t: ds[t][()], ds.shape[0], json.loads(f.attrs["id_map"]),
                json.loads(f.attrs["task_objects"]),
                json.loads(f.attrs["target"]) if "target" in f.attrs else None, f.close)
    z = np.load(npzp)
    seg = z["seg"]
    tgt = json.loads(str(z["target"])) if "target" in z.files else None
    return (lambda t: seg[t], seg.shape[0], json.loads(str(z["id_map"])),
            json.loads(str(z["task_objects"])), tgt, lambda: None)


def decode_frames(mp4: Path, idxs: list[int]):
    want = set(idxs)
    out = {}
    with av.open(str(mp4)) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i in want:
                out[i] = fr.to_ndarray(format="rgb24")
            if i >= max(want):
                break
    return out


def target_ids(id_map: dict, names: list[str]) -> list[int]:
    return [int(k) for k, v in id_map.items() if any(n in v for n in names) and k.isdigit()]


for demo_dir in sorted(ROOT.iterdir()):
    if not demo_dir.is_dir():
        continue
    demo = demo_dir.name
    metas, task_objects = {}, {}
    tgt_names, tgt_info = None, None
    for v in VIEW_ORDER:
        d = demo_dir / v
        if not ((d / "seg.h5").exists() or (d / "seg.npz").exists()):
            continue
        try:
            get, T, id_map, task_objects, tgt, close = open_seg(d)
        except Exception as e:
            print(f"SKIP {demo}/{v}: {type(e).__name__}: {e}", flush=True)
            continue
        metas[v] = {"get": get, "T": T, "id_map": id_map, "close": close}
        if tgt_names is None and tgt:
            tgt_names, tgt_info = [tgt["name"]], tgt
    if not metas:
        continue
    if tgt_names is None:
        tgt_names = [n for n in task_objects.values() if "radio" in n] or list(task_objects.values())[:1]
        tgt_info = {"name": tgt_names[0]}
    Tmin = min(m["T"] for m in metas.values())
    step = max(1, Tmin // 240)
    grid = list(range(0, Tmin, step))

    # pass 1: per-view target pixel counts over the grid
    px = {}
    for v, m in metas.items():
        ids = target_ids(m["id_map"], tgt_names)
        m["ids"] = ids
        px[v] = np.array([int(np.isin(m["get"](t), ids).sum()) if ids else 0 for t in grid])

    # joint usefulness score per grid step. "visible" now means a SUBSTANTIAL mask
    # (>=200 px), and we prefer steps where every live view clears that bar.
    VIS_PX = 200
    n_vis = sum((px[v] >= VIS_PX).astype(int) for v in metas)
    total_px = sum(np.minimum(px[v], 20000) for v in metas)
    wrist_px = np.minimum(px.get("left_wrist", np.zeros(len(grid)))
                          + px.get("right_wrist", np.zeros(len(grid))), 40000)
    score = n_vis * 1e12 + total_px * 10 + wrist_px
    min_gap = max(1, len(grid) // 12)
    chosen = []
    for g in np.argsort(-score):
        if all(abs(g - c) >= min_gap for c in chosen):
            chosen.append(int(g))
        if len(chosen) == K:
            break
    chosen.sort()
    idxs = [grid[g] for g in chosen]
    print(f"{demo}: target={tgt_info['name']} chosen steps={idxs} "
          f"(n_vis={[int(n_vis[g]) for g in chosen]})", flush=True)

    # pass 2: extract frames + masks at chosen steps
    for v, m in metas.items():
        od = OUT / demo / v
        od.mkdir(parents=True, exist_ok=True)
        frames = decode_frames(demo_dir / v / "rgb.mp4", idxs)
        for t in idxs:
            if t in frames:
                I.fromarray(frames[t]).save(od / f"f{t:05d}.jpg", quality=92)
            np.save(od / f"m{t:05d}.npy",
                    np.isin(m["get"](t), m["ids"]) if m["ids"] else np.zeros(m["get"](t).shape, bool))
        vis_frac = float((px[v] > 0).mean())
        json.dump({"demo": demo, "view": v, "T": m["T"], "Tmin": Tmin, "idxs": idxs,
                   "target": tgt_info, "target_ids": m["ids"],
                   "visible_frac": vis_frac, "median_px": int(np.median(px[v]))},
                  open(od / "meta.json", "w"))
        print(f"  {v}: ids={m['ids']} vis={vis_frac:.0%}", flush=True)
        m["close"]()
print("BUNDLE-DONE", flush=True)
