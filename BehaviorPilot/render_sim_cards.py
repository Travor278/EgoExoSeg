"""Render the first BEHAVIOR sim cross-view GT-mask card from pilot outputs.

Inputs (per view under views/{view}/): rgb.mp4 + seg.npz
  seg.npz: seg (T,H,W) uint32 instance ids, id_map json {id: name}, task_objects json {scope: name}
Output: samples/task0000_demo1550_crossview_gt.png + per-view visibility stats.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from crossview_viz import build_montage, decode_video_frames, sample_indices  # noqa: E402

sys.path.insert(0, str(ROOT.parent / "RoboInter-Data"))
from visualize_robointer_masks import overlay_mask  # noqa: E402

VIEWS = ["exo0", "exo1", "exo2", "head", "left_wrist"]
LABELS = {
    "exo0": "exo0 (static external)",
    "exo1": "exo1 (static external)",
    "exo2": "exo2 (static external)",
    "head": "ego head (zed)",
    "left_wrist": "left wrist (realsense)",
}


def load_view(view: str):
    d = ROOT / "views" / view
    z = np.load(d / "seg.npz")
    seg = z["seg"]
    id_map = json.loads(str(z["id_map"]))
    task_objects = json.loads(str(z["task_objects"]))
    return d / "rgb.mp4", seg, id_map, task_objects


def main() -> None:
    # probe frame counts + task objects first (cheap: npz header + json fields)
    Ts, id_maps = {}, {}
    task_objects = {}
    for v in VIEWS:
        z = np.load(ROOT / "views" / v / "seg.npz")
        Ts[v] = z["seg"].shape[0]
        id_maps[v] = json.loads(str(z["id_map"]))
        task_objects = json.loads(str(z["task_objects"])) or task_objects
        del z
    print("frames per view:", Ts)
    print("task objects:", task_objects)

    # pick the manipulated object: a non-agent, non-floor scope entry (radio for task 0)
    target_names = [n for s, n in task_objects.items() if "agent" not in s and "floor" not in s.lower()]
    print("target object names:", target_names)

    def target_ids(id_map: dict) -> set[int]:
        ids = set()
        for k, name in id_map.items():
            if any(t in name for t in target_names):
                try:
                    ids.add(int(k))
                except ValueError:
                    pass
        return ids

    T = min(Ts.values())
    frames_idx = sample_indices(T, 4)
    print(f"steps={T}, montage frames={frames_idx}")

    cols, col_names = [], []
    stats = {}
    for v in VIEWS:
        seg = np.load(ROOT / "views" / v / "seg.npz")["seg"]  # one view at a time
        ids = target_ids(id_maps[v])
        id_list = list(ids)
        vis_cnt, px_list, sampled_masks = 0, [], {}
        for t in range(T):  # chunked: one frame at a time
            m = np.isin(seg[t], id_list) if id_list else np.zeros(seg[t].shape, bool)
            s = int(m.sum())
            vis_cnt += s > 0
            px_list.append(s)
            if t in frames_idx:
                sampled_masks[t] = m
        del seg
        stats[v] = (sorted(ids), vis_cnt / T, int(np.median(px_list)))
        rgb = decode_video_frames(str(ROOT / "views" / v / "rgb.mp4"), frames_idx)
        cols.append([overlay_mask(rgb[k], sampled_masks[f]) for k, f in enumerate(frames_idx)])
        col_names.append(LABELS[v])
        print(f"{v:12s} target ids={stats[v][0]} visible_frames={stats[v][1]:.0%} median_px={stats[v][2]}")

    rows = []
    for k, f in enumerate(frames_idx):
        rows.append((f"step {f}", f"t={f / 30:.1f}s", [cols[c][k] for c in range(len(VIEWS))]))
    out = build_montage(
        ROOT / "samples" / "task0000_demo1550_crossview_gt.png",
        "BEHAVIOR-1K sim pilot · turning_on_radio · demo 1550",
        "renderer GT instance masks (green) in ALL views - zero annotation · viewer-camera multi-pass replay (OG #2312 workaround)",
        col_names,
        rows,
        footer="OmniGibson replay of official teleop demo · 3 self-placed static exo cams + ego head + left wrist · masks = seg_instance of the task object",
        tile=(300, 225),
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
