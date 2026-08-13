"""Remote-side card bundle v2: per demo, pick the 4 MOST USEFUL steps
(joint-visibility scored: prefer steps where the target is visible in the most
views, tiebreak by wrist then head pixel count, enforce min separation), then
extract those frames (jpg) + target masks (npy) + stats into ~/behavior_pilot/bundle/.

Handles both seg.h5 (v2.2 streaming) and seg.npz (v1). Corrupt files are skipped.
"""
import json
import os
import re
import sys
from pathlib import Path

import av
import h5py
import numpy as np
import PIL.Image as I

_R = os.path.expanduser(os.environ.get("PILOT_ROOT", "~/behavior_pilot"))
ROOT = Path(_R) / "views"
OUT = Path(_R) / "bundle"
K = 8  # rows per card (sampled manipulation steps); 8 gives more usable rows
       # without repetition, min_gap below keeps them spread across the episode
VIEW_ORDER = ["exo0", "exo1", "exo2", "exo3", "exo4", "exo5", "exo6",
              "head", "head_gaze", "left_wrist", "right_wrist"]


def open_seg(d: Path):
    h5p, npzp = d / "seg.h5", d / "seg.npz"
    if h5p.exists():
        f = h5py.File(h5p, "r")
        ds = f["seg"]
        return (lambda t: ds[t][()], ds.shape[0], json.loads(f.attrs["id_map"]),
                json.loads(f.attrs["task_objects"]),
                json.loads(f.attrs["target"]) if "target" in f.attrs else None, f.close,
                json.loads(f.attrs["manipulanda"]) if "manipulanda" in f.attrs else [])
    z = np.load(npzp)
    seg = z["seg"]
    tgt = json.loads(str(z["target"])) if "target" in z.files else None
    return (lambda t: seg[t], seg.shape[0], json.loads(str(z["id_map"])),
            json.loads(str(z["task_objects"])), tgt, lambda: None, [])


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
    """Ids for @names, preferring an exact instance match.

    Two annotators are in play. seg_instance_id labels are prim paths carrying the
    INSTANCE name ("...pizza_90..."); seg_semantic labels are the bare CATEGORY
    ("pizza"). An earlier version accepted a category substring as a fallback on
    every label, which quietly destroyed instance separation: in a scene with two
    pizzas, pizza_89 and pizza_90 both matched both prim paths and got byte-identical
    masks (and boxing_gloves matched 39 ids). So: take exact instance matches when
    any exist, and only fall back to a whole-label category match, which is what a
    semantic label looks like.
    """
    digits = [(k, str(v)) for k, v in id_map.items() if str(k).isdigit()]
    exact = [int(k) for k, label in digits if any(n in label for n in names)]
    if exact:
        return exact
    cats = {re.sub(r"_\d+$", "", n) for n in names}
    return [int(k) for k, label in digits if label in cats]


# Optional demo ids on the command line. Required when two shards share one
# machine: without it each shard would also bundle the OTHER shard's demo while
# its seg.h5 is still half-written, and the resulting bundle/ dir would make the
# owning shard skip that demo as "cached".
ONLY = set(sys.argv[1:])

for demo_dir in sorted(ROOT.iterdir()):
    if not demo_dir.is_dir():
        continue
    demo = demo_dir.name
    if ONLY and demo not in ONLY:
        continue
    metas, task_objects = {}, {}
    tgt_names, tgt_info = None, None
    all_targets = []
    for v in VIEW_ORDER:
        d = demo_dir / v
        if not ((d / "seg.h5").exists() or (d / "seg.npz").exists()):
            continue
        try:
            get, T, id_map, task_objects, tgt, close, alts = open_seg(d)
        except Exception as e:
            print(f"SKIP {demo}/{v}: {type(e).__name__}: {e}", flush=True)
            continue
        metas[v] = {"get": get, "T": T, "id_map": id_map, "close": close}
        if tgt_names is None and tgt:
            tgt_names, tgt_info = [tgt["name"]], tgt
        if alts and not all_targets:
            all_targets = alts
    if not metas:
        continue
    if tgt_names is None:
        tgt_names = [n for n in task_objects.values() if "radio" in n] or list(task_objects.values())[:1]
        tgt_info = {"name": tgt_names[0]}
    Tmin = min(m["T"] for m in metas.values())
    step = max(1, Tmin // 240)
    grid = list(range(0, Tmin, step))

    # One card per manipulated object. The seg buffer already holds every instance
    # id for every frame, so a second target is a different mask lookup over the
    # SAME renders — no extra GPU at all. Tasks that move two objects ("put the
    # apple and the banana away") therefore yield two cards, each with its own
    # manipulation window and its own chosen steps.
    targets = all_targets or ([tgt_names[0]] if tgt_names else [])
    for _ti, _tname in enumerate(targets):
        tgt_names = [_tname]
        tgt_info = dict(tgt_info or {}, name=_tname)
        # primary keeps the plain demo id so existing tooling is unaffected
        out_demo = demo if _ti == 0 else f"{demo}__{_tname}"
        print(f"{demo}: target {_ti + 1}/{len(targets)} = {_tname}", flush=True)
        # pass 1: per-view target pixel counts over the grid
        px = {}
        for v, m in metas.items():
            ids = target_ids(m["id_map"], tgt_names)
            m["ids"] = ids
            px[v] = np.array([int(np.isin(m["get"](t), ids).sum()) if ids else 0 for t in grid])

        # Some steps render as a uniformly black frame in one view — the settle renders
        # do not always land the jumped-to state, and PILOT_SETTLE_RENDERS=4 only made
        # it rarer (32 of 704 sampled cells in one review set, all on the moving ego).
        # Such a frame is indistinguishable from "the object isn't visible" to a
        # reviewer, so it must never be picked as one of the K rows.
        # It is detectable from the segmentation alone: a failed frame is ONE id across
        # the whole image, where a real interior has dozens (measured: 1 vs 37).
        # Subsample, since only the ORDER of magnitude matters.
        def _blank(m, t):
            return len(np.unique(m["get"](t)[::8, ::8])) <= 2
        blank = np.zeros(len(grid), bool)
        for v, m in metas.items():
            blank |= np.array([_blank(m, t) for t in grid])
        if blank.any():
            print(f"  {int(blank.sum())}/{len(grid)} grid steps have a blank render "
                  f"in some view — excluded from sampling", flush=True)

        # joint usefulness score per grid step. "visible" now means a SUBSTANTIAL mask
        # (>=200 px), and we prefer steps where every live view clears that bar.
        VIS_PX = 200
        n_vis = sum((px[v] >= VIS_PX).astype(int) for v in metas)
        total_px = sum(np.minimum(px[v], 20000) for v in metas)
        # v4: the recipe is now "several exo + ego"; wrists are no longer rendered.
        # Everything below keyed off wrist pixels to find the manipulation window and
        # to rank steps, so fall back to the ego view, which is the close-up that
        # remains. With wrists present this is unchanged.
        wrist_px = np.minimum(px.get("left_wrist", np.zeros(len(grid)))
                              + px.get("right_wrist", np.zeros(len(grid))), 40000)
        if not any(v.endswith("wrist") for v in metas):
            wrist_px = np.minimum(px.get("head_gaze", px.get("head", np.zeros(len(grid)))), 40000)
        # A step where NO wrist camera sees the target is a dead row — the point of
        # the card is the manipulation, and those rows render as empty black tiles
        # ("object not visible"). Rank them strictly last so they are only used if
        # there genuinely aren't K manipulation steps to show.
        wrist_vis = (wrist_px >= VIS_PX).astype(int)
        score = wrist_vis * 1e15 + n_vis * 1e12 + total_px * 10 + wrist_px
        # Rank blank-render steps below every real one. Not -inf: if an episode has
        # fewer than K clean steps we still want a card, just built from the least bad
        # remainder, which is how the existing n_vis==0 exclusion already degrades.
        score = np.where(blank, -1e14, score)
        # min_gap spreads the K rows across the episode so they are not near-duplicates.
        # But a fixed gap forces a span of (K-1)*gap: with K=8 and gap=len(grid)//12=20
        # that is 140 of 240 grid points, i.e. 58% of the episode. BEHAVIOR episodes
        # spend the first stretch navigating, so the wrists only see the target for
        # ~40% — narrower than the span the gap demands. The tail rows then had to come
        # from outside the manipulation window and rendered as "object not visible"
        # (3 of 8 rows on demo 270). Shrink the gap so K rows fit inside the window that
        # actually shows manipulation, while still spreading them evenly within it.
        # Sample inside the MANIPULATION WINDOW, not across the whole episode.
        #
        # A DROID episode is 112 frames of pure tabletop manipulation, so spreading the
        # rows over the whole clip lands them all on the interaction — their wrist tiles
        # show the object from frame 22 on. A BEHAVIOR episode opens with the robot
        # driving across the house; spreading rows over all of it spends most of them on
        # navigation, where the wrist cameras face the floor. Same sampling rule, very
        # different result, because the denominators are different.
        # So bound the grid by the first and last step where a wrist actually sees the
        # target and lay the K rows out inside that, which is the window DROID gets for
        # free. Fall back to the full grid when the wrists never see it.
        lo_i, hi_i = 0, len(grid) - 1
        vis_idx = np.flatnonzero(wrist_vis)
        if len(vis_idx) >= 2:
            pad = max(1, (vis_idx[-1] - vis_idx[0]) // 10)  # a little lead-in / follow-through
            lo_i = max(0, int(vis_idx[0]) - pad)
            hi_i = min(len(grid) - 1, int(vis_idx[-1]) + pad)
            print(f"  manipulation window: grid[{lo_i}:{hi_i}] of {len(grid)} "
                  f"({(hi_i - lo_i + 1) / len(grid):.0%} of episode)", flush=True)
        win = hi_i - lo_i + 1
        in_win = (np.arange(len(grid)) >= lo_i) & (np.arange(len(grid)) <= hi_i)
        # HARD-exclude steps where no view sees the target. Ranking them last is not
        # enough: min_gap can still be forced onto one when no acceptable step sits far
        # enough from the rows already taken, and such a step renders as an all-black
        # row in EVERY column at once (demo 631220 step 223 — both the ego and the wrist
        # pass, which are separate processes, went black on the same recorded state).
        # A row that shows nothing anywhere is never worth a slot.
        alive = n_vis > 0
        score = np.where(in_win & alive, score, -np.inf)
        # Spread the rows evenly inside the window, but relax the spacing rather than
        # give a slot to a dead step: fewer, well-separated live rows beat K rows where
        # one is blank.
        chosen = []
        for min_gap in (max(1, win // (K + 1)), max(1, win // (2 * K)), 1):
            chosen = []
            for g in np.argsort(-score):
                if score[g] == -np.inf:
                    break
                if all(abs(g - c) >= min_gap for c in chosen):
                    chosen.append(int(g))
                if len(chosen) == K:
                    break
            if len(chosen) == K:
                break
        chosen.sort()
        idxs = [grid[g] for g in chosen]
        # PILOT_PIN_IDXS: reuse the frame numbers this card already has instead of
        # re-picking them. A reviewer's keep/drop verdicts are stored per (card, frame),
        # so re-sampling would silently invalidate work already done -- the verdicts
        # would point at frames that no longer exist. Pinning means a re-render can
        # swap in better camera angles at the SAME timesteps, which is the only kind of
        # fix that is safe to apply underneath someone mid-review.
        if os.environ.get("PILOT_PIN_IDXS") == "1":
            # Any view's meta carries the same idxs list, so take the first one that
            # still has a bundle on disk from the previous run.
            pinned = None
            for _prev in sorted((OUT / out_demo).glob("*/meta.json")) if (OUT / out_demo).is_dir() else []:
                pinned = json.loads(_prev.read_text()).get("idxs")
                if pinned:
                    break
            if pinned:
                idxs = [t for t in pinned if t < Tmin]
                print(f"{demo}: pinned to existing frames {idxs}", flush=True)
        print(f"{demo}: target={tgt_info['name']} chosen steps={idxs} "
              f"(n_vis={[int(n_vis[g]) for g in chosen]})", flush=True)
        # A target no camera ever sees yields no rows. Skip it — before target_ids
        # required an exact instance match, the category fallback always matched
        # something so this could not happen; now it can, and letting it through
        # crashed decode_frames on max() of an empty set, which aborted the whole
        # demo and silently dropped every target after this one.
        if not idxs:
            print(f"  SKIP {tgt_info['name']}: never visible in any view", flush=True)
            continue

        # pass 2: extract frames + masks at chosen steps
        for v, m in metas.items():
            od = OUT / out_demo / v
            od.mkdir(parents=True, exist_ok=True)
            # Views do NOT share a time base. Each was rendered at its own stride, so
            # its frame COUNT depends on the budget in force when it ran, and the ego
            # views were later re-rendered separately: 26 cards ended up with exos at
            # ~300 frames against an ego at ~121. Indexing every view with one list put
            # different moments side by side -- exo frame 66 was 20% into the episode
            # while ego frame 66 was 55% -- which is visible as columns that plainly do
            # not match. Tmin only kept the index in range; it never made it mean the
            # same thing. So sample by TIME FRACTION and convert per view.
            # The file name keeps the reference index so a card's frame numbering (and
            # any review verdicts keyed to it) stays stable.
            span = max(1, Tmin - 1)
            local = {t: min(m["T"] - 1, int(round(t / span * (m["T"] - 1)))) for t in idxs}
            frames = decode_frames(demo_dir / v / "rgb.mp4", sorted(set(local.values())))
            for t in idxs:
                lt = local[t]
                if lt in frames:
                    I.fromarray(frames[lt]).save(od / f"f{t:05d}.jpg", quality=92)
                np.save(od / f"m{t:05d}.npy",
                        np.isin(m["get"](lt), m["ids"]) if m["ids"] else np.zeros(m["get"](lt).shape, bool))
            vis_frac = float((px[v] > 0).mean())
            json.dump({"demo": out_demo, "view": v, "T": m["T"], "Tmin": Tmin, "idxs": idxs,
                       "target": tgt_info, "target_ids": m["ids"],
                       "visible_frac": vis_frac, "median_px": int(np.median(px[v]))},
                      open(od / "meta.json", "w"))
            print(f"  {v}: ids={m['ids']} vis={vis_frac:.0%}", flush=True)

    for v, m in metas.items():
        m["close"]()
print("BUNDLE-DONE", flush=True)
