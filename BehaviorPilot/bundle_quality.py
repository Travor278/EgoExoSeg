"""Score every bundled demo so weak cards can be re-run against another episode.

The production runner's acceptance test (MIN_TRACK) only asks "did the tracking
views survive without segfaulting". It cannot tell whether the robot actually
manipulated the target — a short episode where nothing happens passes cleanly and
still yields a card with dead wrist columns. This script measures that.

Usage:
    python bundle_quality.py              # table + re-run list
    python bundle_quality.py --plan       # just the demo ids, for prod_plan editing
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Same override as render_bundle_cards.py, so a production run can be scored
# without mixing it into the historical bundle/.
ROOT = Path(os.environ.get("PILOT_BUNDLE") or Path(__file__).resolve().parent / "bundle")
VIEW_ORDER = ["exo0", "exo1", "exo2", "head", "head_gaze", "left_wrist", "right_wrist"]
LIVE = 0.05      # same threshold render_bundle_cards.py uses to drop a column
WRIST_MIN = 0.20  # a card needs at least one wrist column with real coverage


def load(demo_dir: Path) -> dict | None:
    metas = {}
    for v in VIEW_ORDER:
        p = demo_dir / v / "meta.json"
        if p.exists():
            metas[v] = json.loads(p.read_text())
    if not metas:
        return None
    any_meta = next(iter(metas.values()))
    live = [v for v, m in metas.items() if m["visible_frac"] >= LIVE]
    wrists = [metas[v]["visible_frac"] for v in ("left_wrist", "right_wrist") if v in metas]
    exos = [v for v in live if v.startswith("exo")]
    return {
        "demo": demo_dir.name,
        "task": int(demo_dir.name) // 10000,
        "T": any_meta.get("Tmin", 0),
        "target": any_meta["target"].get("name", "?"),
        "n_live": len(live),
        "n_exo": len(exos),
        "wrist_best": max(wrists) if wrists else 0.0,
        "head": metas.get("head_gaze", {}).get("visible_frac", 0.0),
        "has_wrist": any(v.endswith("wrist") for v in metas),
        "live": live,
    }


def verdict(r: dict) -> str:
    """Why a card is weak, most-informative reason first.

    `empty` and `weak-wrist` must not be conflated: a cols=0 card is unusable and
    should be dropped, while a card with a live ego column and thin wrist coverage
    is still evidence of the manipulation. Both used to report "no-manipulation",
    which would have quietly deleted usable cards during final selection.
    """
    if r["n_live"] == 0:
        return "empty"             # nothing visible in any view -> discard
    if r["n_exo"] == 0:
        return "no-exo"            # ego/wrist only; exo poses were blind
    if r.get("has_wrist") and r["wrist_best"] < WRIST_MIN:
        return "weak-wrist"        # wrists barely saw the target
    if r["n_exo"] < 2:
        return "single-exo"        # lost an exo column to a segfault
    # v4 recipe is 2 exo + ego = 3 columns; the old one wanted 5 (incl. wrists).
    want = 5 if r.get("has_wrist") else 3
    if r["n_live"] < want:
        return "thin"
    return "ok"


def main() -> None:
    rows = [r for d in sorted(ROOT.iterdir()) if d.is_dir() and (r := load(d))]
    if not rows:
        print("no bundles under", ROOT)
        return
    rows.sort(key=lambda r: (r["wrist_best"], r["n_live"]))

    weak = [r for r in rows if verdict(r) not in ("ok", "thin")]
    if "--plan" in sys.argv:
        print(" ".join(r["demo"] for r in weak))
        return

    print(f"{'demo':>8} {'task':>5} {'T':>6} {'cols':>4} {'exo':>3} "
          f"{'wrist':>6} {'head':>6}  verdict")
    for r in rows:
        print(f"{r['demo']:>8} {r['task']:>5} {r['T']:>6} {r['n_live']:>4} {r['n_exo']:>3} "
              f"{r['wrist_best']:>6.0%} {r['head']:>6.0%}  {verdict(r)}")

    n = len(rows)
    print(f"\n{n} demos bundled")
    for v in ("ok", "thin", "single-exo", "weak-wrist", "no-exo", "empty"):
        c = sum(1 for r in rows if verdict(r) == v)
        print(f"  {v:>16}: {c:3d}  ({c / n:.0%})")
    # T is the useful knob: if weak cards cluster at low T, a length floor in the
    # runner is a cheaper fix than a visibility gate (no wasted render passes).
    if weak:
        ok_T = sorted(r["T"] for r in rows if verdict(r) == "ok")
        weak_T = sorted(r["T"] for r in weak)
        med = lambda a: a[len(a) // 2] if a else 0  # noqa: E731
        print(f"\n  median episode length — ok: {med(ok_T)}  weak: {med(weak_T)}")
        print(f"  re-run candidates: {len(weak)}")
        print("  python bundle_quality.py --plan   # ids for a targeted second pass")


if __name__ == "__main__":
    main()
