"""Enrich gallery_manifest.json with DROID scene identity.

Chain: RoboInter video_id -> RoboInter_Data_RawPath_Qmapping.json (id -> original
gs://xembodiment_data/r2d2/r2d2-data-full/<LAB>/success/<date>/<ts>/trajectory.h5)
-> public GCS bucket gresearch/robotics/droid_raw/1.0.1/<same suffix>/metadata_*.json
-> {scene_id, lab, building, current_task, camera extrinsics}.

Writes:
  samples/gallery_manifest.json   (adds lab/scene_id/building per DROID entry)
  samples/droid_episode_meta.json (full per-episode metadata incl. extrinsics)
Queued downloads with retry; already-enriched entries are skipped (resume).
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "samples" / "gallery_manifest.json"
META_OUT = ROOT / "samples" / "droid_episode_meta.json"
RAWMAP = ROOT / "RoboInter_Data_RawPath_Qmapping.json"
GCS_PREFIX = "robotics/droid_raw/1.0.1/"
RAW_PREFIX = "gs://xembodiment_data/r2d2/r2d2-data-full/"


def http_json(url: str, tries: int = 3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"  retry ({i + 1}) {type(e).__name__}")
            time.sleep(4)


def gcs_find_metadata(suffix_dir: str) -> str | None:
    prefix = urllib.parse.quote(GCS_PREFIX + suffix_dir, safe="")
    listing = http_json(
        f"https://storage.googleapis.com/storage/v1/b/gresearch/o?prefix={prefix}&maxResults=60"
    )
    for item in listing.get("items", []):
        name = item["name"]
        if "/metadata_" in name and name.endswith(".json"):
            return name
    return None


def gcs_get(name: str):
    url = "https://storage.googleapis.com/gresearch/" + urllib.parse.quote(
        name[len("") :], safe="/"
    ).replace("+", "%2B")
    # object names start with 'robotics/...'; public direct URL is bucket/name
    return http_json(url)


def main() -> None:
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    need = [e for e in entries if e["ds"] == "droid" and "scene_id" not in e]
    print(f"droid entries to enrich: {len(need)}")
    if not need:
        print("nothing to do")
        return

    print(f"loading {RAWMAP.name} (157MB) ...")
    rawmap = json.loads(RAWMAP.read_text(encoding="utf-8"))
    wanted_ids = {f"{e['base']}_exterior_image_1_left": e["base"] for e in need}
    base2suffix: dict[str, str] = {}
    for raw_path, records in rawmap.items():
        for rec in records:
            b = wanted_ids.get(rec.get("id", ""))
            if b:
                base2suffix[b] = raw_path[len(RAW_PREFIX):].rsplit("trajectory.h5", 1)[0]
    del rawmap
    print(f"raw paths resolved: {len(base2suffix)}/{len(need)}")

    meta_all = json.loads(META_OUT.read_text(encoding="utf-8")) if META_OUT.exists() else {}
    ok = fail = 0
    for i, e in enumerate(need, 1):
        suffix = base2suffix.get(e["base"])
        if not suffix:
            fail += 1
            continue
        try:
            if e["base"] not in meta_all:
                name = gcs_find_metadata(suffix)
                if not name:
                    raise RuntimeError("metadata json not found in listing")
                meta_all[e["base"]] = gcs_get(name)
            m = meta_all[e["base"]]
            e["scene_id"] = m.get("scene_id")
            e["lab"] = m.get("lab")
            e["building"] = m.get("building")
            e["task_category"] = m.get("current_task")
            ok += 1
            print(f"[{i}/{len(need)}] {e['base']}: {e['lab']} scene={e['scene_id']}")
        except Exception as exc:
            fail += 1
            parts = suffix.strip("/").split("/")  # <LAB>/success/<date>/<ts>
            e.setdefault("lab", parts[0] if parts else None)
            e.setdefault("collect_date", parts[2] if len(parts) > 2 else None)
            print(f"[{i}/{len(need)}] {e['base']}: FAIL {type(exc).__name__}: {exc} (proxy lab={e.get('lab')})")

    META_OUT.write_text(json.dumps(meta_all, ensure_ascii=False, indent=1), encoding="utf-8")
    MANIFEST.write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
    scenes = {e.get("scene_id") for e in entries if e.get("scene_id") is not None}
    labs = {e.get("lab") for e in entries if e.get("lab")}
    print(f"done: {ok} ok / {fail} fail · unique scenes={len(scenes)} labs={len(labs)}")
    print("ENRICH-DONE")


if __name__ == "__main__":
    main()
