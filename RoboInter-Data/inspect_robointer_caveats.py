from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compact_counter(counter: Counter, limit: int | None = None) -> dict[str, int]:
    items = counter.most_common(limit)
    return {str(k): int(v) for k, v in items}


def feature_summary(meta_dir: Path) -> dict[str, Any]:
    info = load_json(meta_dir / "info.json")
    features = info.get("features", {})
    wanted = [
        "observation.images.primary",
        "observation.images.wrist",
        "annotation.segmentation",
        "Q_annotation.segmentation",
        "annotation.origin_shape",
        "episode_name",
        "camera_view",
    ]
    return {
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "fps": info.get("fps"),
        "codebase_version": info.get("codebase_version"),
        "features": {key: features.get(key) for key in wanted},
    }


def scan_parquet_chunk(chunk_dir: Path, seg_map: dict[str, Any]) -> dict[str, Any]:
    parquets = sorted(chunk_dir.glob("*.parquet"))
    cameras: Counter[str] = Counter()
    origin_shapes: Counter[str] = Counter()
    inline_seg_nonempty = 0
    inline_qseg_nonempty = 0
    rows: list[dict[str, Any]] = []

    for parquet_path in parquets:
        schema = pq.read_schema(parquet_path)
        columns = [
            col
            for col in [
                "episode_index",
                "episode_name",
                "camera_view",
                "annotation.segmentation",
                "Q_annotation.segmentation",
                "annotation.origin_shape",
            ]
            if col in schema.names
        ]
        table = pq.read_table(parquet_path, columns=columns)
        df = table.to_pandas()
        first = df.iloc[0]

        episode_name = str(first["episode_name"])
        camera_view = str(first.get("camera_view", ""))
        origin_shape = str(first.get("annotation.origin_shape", ""))
        has_inline_seg = False
        has_inline_qseg = False

        if "annotation.segmentation" in df:
            values = df["annotation.segmentation"].fillna("").astype(str)
            has_inline_seg = bool((values.str.len() > 0).any())
        if "Q_annotation.segmentation" in df:
            values = df["Q_annotation.segmentation"].fillna("").astype(str)
            has_inline_qseg = bool((values.str.len() > 0).any())

        mapped_npz = seg_map.get(episode_name)
        cameras[camera_view] += 1
        origin_shapes[origin_shape] += 1
        inline_seg_nonempty += int(has_inline_seg)
        inline_qseg_nonempty += int(has_inline_qseg)
        rows.append(
            {
                "parquet": parquet_path.name,
                "episode_index": int(first["episode_index"]),
                "episode_name": episode_name,
                "camera_view": camera_view,
                "frames": int(len(df)),
                "origin_shape": origin_shape,
                "inline_seg_nonempty": has_inline_seg,
                "inline_qseg_nonempty": has_inline_qseg,
                "has_mapped_npz": mapped_npz is not None,
                "mapped_npz": mapped_npz,
            }
        )

    mapped_rows = [row for row in rows if row["has_mapped_npz"]]
    unmapped_rows = [row for row in rows if not row["has_mapped_npz"]]
    return {
        "chunk_dir": str(chunk_dir),
        "episode_count": len(rows),
        "camera_view_counts": compact_counter(cameras, 20),
        "origin_shape_counts": compact_counter(origin_shapes),
        "inline_annotation_segmentation_nonempty_episodes": inline_seg_nonempty,
        "inline_q_annotation_segmentation_nonempty_episodes": inline_qseg_nonempty,
        "segmentation_mapping_coverage": {
            "mapped": len(mapped_rows),
            "total": len(rows),
            "ratio": round(len(mapped_rows) / len(rows), 4) if rows else None,
        },
        "sample_mapped": mapped_rows[:10],
        "sample_unmapped": unmapped_rows[:10],
    }


def droid_pair_stats(seg_map: dict[str, Any], keys: set[str] | None = None) -> dict[str, Any]:
    pairs: dict[str, dict[str, Any]] = defaultdict(dict)
    iterable = keys if keys is not None else set(seg_map)
    for key in iterable:
        if "_exterior_image_1_left" in key:
            base = key.split("_exterior_image_1_left")[0]
            pairs[base]["1"] = seg_map.get(key)
        elif "_exterior_image_2_left" in key:
            base = key.split("_exterior_image_2_left")[0]
            pairs[base]["2"] = seg_map.get(key)

    complete = {base: views for base, views in pairs.items() if "1" in views and "2" in views}
    both_mapped = sum(1 for views in complete.values() if views["1"] is not None and views["2"] is not None)
    one_mapped = sum(1 for views in complete.values() if (views["1"] is not None) ^ (views["2"] is not None))
    none_mapped = sum(1 for views in complete.values() if views["1"] is None and views["2"] is None)

    examples: dict[str, Any] = {}
    for base, views in complete.items():
        if views["1"] is not None and views["2"] is not None and "both_mapped" not in examples:
            examples["both_mapped"] = {
                "base": base,
                "image_1": views["1"],
                "image_2": views["2"],
            }
        if (views["1"] is not None) ^ (views["2"] is not None) and "one_mapped" not in examples:
            examples["one_mapped"] = {
                "base": base,
                "image_1": views["1"],
                "image_2": views["2"],
            }
        if views["1"] is None and views["2"] is None and "none_mapped" not in examples:
            examples["none_mapped"] = {
                "base": base,
                "image_1": views["1"],
                "image_2": views["2"],
            }

    return {
        "complete_exterior_pairs": len(complete),
        "both_views_mapped": both_mapped,
        "one_view_only_mapped": one_mapped,
        "neither_view_mapped": none_mapped,
        "both_views_ratio": round(both_mapped / len(complete), 4) if complete else None,
        "one_view_only_ratio": round(one_mapped / len(complete), 4) if complete else None,
        "neither_view_ratio": round(none_mapped / len(complete), 4) if complete else None,
        "examples": examples,
    }


def mapping_key_patterns(seg_map: dict[str, Any]) -> dict[str, Any]:
    tokens = [
        "exterior_image_1_left",
        "exterior_image_2_left",
        "RH20T",
        "wrist",
        "primary",
        "hand",
        "image_in_hand",
    ]
    counts = Counter()
    for key in seg_map:
        low = key.lower()
        for token in tokens:
            if token.lower() in low:
                counts[token] += 1
    return {
        "total_keys": len(seg_map),
        "non_null_values": sum(value is not None for value in seg_map.values()),
        "null_values": sum(value is None for value in seg_map.values()),
        "token_counts": compact_counter(counts),
    }


def primary_quality_summary(primary_keys: list[str], seg_map: dict[str, Any]) -> dict[str, Any]:
    patterns = Counter()
    for key in primary_keys:
        low = key.lower()
        for token in ["exterior_image_1_left", "exterior_image_2_left", "RH20T", "wrist", "primary"]:
            if token.lower() in low:
                patterns[token] += 1

    droid_pairs: dict[str, dict[str, Any]] = defaultdict(dict)
    for key, value in seg_map.items():
        if "_exterior_image_1_left" in key:
            droid_pairs[key.split("_exterior_image_1_left")[0]]["1"] = value
        elif "_exterior_image_2_left" in key:
            droid_pairs[key.split("_exterior_image_2_left")[0]]["2"] = value

    droid_primary_pair_stats = Counter()
    examples: dict[str, Any] = {}
    for key in [item for item in primary_keys if "exterior_image" in item]:
        if "_exterior_image_1_left" in key:
            base = key.split("_exterior_image_1_left")[0]
            primary_view = "1"
            other_key = f"{base}_exterior_image_2_left"
        else:
            base = key.split("_exterior_image_2_left")[0]
            primary_view = "2"
            other_key = f"{base}_exterior_image_1_left"
        primary_has_mask = seg_map.get(key) is not None
        other_has_mask = seg_map.get(other_key) is not None
        stat_key = f"primary_view_{primary_view}|primary_has_{primary_has_mask}|other_has_{other_has_mask}"
        droid_primary_pair_stats[stat_key] += 1
        examples.setdefault(
            stat_key,
            {
                "primary_key": key,
                "other_key": other_key,
                "primary_npz": seg_map.get(key),
                "other_npz": seg_map.get(other_key),
            },
        )

    return {
        "total_primary_quality_keys": len(primary_keys),
        "keys_present_in_segmentation_map": sum(key in seg_map for key in primary_keys),
        "keys_with_segmentation_npz": sum(seg_map.get(key) is not None for key in primary_keys),
        "token_counts": compact_counter(patterns),
        "droid_primary_pair_stats": compact_counter(droid_primary_pair_stats),
        "droid_primary_pair_examples": examples,
        "sample_keys": primary_keys[:20],
    }


def main() -> None:
    seg_map = load_json(ROOT / "VideoID_2_SegmentationNPZ.json")
    val_keys = set(load_json(ROOT / "val_video.json"))
    primary_keys_path = ROOT / "All_Keys_of_Primary.json"
    primary_keys = load_json(primary_keys_path) if primary_keys_path.exists() else []

    summary = {
        "source_files": {
            "segmentation_map": str(ROOT / "VideoID_2_SegmentationNPZ.json"),
            "validation_keys": str(ROOT / "val_video.json"),
            "primary_quality_keys": str(primary_keys_path) if primary_keys_path.exists() else None,
        },
        "official_metadata": {
            "droid": feature_summary(ROOT / "meta" / "lerobot_droid_anno"),
            "rh20t": feature_summary(ROOT / "meta" / "lerobot_rh20t_anno"),
        },
        "segmentation_mapping": mapping_key_patterns(seg_map),
        "droid_exterior_pair_stats_all": droid_pair_stats(seg_map),
        "droid_exterior_pair_stats_validation": droid_pair_stats(seg_map, val_keys),
        "primary_quality_keys": primary_quality_summary(primary_keys, seg_map) if primary_keys else None,
        "sampled_parquet_chunks": {
            "droid_chunk_000": scan_parquet_chunk(
                ROOT / "data" / "lerobot_droid_chunk-000" / "chunk-000",
                seg_map,
            ),
            "rh20t_chunk_000": scan_parquet_chunk(
                ROOT / "data" / "lerobot_rh20t_chunk-000" / "chunk-000",
                seg_map,
            ),
        },
    }

    output = ROOT / "caveat_check_summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {output}")
    print("Segmentation map:", summary["segmentation_mapping"])
    print("DROID all exterior pair stats:", summary["droid_exterior_pair_stats_all"])
    print("DROID validation exterior pair stats:", summary["droid_exterior_pair_stats_validation"])
    print(
        "DROID chunk coverage:",
        summary["sampled_parquet_chunks"]["droid_chunk_000"]["segmentation_mapping_coverage"],
    )
    print(
        "RH20T chunk coverage:",
        summary["sampled_parquet_chunks"]["rh20t_chunk_000"]["segmentation_mapping_coverage"],
    )


if __name__ == "__main__":
    main()
