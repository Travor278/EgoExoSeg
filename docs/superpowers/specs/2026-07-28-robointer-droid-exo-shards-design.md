# RoboInter DROID Dual-Exo Relation Shards Design

Date: 2026-07-28

## Goal

Build a frame-level dual-exo relation dataset from the DROID portion of
`InternRobotics/RoboInter-Data`.

The output keeps every synchronized frame from the 56,566 trajectories for
which both `exterior_image_1_left` and `exterior_image_2_left` have official
RoboInter segmentation NPZ files. Each logical sample contains both camera
images, both pixel masks, the instruction, stable source identifiers, and
visibility flags.

The dataset follows the logical semantics of
`jaychempan/Ego-Exo4D-Relation-Test` while using sharded WebDataset TAR files
instead of tens of millions of loose JPEG files and monolithic JSON files.

## Scope

Included:

- DROID only.
- 56,566 trajectory pairs with non-null official masks for both external views.
- All 14,035,692 synchronized timestamps, including frames where one or both
  masks are empty.
- 28,071,384 decoded 180 x 320 external-view images.
- Official RoboInter masks converted to COCO compressed RLE without geometric
  modification.
- Both logical directions, `exo1_to_exo2` and `exo2_to_exo1`, represented by
  one physical sample.

Excluded:

- RH20T.
- Wrist videos and generated wrist masks.
- DROID trajectories missing either external-view mask.
- RoboInter action/state arrays and intermediate representations other than the
  task instruction and segmentation masks.
- Original high-resolution DROID videos. This version uses the aligned
  180 x 320, 10 fps RoboInter LeRobot videos to preserve exact mask alignment.

## Source Data

Required source artifacts:

- `VideoID_2_SegmentationNPZ.json`
- `RoboInter_Data_RawPath_Qmapping.json`
- DROID LeRobot metadata
- DROID LeRobot parquet TAR chunks, used to map `episode_index` to
  `episode_name`
- DROID LeRobot video TAR chunks
- The contiguous DROID prefix of the split segmentation ZIP

The full RoboInter repository is about 408 GB and contains RH20T, wrist videos,
action/state data, annotation LMDBs, demos, and duplicate packaging that are
not needed for this output.

The selected 113,132 NPZ files occupy about 10.13 GiB. The DROID members are a
contiguous approximately 12.93 GiB prefix of the concatenated segmentation ZIP,
so the pipeline downloads that prefix instead of the complete approximately
49.7 GiB mask archive.

## Download Strategy

The destination host cannot reach `huggingface.co` directly. The tested source
is `https://hf-mirror.com`, with approximately 1.49 MB/s for one large range
and 2.58 MiB/s for sixteen concurrent ranges.

Downloads use:

- sixteen HTTP Range workers;
- resumable part files;
- fixed-size ranges recorded in a SQLite state database;
- retry with bounded exponential backoff;
- atomic rename after size and digest verification;
- source revision and remote size recorded in the manifest;
- no final filename for incomplete content.

Data and video TAR chunks are processed chunk by chunk. Temporary source chunks
are removed only after their required outputs pass validation.

Video TAR files store wrist members before primary members. The pipeline locates
the first primary TAR header through validated range probes, downloads the
primary suffix, and extracts only selected episode indices. If a TAR layout
does not match this invariant, it falls back to downloading and scanning the
whole TAR rather than guessing an offset.

## Output Layout

```text
robointer_droid_dual_exo/
  dataset.json
  shards.jsonl
  indexes/
    shard-000000.parquet
    shard-000001.parquet
    ...
  shards/
    shard-000000.tar
    shard-000001.tar
    ...
  state/
    pipeline.sqlite3
  logs/
    pipeline.log
```

Shards target 2 GiB and close only between samples. Every completed shard is
immutable and has a SHA-256 digest in `shards.jsonl`.

Each sample uses a globally unique key:

```text
droid_{trajectory_id}_{frame_index:06d}
```

and contains:

```text
{key}.exo1.jpg
{key}.exo2.jpg
{key}.json
```

The JPEG encoder uses quality 95 with fixed settings recorded in
`dataset.json`.

## Sample Schema

```json
{
  "schema_version": "1.0",
  "key": "droid_10010_000034",
  "dataset": "RoboInter-Data",
  "source": "DROID",
  "trajectory_id": "10010",
  "frame_index": 34,
  "fps": 10,
  "instruction": "move the bowl to the right",
  "relations": ["exo1_to_exo2", "exo2_to_exo1"],
  "exo1": {
    "camera": "exterior_image_1_left",
    "width": 320,
    "height": 180,
    "mask": {
      "size": [180, 320],
      "counts": "COCO_COMPRESSED_RLE"
    },
    "area": 1234,
    "visible": true
  },
  "exo2": {
    "camera": "exterior_image_2_left",
    "width": 320,
    "height": 180,
    "mask": {
      "size": [180, 320],
      "counts": "COCO_COMPRESSED_RLE"
    },
    "area": 987,
    "visible": true
  },
  "co_visible": true,
  "source_episode_indices": {
    "exo1": 3,
    "exo2": 1
  }
}
```

Empty masks are encoded as valid zero-area COCO RLE values. `visible` is
equivalent to `area > 0`; `co_visible` is the conjunction of both view flags.

The Parquet index repeats only query-relevant scalar fields and shard/member
locations. Mask RLE strings remain in the per-sample JSON.

## Data Flow

1. Resolve the pinned Hugging Face dataset revision and enumerate source files.
2. Build the 56,566-pair allowlist from the segmentation mapping.
3. Parse DROID parquet chunks to map both camera episode names to episode
   indices, lengths, and instructions.
4. Extract the selected NPZ masks from the DROID segmentation prefix.
5. For each video chunk, extract selected primary MP4 members.
6. Pair external-view episodes by trajectory ID and validate equal frame counts.
7. Decode synchronized frames, convert both masks to COCO RLE, and write
   samples to the current TAR shard.
8. Close, hash, and register each shard atomically.
9. Write the per-shard Parquet index and update the global shard manifest.
10. Run final aggregate validation and emit `dataset.json`.

Workers may decode and encode samples concurrently, but one ordered writer owns
the active TAR shard. Deterministic ordering is by numeric trajectory ID and
then frame index.

## Resume and Failure Handling

The SQLite state database records source ranges, extracted episodes, completed
trajectories, shard state, and validation results.

On restart:

- verified immutable shards are not opened or rewritten;
- an incomplete active shard is discarded and rebuilt from its first
  trajectory;
- completed downloads and extracted artifacts are reused after size checks;
- a failed trajectory is recorded with its reason and retried after the normal
  queue;
- aggregate completion fails if any selected pair is missing.

Temporary files and source chunks are retained on validation failure for
diagnosis. Cleanup happens only after the dependent shard is committed.

## Validation

Preflight validation:

- exactly 56,566 trajectory IDs;
- exactly 113,132 selected camera masks;
- both camera episode indices resolved for every trajectory;
- enough free bytes and inodes;
- mirror supports byte ranges.

Per-trajectory validation:

- two MP4 streams decode successfully;
- both video lengths and both mask lengths agree;
- frame dimensions are 180 x 320;
- masks are binary and RLE round-trips exactly;
- JPEGs decode to the expected dimensions.

Per-shard validation:

- unique sample keys;
- exactly three TAR members per sample;
- JSON schema validation;
- TAR can be read from beginning to end;
- Parquet row count equals TAR sample count;
- SHA-256 recorded after the shard is closed.

Final validation:

- 56,566 trajectories;
- 14,035,692 logical samples;
- 28,071,384 images;
- sample count by visibility state;
- no unresolved or failed trajectory;
- random visual overlay audit from early, middle, and late shards.

## Master Dataset and Optional Merge

`shards.jsonl` is the canonical master index. Training code consumes the shard
list as one dataset without physically joining TAR files.

An optional merge command can later rewrite all members into one master TAR and
build a fresh seek index. It must not use raw byte concatenation because
intermediate TAR end markers can stop standard readers. The merge is
deterministic and does not alter sample keys or JSON.

## Licensing and Provenance

The output is a derivative of RoboInter-Data and retains its
CC BY-NC-SA 4.0 terms. `dataset.json` records the upstream repository,
revision, license, construction code revision, creation time, and all encoding
parameters.

