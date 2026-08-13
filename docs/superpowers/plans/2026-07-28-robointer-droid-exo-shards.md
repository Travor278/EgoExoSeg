# RoboInter DROID Dual-Exo Shards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, validate, deploy, and launch a resumable pipeline that converts all 56,566 fully annotated DROID dual-exo trajectories into frame-level WebDataset shards.

**Architecture:** A small Python package separates catalog construction, range downloads, source archive extraction, mask encoding, and immutable shard writing. A CLI composes those units, persists progress in SQLite, validates every trajectory and shard, and is deployed to the Dell host where a synthetic pilot precedes the background full run.

**Tech Stack:** Python 3.11+, requests, NumPy, OpenCV, PyArrow, pycocotools, Typer, SQLite, tarfile, pytest, POSIX shell, Hugging Face mirror HTTP Range.

## Global Constraints

- Pin upstream RoboInter-Data revision to `0208b79c34eca4d214cdb5d07f5ae0f7cc634340`.
- Process DROID only and require both official external-view masks.
- Expected totals are 56,566 trajectories, 14,035,692 samples, 28,071,384 JPEG images, and 113,132 selected NPZ masks.
- Preserve all synchronized frames, including zero-area masks.
- Use 180 x 320 images at 10 fps and JPEG quality 95.
- Store masks as COCO compressed RLE and verify exact decode round trips.
- Target 2 GiB immutable TAR shards and write only complete shards to the final directory.
- Use `https://hf-mirror.com` with sixteen range workers, retries, resumable state, and atomic finalization.
- Treat `shards.jsonl` as the canonical master dataset index.
- Retain CC BY-NC-SA 4.0 provenance in `dataset.json`.
- Do not modify, stage, or commit unrelated existing worktree changes.

---

### Task 1: Pair Catalog and Stable Data Models

**Files:**
- Create: `RoboInter-Data/relation_shards/__init__.py`
- Create: `RoboInter-Data/relation_shards/models.py`
- Create: `RoboInter-Data/relation_shards/catalog.py`
- Create: `RoboInter-Data/tests/test_relation_catalog.py`

**Interfaces:**
- Consumes: `VideoID_2_SegmentationNPZ.json` and `RoboInter_Data_RawPath_Qmapping.json`.
- Produces: `CameraEpisode`, `TrajectoryPair`, `build_pair_allowlist(mapping: dict) -> list[str]`, and `build_trajectory_lengths(qmapping: dict, allowlist: set[str]) -> dict[str, int]`.

- [ ] **Step 1: Write failing catalog tests**

```python
def test_build_pair_allowlist_requires_two_non_null_masks():
    mapping = {
        "10_exterior_image_1_left": "/a/10_exterior_image_1_left.npz",
        "10_exterior_image_2_left": "/a/10_exterior_image_2_left.npz",
        "11_exterior_image_1_left": "/a/11_exterior_image_1_left.npz",
        "11_exterior_image_2_left": None,
    }
    assert build_pair_allowlist(mapping) == ["10"]


def test_build_trajectory_lengths_reads_range_nop_count():
    qmapping = {
        "gs://source": [
            {"id": "10_exterior_image_1_left", "range_nop": [4, 9, 6]},
            {"id": "10_exterior_image_2_left", "range_nop": [4, 9, 6]},
        ]
    }
    assert build_trajectory_lengths(qmapping, {"10"}) == {"10": 6}
```

- [ ] **Step 2: Run the tests and verify import failure**

Run:

```powershell
python -m pytest RoboInter-Data/tests/test_relation_catalog.py -v
```

Expected: collection fails because `relation_shards.catalog` does not exist.

- [ ] **Step 3: Implement immutable models and catalog functions**

Implement:

```python
@dataclass(frozen=True, slots=True)
class CameraEpisode:
    trajectory_id: str
    camera: Literal["exterior_image_1_left", "exterior_image_2_left"]
    episode_index: int
    frame_count: int
    instruction: str


@dataclass(frozen=True, slots=True)
class TrajectoryPair:
    trajectory_id: str
    exo1: CameraEpisode
    exo2: CameraEpisode
```

`build_pair_allowlist` must sort numeric identifiers by integer value and reject malformed camera keys. `build_trajectory_lengths` must require equal `range_nop[2]` values on both views and raise `ValueError` otherwise.

- [ ] **Step 4: Run catalog tests**

Run:

```powershell
python -m pytest RoboInter-Data/tests/test_relation_catalog.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit catalog implementation**

```powershell
git add -- RoboInter-Data/relation_shards/__init__.py RoboInter-Data/relation_shards/models.py RoboInter-Data/relation_shards/catalog.py RoboInter-Data/tests/test_relation_catalog.py
git commit -m "Add RoboInter dual-exo pair catalog"
```

### Task 2: Resumable Concurrent Range Downloader

**Files:**
- Create: `RoboInter-Data/relation_shards/state.py`
- Create: `RoboInter-Data/relation_shards/download.py`
- Create: `RoboInter-Data/tests/test_range_download.py`

**Interfaces:**
- Consumes: URL, expected byte count, output path, worker count, and `PipelineState`.
- Produces: `PipelineState`, `RemoteFile`, `RangeReader.read(start: int, length: int) -> bytes`, `probe_remote(url: str) -> RemoteFile`, and `download_ranges(remote: RemoteFile, output: Path, state: PipelineState, workers: int = 16) -> Path`.

- [ ] **Step 1: Write a failing local HTTP range test**

Create a pytest HTTP handler that serves a deterministic 5 MiB byte string, honors `Range`, and records requested intervals. The test must interrupt after two completed ranges, reopen the SQLite database, resume, and assert:

```python
assert output.read_bytes() == payload
assert state.completed_range_count(remote.url) == state.total_range_count(remote.url)
assert not Path(str(output) + ".partial").exists()
```

- [ ] **Step 2: Run the downloader test and verify failure**

Run:

```powershell
python -m pytest RoboInter-Data/tests/test_range_download.py -v
```

Expected: import failure for `relation_shards.download`.

- [ ] **Step 3: Implement the SQLite state and downloader**

Use a `downloads` table keyed by `(url, start, end)` with status, attempts, and SHA-256. Preallocate `<output>.partial`, write ranges with `os.pwrite` on Linux and a locked seek/write fallback on Windows, then `os.replace` only after every range and the full expected size validate.

Retry HTTP 408, 429, and 5xx responses six times with delays `1, 2, 4, 8, 16, 30` seconds. Require status 206 for partial requests and validate `Content-Range`.

- [ ] **Step 4: Run downloader tests**

Run:

```powershell
python -m pytest RoboInter-Data/tests/test_range_download.py -v
```

Expected: interruption, resume, and corruption tests pass.

- [ ] **Step 5: Commit downloader**

```powershell
git add -- RoboInter-Data/relation_shards/state.py RoboInter-Data/relation_shards/download.py RoboInter-Data/tests/test_range_download.py
git commit -m "Add resumable mirror range downloader"
```

### Task 3: Segmentation Archive and COCO RLE Conversion

**Files:**
- Create: `RoboInter-Data/relation_shards/masks.py`
- Create: `RoboInter-Data/tests/test_masks.py`

**Interfaces:**
- Consumes: existing `segzip_index.json`, concatenated DROID prefix blob, and selected camera IDs.
- Produces: `ZipEntry = tuple[int, int, int, int]` (`local_header_offset`, `stored_size`, `method`, `raw_size`), `droid_prefix_end(entries: dict) -> int`, `extract_stored_member(blob: Path, entry: ZipEntry, output: Path) -> Path`, `normalize_masks(npz_path: Path) -> np.ndarray`, and `encode_coco_rle(mask: np.ndarray) -> dict[str, object]`.

- [ ] **Step 1: Write failing mask tests**

Generate two temporary NPZ files with shape `(1, 3, 1, 4, 5)`, including an empty frame. Assert normalization returns `(3, 4, 5)` boolean masks. Encode each mask, decode with `pycocotools.mask.decode`, and assert exact equality and `area == mask.sum()`.

Create a synthetic stored ZIP member with a local header at a nonzero offset and verify `extract_stored_member` returns the exact NPZ bytes.

- [ ] **Step 2: Run mask tests and verify failure**

Run:

```powershell
python -m pytest RoboInter-Data/tests/test_masks.py -v
```

- [ ] **Step 3: Implement archive extraction and RLE**

Parse and validate the ZIP local header signature, filename length, extra length, compression method zero, compressed size, and uncompressed size. Accept only binary or exact 0/1 masks. Convert Fortran-order arrays with `pycocotools.mask.encode` and decode byte `counts` to ASCII for JSON.

- [ ] **Step 4: Run mask tests**

Run:

```powershell
python -m pytest RoboInter-Data/tests/test_masks.py -v
```

Expected: all round-trip and malformed-input tests pass.

- [ ] **Step 5: Commit mask support**

```powershell
git add -- RoboInter-Data/relation_shards/masks.py RoboInter-Data/tests/test_masks.py
git commit -m "Add RoboInter mask extraction and RLE encoding"
```

### Task 4: TAR Mapping and Primary Video Extraction

**Files:**
- Create: `RoboInter-Data/relation_shards/source_tar.py`
- Create: `RoboInter-Data/tests/test_source_tar.py`

**Interfaces:**
- Consumes: local parquet TARs, remote video TAR URLs, and the selected trajectory allowlist.
- Produces: `read_episode_catalog(data_tar: Path, allowlist: set[str]) -> dict[int, CameraEpisode]`, `find_primary_offset(reader: RangeReader, size: int) -> int`, and `extract_selected_primary(reader: RangeReader, start: int, selected: set[int], output_dir: Path) -> dict[int, Path]`.

- [ ] **Step 1: Write failing synthetic TAR tests**

Build a TAR containing wrist members followed by primary members in shuffled episode order. Assert `find_primary_offset` identifies the exact first primary header and `extract_selected_primary` writes only requested episode indices.

Build a parquet TAR with one-row episode names and instructions and assert two camera records resolve to the expected `CameraEpisode` objects.

- [ ] **Step 2: Run TAR tests and verify failure**

Run:

```powershell
python -m pytest RoboInter-Data/tests/test_source_tar.py -v
```

- [ ] **Step 3: Implement validated TAR probing**

Validate each candidate header using the `ustar` magic, octal size, stored checksum, and a member name matching:

```text
chunk-NNN/observation.images.primary/episode_NNNNNN.mp4
```

Use 8 MiB probe windows and binary search the wrist-to-primary transition. If no validated transition exists, return offset zero so callers safely scan the full TAR.

- [ ] **Step 4: Run TAR tests**

Run:

```powershell
python -m pytest RoboInter-Data/tests/test_source_tar.py -v
```

Expected: ordered, shuffled, fallback, and truncated TAR tests pass.

- [ ] **Step 5: Commit TAR support**

```powershell
git add -- RoboInter-Data/relation_shards/source_tar.py RoboInter-Data/tests/test_source_tar.py
git commit -m "Add selective primary video extraction"
```

### Task 5: Immutable WebDataset Shard Writer

**Files:**
- Create: `RoboInter-Data/relation_shards/shards.py`
- Create: `RoboInter-Data/relation_shards/schema.py`
- Create: `RoboInter-Data/tests/test_shards.py`

**Interfaces:**
- Consumes: synchronized RGB frames, two masks, `TrajectoryPair`, and output root.
- Produces: `build_sample(...) -> Sample`, `ShardWriter.add(sample: Sample) -> None`, `ShardWriter.close() -> ShardRecord`, and `validate_shard(record: ShardRecord) -> None`.

- [ ] **Step 1: Write failing shard tests**

Write three synthetic samples with one empty mask and a 10 KiB shard limit. Assert:

```python
assert len(records) >= 2
assert all(record.sha256 for record in records)
assert total_tar_members == 3 * 3
assert parquet_rows == 3
assert decoded_empty_mask.sum() == 0
assert len(set(sample_keys)) == 3
```

- [ ] **Step 2: Run shard tests and verify failure**

Run:

```powershell
python -m pytest RoboInter-Data/tests/test_shards.py -v
```

- [ ] **Step 3: Implement schema and ordered writer**

Encode OpenCV RGB frames as JPEG with parameters:

```python
[cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_OPTIMIZE, 0]
```

Write `{key}.exo1.jpg`, `{key}.exo2.jpg`, and compact UTF-8 `{key}.json`. Close only between samples, validate the temporary TAR, compute SHA-256, write the Parquet index, then atomically move both into final directories.

- [ ] **Step 4: Run shard tests**

Run:

```powershell
python -m pytest RoboInter-Data/tests/test_shards.py -v
```

Expected: shard rollover, deterministic ordering, schema, digest, and corruption tests pass.

- [ ] **Step 5: Commit shard writer**

```powershell
git add -- RoboInter-Data/relation_shards/shards.py RoboInter-Data/relation_shards/schema.py RoboInter-Data/tests/test_shards.py
git commit -m "Add immutable dual-exo WebDataset shards"
```

### Task 6: Pipeline CLI and Synthetic End-to-End Test

**Files:**
- Create: `RoboInter-Data/relation_shards/pipeline.py`
- Create: `RoboInter-Data/relation_shards/cli.py`
- Create: `RoboInter-Data/tests/test_pipeline_e2e.py`
- Create: `RoboInter-Data/requirements-relation-shards.txt`

**Interfaces:**
- Consumes: package components from Tasks 1-5.
- Produces CLI commands `preflight`, `catalog`, `pilot`, `run`, `validate`, `status`, and `merge`.

- [ ] **Step 1: Write a failing synthetic end-to-end test**

Serve synthetic metadata, data TAR, video TAR, and segmentation bytes through the local range server. Run:

```python
result = runner.invoke(app, ["run", "--config", str(config)])
assert result.exit_code == 0
```

Then assert two trajectories, five frames, ten JPEGs, five JSON members, exact RLE round trips, a complete SQLite state, and a passing `validate` command.

- [ ] **Step 2: Run the end-to-end test and verify failure**

Run:

```powershell
python -m pytest RoboInter-Data/tests/test_pipeline_e2e.py -v
```

- [ ] **Step 3: Implement orchestration and CLI**

`preflight` verifies source revision, range support, free space, inodes, dependencies, and expected allowlist totals. `pilot` accepts `--trajectory-limit`. `run` processes deterministic chunk order and updates status after every trajectory. `status --json` returns counts suitable for monitoring. `merge` rewrites TAR members and rejects duplicate keys.

- [ ] **Step 4: Run the entire local test suite**

Run:

```powershell
python -m pytest RoboInter-Data/tests -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit CLI and integration tests**

```powershell
git add -- RoboInter-Data/relation_shards/pipeline.py RoboInter-Data/relation_shards/cli.py RoboInter-Data/tests/test_pipeline_e2e.py RoboInter-Data/requirements-relation-shards.txt
git commit -m "Add RoboInter relation shard pipeline CLI"
```

### Task 7: Remote Bootstrap and Real-Source Pilot

**Files:**
- Create: `RoboInter-Data/remote/bootstrap_relation_shards.sh`
- Create: `RoboInter-Data/remote/run_relation_shards.sh`
- Create: `RoboInter-Data/remote/status_relation_shards.sh`
- Create: `RoboInter-Data/README_relation_shards.md`

**Interfaces:**
- Consumes: committed package, Dell SSH host, and mirror.
- Produces: remote environment, `/home/dell/datasets/robointer_droid_dual_exo`, pilot shards, logs, and status commands.

- [ ] **Step 1: Write shell scripts with fixed paths and safe defaults**

Bootstrap creates:

```text
/home/dell/datasets/robointer_droid_dual_exo/{sources,work,shards,indexes,state,logs}
```

It installs dependencies into `/home/dell/miniforge3/envs/robointer-shards` and never deletes outside that dataset root. Run scripts use `flock` to prevent duplicate writers and `nohup` for the full run.

- [ ] **Step 2: Check shell syntax and CLI help locally**

Run:

```powershell
bash -n RoboInter-Data/remote/bootstrap_relation_shards.sh
bash -n RoboInter-Data/remote/run_relation_shards.sh
bash -n RoboInter-Data/remote/status_relation_shards.sh
python -m relation_shards.cli --help
```

- [ ] **Step 3: Deploy code and bootstrap the remote environment**

Copy only tracked pipeline files to `/home/dell/code/EgoExoSeg/RoboInter-Data`, run bootstrap, and execute `preflight`. Require revision, disk, inode, range, and 56,566-pair checks to pass.

- [ ] **Step 4: Run a real-source two-pair pilot**

Run `pilot --trajectory-limit 2 --shard-size-mib 64`, then `validate`. Inspect one overlay from each pair and verify both directions, empty-mask handling, and instruction text.

- [ ] **Step 5: Commit deployment scripts and documentation**

```powershell
git add -- RoboInter-Data/remote RoboInter-Data/README_relation_shards.md
git commit -m "Add remote RoboInter shard deployment"
```

### Task 8: Launch and Verify the Full Background Run

**Files:**
- Modify: `RoboInter-Data/README_relation_shards.md`

**Interfaces:**
- Consumes: passing real-source pilot.
- Produces: one active background full-run process, durable log, PID/lock, and confirmed forward progress.

- [ ] **Step 1: Start the full run**

Run the remote wrapper with `nohup`, record PID, start time, pinned revision, and command line in the SQLite state and log header.

- [ ] **Step 2: Verify the process survives session exit**

Reconnect by SSH and require:

```text
lock_held=true
process_alive=true
failed_trajectories=0
completed_ranges>0
```

- [ ] **Step 3: Verify durable forward progress**

Wait for at least one source artifact or shard to finish, run `status --json`, and confirm byte/sample counters increased across two observations.

- [ ] **Step 4: Document monitoring and final validation**

Add exact status, log-tail, restart, validate, and optional master-merge commands to the README. State that restart is idempotent and must use the same output root.

- [ ] **Step 5: Commit the operational handoff**

```powershell
git add -- RoboInter-Data/README_relation_shards.md
git commit -m "Document RoboInter full-run operations"
```

### Task 9: Final Repository Verification

**Files:**
- Verify only; no expected changes.

**Interfaces:**
- Consumes: all implementation commits and active remote run.
- Produces: evidence-backed completion report.

- [ ] **Step 1: Run local tests from a clean Python process**

```powershell
python -m pytest RoboInter-Data/tests -v
```

- [ ] **Step 2: Run static and syntax checks**

```powershell
python -m compileall -q RoboInter-Data/relation_shards
git diff --check HEAD~8..HEAD
```

- [ ] **Step 3: Confirm unrelated worktree changes remain untouched**

Run `git status --short` and compare unrelated paths with the initial status.

- [ ] **Step 4: Confirm remote state**

Require a live process, held lock, pinned source revision, a current log timestamp, no failed trajectory, and increasing completed bytes or samples.

- [ ] **Step 5: Report paths and monitoring commands**

Report the design commit, implementation commits, remote output root, active PID, current counters, and the single status command the user can run.
