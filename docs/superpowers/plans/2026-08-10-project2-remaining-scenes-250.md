# Project 2 Remaining-Scene 250-Task Append Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Append exactly 250 high-quality tri-view tasks to Label Studio Project 2, prioritizing the 65 DROID scene IDs not represented by the current 1,500 tasks, and finish at exactly 1,750 tasks.

**Architecture:** Build an authoritative scene map for the sealed 3,000-source pool from public DROID trajectory metadata, then select unrepresented sources with missing scenes first and underrepresented scenes second. Run `gpt-5.6-sol` with `high` reasoning through the existing Codex reverse proxy on Dell, segment with one serialized SAM3 GPU writer, apply structural/temporal and Sol visual QA, and append exactly 250 accepted tasks through the existing Label Studio writer lock. All data and model computation runs on Dell; the local machine only supervises and retains this plan.

**Tech Stack:** DROID public metadata, Python 3, `requests`, RoboInter `triview_curation`, Codex Responses API, `gpt-5.6-sol`, SAM3, Label Studio Project 2.

## Global Constraints

- Project 2 begins at exactly 1,500 tasks and must end at exactly 1,750 tasks.
- Preserve all existing task IDs, task data, predictions, annotations, and drafts; synchronization is append-only.
- Prioritize all available sources from the 65 currently uncovered official DROID scene IDs before supplementing from underrepresented covered scenes.
- Use exact model `gpt-5.6-sol` with reasoning effort `high`; use at most two concurrent API requests.
- Run all image/model processing on `dell@192.168.105.166`; do not run models locally.
- K3 request count remains `0`; Full Reset consumption remains `0`.
- Run at most one SAM3 GPU writer and hold the canonical writer locks.
- Keep the stable selection/config revision unchanged and place all new artifacts in isolated batch-50 namespaces.
- Every accepted task must identify one physical instance consistently across wrist, exo1, and exo2 and contain editable non-empty brush RLE for all visible required views.
- Every source and overlay URL referenced by newly appended tasks must return HTTP 200 before Label Studio mutation.

---

### Task 1: Freeze the 1,500-task baseline and authoritative scene map

**Files:**
- Create on Dell: `/home/dell/datasets/robointer_triview_1000/state/project2-scene-audit-20260810-v2/trajectory-scenes-3000.jsonl`
- Create on Dell: `/home/dell/datasets/robointer_triview_1000/state/project2-scene-audit-20260810-v2/scene-audit-report.json`
- Create on Dell: `/home/dell/datasets/robointer_triview_1000/state/diverse-batch-000050-v1/state/project2-before-signatures.json`

**Interfaces:**
- Consumes: sealed root `manifests/source-frames-v2.jsonl`, Label Studio Project 2 API, DROID public metadata in the `gresearch` bucket.
- Produces: a unique `trajectory_id -> scene_id/lab/building` map for all 1,800 pool trajectories and a stable signature for all 1,500 existing tasks.

- [ ] Verify Project 2 count is 1,500, the five canonical locks are free, Label Studio serves local files, GPU has at least 12 GiB free, and disk has at least 300 GiB free.
- [ ] Fetch and cache exactly one official metadata JSON per pool trajectory; fail if any trajectory is missing, ambiguous, or unreleased.
- [ ] Recompute current coverage and require exactly 499 represented scene IDs and 65 missing out of 564 before selection.
- [ ] Hash stable task identity/data/prediction/annotation fields for all 1,500 tasks and save the ordered task ID list.

### Task 2: Build and seal a scene-balanced source queue

**Files:**
- Create on Dell: `/home/dell/datasets/robointer_triview_1000/state/diverse-batch-000050-v1/audits/scene-balanced-selection-v1.jsonl`
- Create on Dell: `/home/dell/datasets/robointer_triview_1000/state/diverse-batch-000050-v1/state/selection-report.json`
- Create on Dell: `/home/dell/datasets/robointer_triview_1000/state/diverse-batch-000050-v1/manifests/source-frames-v2.jsonl`

**Interfaces:**
- Consumes: Task 1 scene map, root source manifest and selection audit, live Project 2 source/task keys.
- Produces: a sealed ordered queue of sources absent from Project 2, with `scene_id`, selection score, instruction, and source hashes.

- [ ] Write a failing contract that rejects duplicate source keys, any currently represented task key, missing media, and a selection order that does not exhaust missing-scene candidates first.
- [ ] Rank sources round-robin by scene: first one source per uncovered scene, then second/third sources per uncovered scene, then lowest-frequency represented scenes; within a scene prefer high official sanity, exo sharpness, instruction diversity, and trajectory diversity.
- [ ] Select enough sources to yield at least 250 accepted tasks under the observed batch-49 acceptance rate, bounded by the 1,659 currently unrepresented pool sources.
- [ ] Hardlink the existing sealed source/context media into the isolated batch-50 root and verify all hashes.
- [ ] Run the selection contract and seal the queue/report.

### Task 3: Run bounded Sol/high atomic target localization

**Files:**
- Create on Dell: `/home/dell/datasets/robointer_triview_1000/state/diverse-batch-000050-v1/atomic-vision-proxy-work-sol-v1/`
- Create on Dell: `/home/dell/datasets/robointer_triview_1000/state/diverse-batch-000050-v1/manifests/atomic-prompts-v1.jsonl`
- Create on Dell: `/home/dell/datasets/robointer_triview_1000/state/diverse-batch-000050-v1/state/sol-controller-status.json`

**Interfaces:**
- Consumes: Task 2 sealed source queue and the existing loopback Codex reverse proxy.
- Produces: validated atomic target identities, per-view boxes/points, receipts, and an aggregate prompt seal.

- [ ] Probe `/v1/models` and one three-view source; require response model `gpt-5.6-sol`, reasoning `high`, HTTP 200, K3 count 0, and Full Reset false.
- [ ] Process sources in bounded groups of 30 with parallelism 2 and resumable per-source receipts.
- [ ] Validate every response against the atomic schema, single-instance semantics, normalized coordinate bounds, and cross-view identity.
- [ ] Continue only until the downstream accepted pool can supply 250 tasks; retain all failures and zero-target sources in the audit.
- [ ] Seal prompts and controller status without logging API keys or base64 request bodies.

### Task 4: Run SAM3 and structural/temporal QA

**Files:**
- Create on Dell: `/home/dell/datasets/robointer_triview_1000/state/diverse-batch-000050-v1/independent-sam3-v2/`
- Create on Dell: `/home/dell/datasets/robointer_triview_1000/state/diverse-batch-000050-v1/manifests/independent-sam3-v2.jsonl`
- Create on Dell: `/home/dell/datasets/robointer_triview_1000/state/diverse-batch-000050-v1/manifests/atomic-qa-v1.jsonl`

**Interfaces:**
- Consumes: Task 3 sealed prompts and existing source/context frames.
- Produces: editable masks, overlays, temporal metrics, disposition, and sealed QA records.

- [ ] Acquire `atomic-sam3.lock`, verify free VRAM, and run one SAM3 process over new prompt tasks.
- [ ] Require one result per prompt task, expected visible roles, non-empty RLE, valid dimensions, and no SAM3 failures before QA.
- [ ] Run strict five-frame QA without lowering confidence, overlap, drift, area, or component thresholds.
- [ ] Reject masks with detached speckles, robot/gripper leakage, part-vs-whole errors, or inconsistent physical instances even when SAM3 score is 1.0.
- [ ] Seal SAM3 and QA manifests and record K3 0 / Full Reset 0.

### Task 5: Perform Sol visual QA and choose exactly 250 tasks

**Files:**
- Create on Dell: `/home/dell/datasets/robointer_triview_1000/state/diverse-batch-000050-final-v1/state/sol-overlay-qa/`
- Create on Dell: `/home/dell/datasets/robointer_triview_1000/state/diverse-batch-000050-final-v1/state/selected-250-v1.jsonl`
- Create on Dell: `/home/dell/datasets/robointer_triview_1000/state/diverse-batch-000050-final-v1/state/selection-report.json`

**Interfaces:**
- Consumes: Task 4 candidates, source/overlay contact sheets, scene frequencies after the 1,500-task baseline.
- Produces: exactly 250 accepted task rows ordered by scene diversity and quality.

- [ ] Ask exact `gpt-5.6-sol/high` to inspect each candidate overlay against its source and target identity; require same-instance, coverage, purity, and no robot/background leakage.
- [ ] Use deterministic review thresholds and retain signed receipts for every accepted/rejected candidate.
- [ ] Rank accepted candidates round-robin by newly covered scene first, then post-append scene frequency, trajectory diversity, instruction diversity, and mask quality.
- [ ] Require exactly 250 unique task keys, report how many of the 65 missing scenes become covered, and fail if fewer than 250 accepted candidates exist.
- [ ] Build final Label Studio rows with one tri-view editable prediction per task and seal the manifest.

### Task 6: Publish media and append to Label Studio

**Files:**
- Create on Dell: `/home/dell/datasets/robointer_triview_1000/media/batch50-final-v1/`
- Create on Dell: `/home/dell/datasets/robointer_triview_1000/state/diverse-batch-000050-final-v1/state/publish-preflight.json`
- Create on Dell: `/home/dell/datasets/robointer_triview_1000/state/diverse-batch-000050-final-v1/state/labelstudio-safe-append-report.json`

**Interfaces:**
- Consumes: Task 5 sealed 250-task manifest and Task 1 baseline signatures.
- Produces: Project 2 tasks 1501–1750 and an append-only transaction report.

- [ ] Publish all source, SAM3 overlay, and official-reference files by hardlink under the registered `media` root; verify every referenced URL returns HTTP 200.
- [ ] Re-read Project 2 and fail if the task count is not 1,500 or an intended task/source key already exists.
- [ ] Acquire `labelstudio.lock`, append only the 250 new tasks, and release the lock after the API confirms all writes.
- [ ] Require new task IDs to be exactly 1501–1750, with no annotations/drafts and exactly one editable prediction each.
- [ ] Save a complete transaction report and preserve concurrent human submissions on the original 1,500 tasks.

### Task 7: Verify final count, preservation, media, and scene gain

**Files:**
- Create on Dell: `/home/dell/datasets/robointer_triview_1000/state/project2-scene-audit-20260810-v3/scene-audit-report.json`
- Create on Dell: `/home/dell/datasets/robointer_triview_1000/state/diverse-batch-000050-final-v1/state/final-verification.json`

**Interfaces:**
- Consumes: live Project 2, Task 1 baseline, Task 6 transaction report, authoritative scene map.
- Produces: final evidence that the append is complete and non-destructive.

- [ ] Require Project 2 task count 1,750 and ordered IDs 1–1750 with the original 1,500 IDs unchanged.
- [ ] Compare stable signatures for the original tasks, allowing only timestamped concurrent user annotations/drafts that occurred before the append transaction.
- [ ] Verify 250/250 new tasks have a unique source/task identity, one editable prediction, valid non-empty RLE, zero annotations/drafts, and all media URLs HTTP 200.
- [ ] Recompute official scene coverage and report covered scenes, remaining scenes, buildings, labs, unique trajectories, and unique sources.
- [ ] Recheck Label Studio/tunnel health, five locks, GPU process state, K3 request count 0, Full Reset 0, and run `git diff --check` locally.
