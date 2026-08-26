# V2-SAM Exo2Ego Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the official Exo2Ego V2-Visual and V2-Fusion training/evaluation contracts on two independent 4×H100 distributed-training jobs and compare the resulting frame-level IoU with the paper.

**Architecture:** Freeze one config per expert under the shared `V2SAM_Exo2Ego_20260825` experiment root. Both configs reuse the audited public V2-SAM implementation, the exact `Exo2Ego_FullTrain.json`, the same image tree, deterministic seed `530358027`, resumable optimizer/scheduler/message-hub checkpoints, and an Exo2Ego test JSON. Training runs only in the platform's distributed-training service; the interactive instance is limited to config generation, static checks, and result aggregation.

**Tech Stack:** Python 3.10, PyTorch 2.3.1, MMEngine, XTuner, 4×H100 DDP per job, Playwright-controlled Qizhi UI.

**Spec:** `D:/Code/Work/EgoExoSeg/docs/V2SAM_EGO2EXO_EXPERIMENT_DIARY_AND_PLAN_20260820.md`

## Global Constraints

- Fusion and Visual each receive exactly four H100 GPUs through separate distributed-training jobs.
- Training must start from scratch: `load_from=None` and `resume=False`.
- Fusion follows the official 24-epoch schedule; Visual follows the official 12-epoch schedule unless the source config proves another Exo2Ego contract.
- Checkpoints must persist model, optimizer, scheduler, message hub, and the contrast-schedule counter.
- Official comparison uses the paper-compatible frame/pair-macro IoU; full per-object IoU remains an audit metric and must not be mixed with it.
- PCCS is not reported as reproduced unless source, candidate generation, cycle scoring, and selector evaluation are all executable and independently verified.
- No training process runs in the interactive-modeling instance.

---

### Task 1: Freeze the Exo2Ego data and source identities

**Files:**
- Create: `/inspire/hdd/project/luojianlan/zhubingwen-253108120125/V2SAM_Exo2Ego_20260825/audits/data_contract.json`
- Create: `/inspire/hdd/project/luojianlan/zhubingwen-253108120125/V2SAM_Exo2Ego_20260825/audits/source_contract.json`

**Interfaces:**
- Consumes: official `projects/v2sam_fusion/configs/v2sam.py`, `projects/v2sam_visual/configs/v2sam.py`, `Exo2Ego_FullTrain.json`, Exo2Ego test JSON, shared train/test image roots.
- Produces: immutable absolute paths, counts, SHA256 values, source commit, and expected model class for the two frozen configs.

- [ ] **Step 1: Assert the full train JSON identity**

```bash
sha256sum /inspire/ssd/project/luojianlan/zhubingwen-253108120125/v2sam-repro/train_inputs/egoexo_full_json/Exo2Ego_FullTrain.json
```

Expected: `212e7291990453b087f91dd848dd8a8a8f11a7a0a110fb257be760ca9c8c3058` and 123,381 records.

- [ ] **Step 2: Resolve and hash the Exo2Ego test JSON**

```bash
find /inspire/ssd/project/luojianlan/zhubingwen-253108120125 -type f -iname '*exo2ego*test*.json' -print
```

Expected: exactly one canonical frame-level test JSON after duplicate-content hashes are collapsed.

- [ ] **Step 3: Validate every sampled JSON image path against the selected train/test image roots**

```bash
/inspire/ssd/project/luojianlan/zhubingwen-253108120125/v2sam-repro/.venv/bin/python -I scripts/v2sam/validate_egoexo_json_paths.py --direction exo2ego
```

Expected: zero missing paths for the deterministic sample and a stored full-count audit.

### Task 2: Build and verify the frozen Fusion config

**Files:**
- Create: `/inspire/hdd/project/luojianlan/zhubingwen-253108120125/V2SAM_Exo2Ego_20260825/configs/fusion_exo2ego_official24_seed530358027.py`
- Create: `/inspire/hdd/project/luojianlan/zhubingwen-253108120125/V2SAM_Exo2Ego_20260825/audits/fusion_config_contract.txt`

**Interfaces:**
- Consumes: Task 1 paths and the official `v2sam_fusion` config.
- Produces: an importable 24-epoch `V2SAM` configuration with resumable checkpoints and frame-level evaluation.

- [ ] **Step 1: Copy the official config into a staged file and replace only environment paths, work directory, seed, and checkpoint persistence settings.**
- [ ] **Step 2: Import the staged config with the repository's known-good MMEngine.**
- [ ] **Step 3: Assert `model.type`, batch 16, accumulation 4, AdamW LR `4e-5`, 24 epochs, official warmup/cosine boundaries, Exo2Ego train/test JSONs, `load_from=None`, and `resume=False`.**
- [ ] **Step 4: Assert checkpoint hooks save optimizer and scheduler state.**
- [ ] **Step 5: Atomically rename the staged config to the final frozen path.**

### Task 3: Build and verify the frozen Visual config

**Files:**
- Create: `/inspire/hdd/project/luojianlan/zhubingwen-253108120125/V2SAM_Exo2Ego_20260825/configs/visual_exo2ego_official12_seed530358027.py`
- Create: `/inspire/hdd/project/luojianlan/zhubingwen-253108120125/V2SAM_Exo2Ego_20260825/audits/visual_config_contract.txt`

**Interfaces:**
- Consumes: Task 1 paths and the official `v2sam_visual` config.
- Produces: an importable 12-epoch `V2SAM` Visual configuration with resumable checkpoints and frame-level evaluation.

- [ ] **Step 1: Copy the official Visual config and switch both train and test contracts from Ego2Exo to Exo2Ego.**
- [ ] **Step 2: Apply only environment paths, work directory, seed, and checkpoint persistence changes.**
- [ ] **Step 3: Assert batch 16, accumulation 4, AdamW LR `4e-5`, 12 epochs, official scheduler, Exo2Ego paths, `load_from=None`, and `resume=False`.**
- [ ] **Step 4: Atomically publish the frozen config after a successful import gate.**

### Task 4: Launch two independent distributed-training jobs

**Files:**
- Create: `/inspire/hdd/project/luojianlan/zhubingwen-253108120125/V2SAM_Exo2Ego_20260825/jobs/fusion_job.json`
- Create: `/inspire/hdd/project/luojianlan/zhubingwen-253108120125/V2SAM_Exo2Ego_20260825/jobs/visual_job.json`

**Interfaces:**
- Consumes: Tasks 2 and 3 frozen configs.
- Produces: two Qizhi job IDs, each with 4×H100 and a maximum runtime sufficient for its complete schedule.

- [ ] **Step 1: Create the Fusion job with the exact command below.**

```bash
bash -lc 'cd /inspire/ssd/project/luojianlan/zhubingwen-253108120125/v2sam-repro && PYTHONPATH=<fusion-overlay>:<known-mm> .venv/bin/torchrun --nproc_per_node=4 --master_port=29657 <fusion-overlay>/tools/train.py <frozen-fusion-config> --launcher pytorch'
```

- [ ] **Step 2: Create the Visual job with an independent rendezvous port and the frozen Visual config.**
- [ ] **Step 3: Confirm four ranks, model/checkpoint load, dataset cardinality 123,381, first optimizer step, and nonzero GPU utilization for both jobs.**
- [ ] **Step 4: Store the job IDs and immutable launch commands under `jobs/`.**

### Task 5: Evaluate checkpoints and align with the paper

**Files:**
- Create: `/inspire/hdd/project/luojianlan/zhubingwen-253108120125/V2SAM_Exo2Ego_20260825/evaluations/results.csv`
- Modify: `D:/Code/Work/EgoExoSeg/docs/V2SAM_EGO2EXO_EXPERIMENT_DIARY_AND_PLAN_20260820.md`

**Interfaces:**
- Consumes: stable checkpoints from Tasks 2–4 and canonical Exo2Ego test data.
- Produces: per-object, frame/pair-macro, and legacy metrics with exact checkpoint/config hashes.

- [ ] **Step 1: Evaluate epoch 2 for each run before trusting the remainder of the trajectory.**
- [ ] **Step 2: Continue periodic evaluation without stopping training.**
- [ ] **Step 3: Compare the best single-expert results with paper targets `V2-Visual Exo2Ego=47.29` and `Fusion Exo2Ego=49.61` only under the same frame-level aggregation.**
- [ ] **Step 4: Record every result with checkpoint SHA256, raw-output SHA256, record count, object count, and metric definition.**

### Task 6: Determine PCCS reproducibility

**Files:**
- Create: `/inspire/hdd/project/luojianlan/zhubingwen-253108120125/V2SAM_Exo2Ego_20260825/audits/pccs_reproducibility.md`

**Interfaces:**
- Consumes: official repository, paper, supplement, released model package, and all three expert raw outputs.
- Produces: one of `REPRODUCIBLE_AS_RELEASED`, `REIMPLEMENTABLE_NOT_EXACT`, or `BLOCKED_MISSING_ARTIFACTS` with evidence.

- [ ] **Step 1: Verify whether selector source and an executable entry point exist in the official release.**
- [ ] **Step 2: Verify whether the supplement specifies point sampling, reverse matching, normalization, thresholds, ties, and candidate-mask serialization completely.**
- [ ] **Step 3: Run a deterministic first-32 selector parity test if code exists; otherwise classify the gap without claiming an exact reproduction.**

### Task 7: Stage the later ZSL work without consuming current GPUs

**Files:**
- Create: `D:/Code/Work/EgoExoSeg/docs/superpowers/plans/2026-08-25-v2sam-zsl-dell166.md`

**Interfaces:**
- Consumes: `Travor278/behaviorsim-crossview-seg`, `Travor278/droid-crossview-seg`, HANDAL-X JSONs, ObjectRelator, XSeg, SEEM, and PSALM releases.
- Produces: a separate `.166` inventory and ZSL execution plan; it does not alter the Exo2Ego jobs.

- [ ] **Step 1: Perform a read-only `.166` source/data inventory without storing credentials.**
- [ ] **Step 2: Map each baseline to its released inference entry point and expected JSON schema.**
- [ ] **Step 3: Run V2-SAM zero-shot only after the current Exo2Ego expert checkpoints and evaluation contract are frozen.**
