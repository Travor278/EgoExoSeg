# V2-SAM Remaining Ego2Exo Reproduction Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce every currently runnable Ego2Exo paper configuration with the exact public test split, and explicitly classify configurations that cannot yet be reproduced faithfully.

**Architecture:** Keep each public expert in an immutable overlay copied from public V2-SAM commit `24ae5a05ec8fe5364ef05eda113aa2b9618ad2e1`. Run B/Visual and C/Fusion independently against the same 40,517-pair test JSON; report the legacy stock metric and the complete per-object/per-pair metrics separately. Do not invent A/Anchor or PCCS behavior: those remain blocked until their GT-free runtime and asset contracts exist.

**Tech Stack:** Python 3.10, PyTorch 2.3.1+cu121, MMEngine 0.11.0rc0, torchrun/NCCL 2.20.5, eight H100 GPUs.

**Spec:** `V2SAM_EXPERIMENT_MATRIX_LIVE.md`

## Global Constraints

- Test scope is exactly 40,517 Ego2Exo pairs from JSON SHA256 `86f5803758b3c8da16e0c1f373f001a1cd14633497b05a7ce33ae6ddfcc490de`.
- Seed is `530358027`; match the public config with `deterministic=False`.
- B uses `vp_ego2exo_full.pth` SHA256 `80b3a2ab59b9453734aac2060f9c8a2f530b548b6f312f87efd7e9545aa9e3eb`.
- C uses `fusion_ego2exo_full.pth` SHA256 `f0c986c0296c3eee9a64ee5fef9f48c60e0b1de28a94cf711f6037824da6eddb`.
- Never report legacy stock, complete per-object, and per-pair metrics as the same protocol.
- Never report GT-oracle SaveMask output as PCCS.

---

### Task 1: Freeze the Completed C/Fusion Stock Result

**Files:**
- Read: remote `official-fusion-ego2exo-seed530358027-*/run.log`
- Create: remote immutable result receipt beside the run

**Interfaces:**
- Consumes: completed run with `5065/5065` and exact checkpoint/source identities.
- Produces: one receipt containing run path, source/checkpoint/data hashes, `mean_IoU`, and `mean_Dice`.

- [ ] **Step 1: Verify the terminal run contains no exception**

Run a log scan for `Traceback|RuntimeError|ChildFailedError`, require count zero, and read the MMEngine JSON result.

- [ ] **Step 2: Verify exact accounting**

Require dataset count 40,517, eight ranks, final iteration `5065/5065`, and metric JSON with finite values.

- [ ] **Step 3: Record the completed stock result**

Expected current value: stock IoU approximately `0.4469`; label it `legacy_stock`, not complete per-object.

---

### Task 2: Run B/Visual Full Ego2Exo

**Files:**
- Create: remote `v2sam-visual-official-24ae5a-overlay-v1/`
- Read: `V2sam/projects/v2sam_visual/`
- Read: `vp_ego2exo_full.pth`

**Interfaces:**
- Consumes: public Visual project, verified SAM2 and Visual checkpoints, complete test split.
- Produces: full-test legacy stock IoU/Dice JSON and log.

- [ ] **Step 1: Verify the public Visual source and checkpoint identities**

Require public commit `24ae5a05ec8fe5364ef05eda113aa2b9618ad2e1` and Visual checkpoint SHA256 `80b3a2ab59b9453734aac2060f9c8a2f530b548b6f312f87efd7e9545aa9e3eb`.

- [ ] **Step 2: Create a no-replace Visual overlay**

Copy the public repository and map `projects/v2sam_visual` to `projects/v2sam` without editing the source checkout.

- [ ] **Step 3: Launch the full test**

Use seed `530358027`, `deterministic=False`, the exact 40,517-pair JSON, and either GPUs 0-3 or all eight GPUs when no companion job is running.

- [ ] **Step 4: Verify and record the result**

Require terminal iteration, zero errors, result JSON, GPU recovery, and exact run/checkpoint/config hashes.

---

### Task 3: Run C/Fusion Complete Metric

**Files:**
- Read: `v2sam_metric_fix/seg_metric_full.py`
- Create: remote Fusion metric overlay and raw per-pair result JSON

**Interfaces:**
- Consumes: the same official C/Fusion model/checkpoint/test split as Task 1.
- Produces: `stock_IoU`, `perobject_IoU`, `perpair_IoU`, corresponding Dice values, `n_pairs=40517`, and `n_objects=100223`.

- [ ] **Step 1: Verify the complete metric source**

Require one result entry per pair and flatten only inside `compute_metrics`; reject the V2-SAM-O last-object metric.

- [ ] **Step 2: Run the complete metric over the full split**

Use a separate immutable overlay and output raw per-pair lists for independent recomputation.

- [ ] **Step 3: Independently recompute metrics**

Read raw JSON without importing project code and reproduce all six metrics and both counts exactly.

- [ ] **Step 4: Record both protocols**

Expected historical values are stock about `0.4465`, per-object about `0.4024`, and per-pair about `0.4493`; differences require investigation rather than forced matching.

---

### Task 4: Classify A/Anchor and Multi-Expert PCCS

**Files:**
- Read: `docs/superpowers/plans/2026-08-14-v2sam-pccs-strict-multiexpert.md`
- Read: `V2SAM_EXPERIMENT_MATRIX_LIVE.md`

**Interfaces:**
- Produces: explicit runnable/blocked status and exact missing prerequisites.

- [ ] **Step 1: Audit A/Anchor availability**

Require a standalone GT-free public prediction entrypoint using official SAM2 decoder plus DINO correspondence. The current public repo note alone is insufficient.

- [ ] **Step 2: Audit A+B and A+B+C PCCS availability**

Require immutable A/B/C candidate masks and a selector whose API and behavior do not depend on target GT. Reject `SegMetric_TripleDecoder_SaveMask` as oracle.

- [ ] **Step 3: Report blockers without fabricating results**

Current expected status: A requires a reviewed reconstruction; A+B and A+B+C require the unfinished strict PCCS implementation plan. These are not runnable paper-faithful products today.

---

### Task 5: Publish the Ego2Exo Matrix

**Files:**
- Modify: `V2SAM_EXPERIMENT_MATRIX_LIVE.md`

**Interfaces:**
- Consumes: verified receipts from Tasks 1-4.
- Produces: one table separating paper values, reproduced stock values, complete metrics, and blocked methods.

- [ ] **Step 1: Add exact results and evidence paths**

Record code, data, checkpoint, run, raw metric, and result hashes.

- [ ] **Step 2: Verify terminology and arithmetic**

Check percentages versus fractions and prevent validation/test or stock/per-object conflation.

- [ ] **Step 3: Run documentation validation**

Run `git diff --check -- V2SAM_EXPERIMENT_MATRIX_LIVE.md` and independently verify every reported number from immutable artifacts.
