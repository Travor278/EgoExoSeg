# V2-SAM Strict Multi-Experts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and formally evaluate A, B, C, and GT-free A+B+C+PCCS on DROID and BehaviorSim with immutable masks plus IoU, Dice, J, F, and J&F.

**Architecture:** Compose the reviewed B/C single-expert paths with a narrow A producer, run the three producers sequentially, and make PCCS a pure postprocessor over candidate masks and reverse-cycle evidence. Freeze the selected mask before a separate offline metric process receives target GT; retain the existing hidden staging, strict receipt, aggregate, and atomic publication contracts.

**Tech Stack:** Python 3.10, PyTorch/MMEngine, NumPy, OpenCV, scikit-image, COCO RLE/pycocotools, Bash, pytest, official DAVIS 2017 evaluation code pinned at `ac7c43fca936f9722837b7fbd337d284ba37004b`.

**Spec:** `docs/superpowers/specs/2026-08-14-v2sam-pccs-strict-multiexpert-design.md`

## Global Constraints

- Implement in an isolated worktree forked from reviewed cross-view commit `8649bc30e91c0feab00fa204637178e82605dbba`.
- Claim label is `paper_faithful_reconstruction`; never label the result as unreleased official Multi-Experts code.
- PCCS and all candidate producers must not receive target GT, a target-GT path, or a target-derived metric.
- A uses official SAM2 plus DINOv3 and has no expert checkpoint; B uses the 964-tensor Visual checkpoint without DINO; C uses the 1,335-tensor Fusion checkpoint plus the exact 368-tensor external DINO closure.
- Run candidate producers serially at batch size 1; do not share mutable model instances in the first implementation.
- Canonical masks are COCO RLE JSON; selected RLE must be byte-identical to its selected candidate RLE.
- DROID/BehaviorSim J/F/J&F are `davis_style_auxiliary`, not DAVIS benchmark scores.
- Formal product failure produces no final run root and is not automatically retried.
- Preserve unrelated files, protected `.166` services, existing B/C formal artifacts, and current checkpoint/data SHA identities.

---

## File Structure

Create focused modules rather than adding all logic to the existing large job-spec file:

- `v2sam_joint/pccs_selector.py`: pure mask/evidence validation and deterministic GT-free selection.
- `v2sam_joint/pccs_masks.py`: canonical binary COCO RLE serialization and exact mask artifact validation.
- `third_party/davis2017/metrics.py`: pinned upstream J/F implementation.
- `third_party/davis2017/LICENSE`: upstream BSD-3-Clause license.
- `v2sam_joint/davis_metrics.py`: strict wrapper producing IoU/Dice/J/F/J&F rows.
- `v2sam_joint/pccs_cycle.py`: target-mask-to-query reverse DINO cycle evidence.
- `v2sam_joint/pccs_job.py`: inference/metric manifest validation, selection, scoring, bundle validation, and CLI.
- `projects/v2sam_anchor/models/v2sam.py`: narrow A producer using reviewed SAM2/DINO components.
- `projects/v2sam_anchor/models/__init__.py`: explicit A model exports.
- `projects/v2sam_anchor/configs/v2sam.py`: A evaluation config with the existing dataset/runner conventions.
- `v2sam_joint/seg_metric_full.py`: optional canonical prediction writer used by reviewed B/C paths.
- `v2sam_joint/eval_runtime_overlay.py`: exact A/ABC overlay roles and prediction-output binding.
- `v2sam_joint/crossview_job_spec.py`: ABC job schema, artifact closure, provenance, finalize, and aggregate validation.
- `scripts/eval_expert_matrix.sh`: replace blocked triple preflight with exact A/ABC preflight and sequential candidate execution.
- `scripts/eval_crossview_zero_shot.sh`: admit `A` and `ABC`, preserve smoke-before-full scheduling.
- `tests/test_pccs_selector.py`: selector and GT-invariance tests.
- `tests/test_pccs_masks.py`: RLE and artifact validation tests.
- `tests/test_davis_metrics.py`: official-metric parity and edge cases.
- `tests/test_pccs_cycle.py`: reverse-cycle evidence contracts.
- `tests/test_pccs_job.py`: manifest isolation, bundle, scoring, tamper, and CLI tests.
- `tests/test_anchor_expert.py`: real-config A schema and fake-tensor strict-load tests.
- `tests/test_eval_runtime_overlay.py`: A/ABC overlay positive and tamper cases.
- `tests/test_eval_expert_matrix.py`: execution-plan, formal chain, aggregate, and publication integration.

---

### Task 1: Pure GT-Free PCCS Selector

**Files:**
- Create: `v2sam_joint/pccs_selector.py`
- Create: `tests/test_pccs_selector.py`

**Interfaces:**
- Consumes: query binary mask, three candidate masks, and reverse query-view points for experts A/B/C.
- Produces: `select_pccs(query_mask: np.ndarray, candidates: Mapping[str, np.ndarray], reverse_points: Mapping[str, Sequence[Sequence[float]]]) -> SelectionDecision` and `decision_document(decision: SelectionDecision) -> dict[str, object]`.

- [ ] **Step 1: Write selector RED tests**

```python
def test_selector_prefers_inside_count_then_cycle_distance():
    query = square_mask(32, 8, 24)
    masks = {name: square_mask(32, 4, 28) for name in ("A", "B", "C")}
    points = {
        "A": [(9.0, 9.0), (10.0, 10.0)],
        "B": [(9.0, 9.0), (10.0, 10.0), (40.0, 40.0)],
        "C": [(12.0, 12.0), (13.0, 13.0)],
    }
    decision = select_pccs(query_mask=query, candidates=masks,
                           reverse_points=points)
    assert decision.selected_expert == "A"
    assert decision.reason == "pccs_cycle_points"

@pytest.mark.parametrize("replacement", ["random", "zero", "absent"])
def test_selector_api_has_no_target_gt_and_is_gt_invariant(replacement):
    assert "target_gt" not in inspect.signature(select_pccs).parameters
```

Add explicit RED cases for candidate permutation, exact ties, only one valid
expert, no cycle points (`C,B,A` fallback), empty query, missing expert,
nonbinary mask, shape mismatch, NaN/Inf points, and duplicate expert keys.

- [ ] **Step 2: Run selector tests and record RED**

Run:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_pccs_selector.py
```

Expected: collection fails because `v2sam_joint.pccs_selector` does not exist.

- [ ] **Step 3: Implement the minimal pure selector**

```python
EXPERT_ORDER = ("A", "B", "C")
FALLBACK_ORDER = ("C", "B", "A")

@dataclass(frozen=True)
class CandidateEvidence:
    expert: str
    mask_sha256: str
    inside_count: int
    cycle_distance: float | None
    valid_cycle: bool

@dataclass(frozen=True)
class SelectionDecision:
    selected_expert: str
    reason: str
    evidence: tuple[CandidateEvidence, ...]

def select_pccs(*, query_mask: np.ndarray,
                 candidates: Mapping[str, np.ndarray],
                 reverse_points: Mapping[str, Sequence[Sequence[float]]]
                 ) -> SelectionDecision:
    query = require_binary_mask(query_mask, "query_mask")
    require_exact_experts(candidates)
    require_exact_experts(reverse_points)
    query_points = representative_query_points(query)
    evidence = tuple(
        build_candidate_evidence(
            expert=expert,
            query_mask=query,
            query_points=query_points,
            candidate=require_binary_mask(candidates[expert], expert),
            reverse_points=reverse_points[expert],
        )
        for expert in EXPERT_ORDER
    )
    valid = [item for item in evidence if item.valid_cycle]
    if valid:
        selected = min(
            valid,
            key=lambda item: (
                -item.inside_count,
                item.cycle_distance,
                EXPERT_ORDER.index(item.expert),
            ),
        )
        reason = "pccs_cycle_points"
    else:
        selected = next(
            item for expert in FALLBACK_ORDER
            for item in evidence
            if item.expert == expert
        )
        reason = "fallback_empty_query" if not query.any() else "fallback_no_cycle"
    return SelectionDecision(selected.expert, reason, evidence)
```

Private helpers `require_binary_mask`, `require_exact_experts`,
`representative_query_points`, and `build_candidate_evidence` are defined in
the same file. They validate finite points, compute deterministic
center-and-contour query points, calculate inside counts and mean cycle
distance, and never accept a target-GT argument.

- [ ] **Step 4: Run selector tests and verify GREEN**

Run the Step 2 command. Expected: all selector cases pass with no GPU process.

- [ ] **Step 5: Commit Task 1**

```bash
git add v2sam_joint/pccs_selector.py tests/test_pccs_selector.py
git commit -m "feat: add deterministic GT-free PCCS selector"
```

---

### Task 2: Canonical Mask Artifacts and DAVIS Metrics

**Files:**
- Create: `v2sam_joint/pccs_masks.py`
- Create: `third_party/davis2017/metrics.py`
- Create: `third_party/davis2017/LICENSE`
- Create: `v2sam_joint/davis_metrics.py`
- Create: `tests/test_pccs_masks.py`
- Create: `tests/test_davis_metrics.py`

**Interfaces:**
- Produces: `encode_binary_mask(mask) -> dict`, `decode_binary_mask(doc) -> np.ndarray`, `canonical_mask_bytes(mask) -> bytes`, and `score_binary_mask(pred, gt) -> dict[str, float]`.
- Consumes later: Task 5 candidate writer and Task 6 offline scorer.

- [ ] **Step 1: Write canonical RLE RED tests**

```python
def test_canonical_rle_round_trip_is_byte_stable():
    mask = np.zeros((17, 23), dtype=np.uint8)
    mask[3:11, 7:19] = 1
    first = canonical_mask_bytes(mask)
    second = canonical_mask_bytes(decode_binary_mask(json.loads(first)))
    assert first == second
```

Add rejection cases for bool/uint8 normalization disagreements, nonbinary
values, wrong dimensions, duplicate JSON keys, extra fields, malformed RLE,
and size mismatch.

- [ ] **Step 2: Write DAVIS metric RED tests**

Vendor `davis2017/metrics.py` from upstream commit
`ac7c43fca936f9722837b7fbd337d284ba37004b`, then write golden tests that
call both the vendored function and `score_binary_mask`:

```python
def test_wrapper_matches_pinned_davis_for_shifted_boundary():
    gt = square_mask(64, 16, 48)
    pred = np.roll(gt, 3, axis=1)
    score = score_binary_mask(pred, gt)
    assert score["J"] == pytest.approx(db_eval_iou(gt, pred), abs=1e-12)
    assert score["F"] == pytest.approx(db_eval_boundary(gt, pred), abs=1e-12)
    assert score["J&F"] == pytest.approx((score["J"] + score["F"]) / 2)
```

Cover identical, both-empty, one-empty, disjoint, one-pixel, and shifted masks.

- [ ] **Step 3: Run mask/metric tests and record RED**

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/test_pccs_masks.py tests/test_davis_metrics.py
```

Expected: wrapper and mask modules are missing.

- [ ] **Step 4: Implement canonical masks and strict metric wrapper**

```python
def score_binary_mask(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = require_binary_2d(pred, "prediction")
    gt = require_binary_2d(gt, "ground_truth")
    intersection = int(np.logical_and(pred, gt).sum())
    pred_area, gt_area = int(pred.sum()), int(gt.sum())
    union = pred_area + gt_area - intersection
    iou = 1.0 if union == 0 else intersection / union
    dice = 1.0 if pred_area + gt_area == 0 else 2 * intersection / (pred_area + gt_area)
    j = float(db_eval_iou(gt, pred))
    f = float(db_eval_boundary(gt, pred))
    return {"IoU": iou, "Dice": dice, "J": j, "F": f, "J&F": (j + f) / 2}
```

Pin and record the upstream commit and vendored file SHA in module constants.

- [ ] **Step 5: Run tests and verify GREEN**

Run the Step 3 command. Expected: all golden and tamper cases pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add v2sam_joint/pccs_masks.py v2sam_joint/davis_metrics.py \
  third_party/davis2017 tests/test_pccs_masks.py tests/test_davis_metrics.py
git commit -m "feat: add canonical masks and DAVIS-style metrics"
```

---

### Task 3: Inference Isolation and Prediction Bundle Contract

**Files:**
- Create: `v2sam_joint/pccs_job.py`
- Create: `tests/test_pccs_job.py`

**Interfaces:**
- Produces: `validate_inference_manifest(path, trusted_root)`, `select_bundle(job_root, inference_manifest, output)`, `score_bundle(selected_root, metric_manifest, output)`, and `validate_bundle(job_root, spec)`.
- Consumes: selector from Task 1 and RLE/metrics from Task 2.

- [ ] **Step 1: Write split-manifest RED tests**

```python
def test_inference_manifest_rejects_target_gt_fields(tmp_path):
    manifest = valid_inference_manifest(tmp_path)
    manifest["records"][0]["target_mask"] = "/forbidden/gt.json"
    path = write_canonical(tmp_path / "inference.json", manifest)
    with pytest.raises(PccsContractError, match="target GT"):
        validate_inference_manifest(path, tmp_path)
```

Add real/random/zero/absent target-GT invariance, metric-before-selection
rejection, selected/candidate byte mismatch, missing/extra/duplicate artifact,
outside path, symlink, special file, noncanonical JSON, and self-consistent
tamper cases.

- [ ] **Step 2: Run bundle tests and record RED**

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_pccs_job.py
```

Expected: `PccsContractError` and job APIs are missing.

- [ ] **Step 3: Implement strict documents and no-replace CLI outputs**

```python
def select_bundle(job_root: Path, inference_manifest: Path,
                  output: Path) -> dict[str, object]:
    inference = validate_inference_manifest(inference_manifest, job_root)
    decisions = []
    for record in inference["records"]:
        candidates = load_exact_candidates(job_root, record)
        cycle = load_exact_cycle_evidence(job_root, record)
        decision = select_pccs(query_mask=load_query_mask(record),
                               candidates=candidates,
                               reverse_points=cycle)
        decisions.append(publish_selected_mask(job_root, record, decision))
    document = canonical_selection_document(inference, decisions)
    publish_json_noreplace(output, document)
    return document
```

CLI subcommands are exact and single-use:

```text
python -m v2sam_joint.pccs_job select --job-root ABS --inference ABS --output ABS
python -m v2sam_joint.pccs_job score --job-root ABS --metric-manifest ABS --output ABS
python -m v2sam_joint.pccs_job validate --job-root ABS --spec ABS
```

- [ ] **Step 4: Run bundle tests and verify GREEN**

Run the Step 2 command. Expected: all isolation, publication, and tamper cases pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add v2sam_joint/pccs_job.py tests/test_pccs_job.py
git commit -m "feat: add isolated PCCS prediction bundle contract"
```

---

### Task 4: Strict A/Anchor Producer

**Files:**
- Create: `projects/v2sam_anchor/models/v2sam.py`
- Create: `projects/v2sam_anchor/models/__init__.py`
- Create: `projects/v2sam_anchor/configs/v2sam.py`
- Create: `tests/test_anchor_expert.py`

**Interfaces:**
- Produces: MMEngine-compatible `V2SAMAnchor.predict(data)` returning `pred_masks` and anchor/cycle inputs in target image coordinates.
- Consumes: reviewed Fusion DINO correspondence utility and official SAM2 config/checkpoint, but no B/C checkpoint.

- [ ] **Step 1: Write real-config and fake-state RED tests**

```python
def test_anchor_config_has_no_expert_checkpoint_or_trainable_matcher():
    cfg = Config.fromfile(ANCHOR_CONFIG)
    assert cfg.model.type is V2SAMAnchor
    assert "expert_checkpoint" not in cfg.model
    assert cfg.model.train_cfg is None

def test_anchor_strictly_loads_official_sam2_schema(fake_sam2_state):
    model = build_anchor_with_fake_external_modules()
    result = model.sam2_model.load_state_dict(fake_sam2_state, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
```

Add no-target-GT signature, target-size-from-image, 900-tensor SAM2 schema,
368-tensor DINO closure, frozen-state, missing/extra/shape, source symlink,
wrong hash, and post-inference mutation cases.

- [ ] **Step 2: Run A tests and record RED**

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_anchor_expert.py
```

Expected: anchor package/config are absent.

- [ ] **Step 3: Implement the narrow A producer**

```python
class V2SAMAnchor(nn.Module):
    def __init__(self, sam2_cfg, dinov3_cfg, train_cfg=None, test_cfg=None):
        super().__init__()
        if train_cfg is not None:
            raise ValueError("A is inference-only")
        self.sam2_model = build_sam2(sam2_cfg)
        self.correspondence = SparseCorrespondence(**dinov3_cfg)

    @torch.no_grad()
    def predict(self, data):
        reject_target_gt_keys(data)
        target_hw = decoded_target_hw(data["target_img_path"])
        points = self.correspondence.query_to_target(
            data["query_img_path"], data["target_img_path"],
            data["raw_prompt_masks"])
        return self.sam2_model.predict_from_points(data, points, target_hw)
```

Use existing reviewed constructors and coordinate transforms; do not copy old
TripleDecoder GT arguments.

- [ ] **Step 4: Run A tests plus existing B/C overlay tests**

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/test_anchor_expert.py tests/test_eval_runtime_overlay.py
```

Expected: A tests pass and B/C behavior remains unchanged.

- [ ] **Step 5: Commit Task 4**

```bash
git add projects/v2sam_anchor tests/test_anchor_expert.py
git commit -m "feat: add strict inference-only anchor expert"
```

---

### Task 5: Candidate Mask Writer and Reverse-Cycle Evidence

**Files:**
- Create: `v2sam_joint/pccs_cycle.py`
- Create: `tests/test_pccs_cycle.py`
- Modify: `v2sam_joint/seg_metric_full.py`
- Modify: `tests/test_seg_metric_full.py`
- Modify: `v2sam_joint/eval_runtime_overlay.py`
- Modify: `tests/test_eval_runtime_overlay.py`

**Interfaces:**
- Produces: optional canonical candidate RLE files from A/B/C and `project_reverse_cycle(query_image: Path, target_image: Path, candidate_mask: np.ndarray, query_mask: np.ndarray, correspondence: SparseCorrespondence) -> tuple[tuple[float, float], ...]`.
- Consumes: exact prediction-output path from overlay and DINO closure from Task 4.

- [ ] **Step 1: Write candidate-writer and cycle RED tests**

```python
def test_segmetric_prediction_writer_matches_scored_mask(tmp_path):
    metric = SegMetricFull(prediction_output=tmp_path / "candidates")
    metric.process(one_object_batch(), one_object_prediction())
    written = decode_binary_mask(load_only_rle(tmp_path))
    assert np.array_equal(written, expected_prediction_mask())

def test_reverse_cycle_has_no_target_gt_parameter():
    assert "target_gt" not in inspect.signature(project_reverse_cycle).parameters
```

Add target-image-size, object-ID ordering, duplicate output, DINO mutation,
NaN points, source candidate tamper, and random/zero/absent GT invariance.

- [ ] **Step 2: Run focused tests and record RED**

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/test_seg_metric_full.py tests/test_pccs_cycle.py \
  tests/test_eval_runtime_overlay.py
```

Expected: prediction-output and cycle APIs are absent.

- [ ] **Step 3: Implement optional writer and pure cycle adapter**

```python
def project_reverse_cycle(*, query_image: Path, target_image: Path,
                          candidate_mask: np.ndarray,
                          query_mask: np.ndarray,
                          correspondence: SparseCorrespondence
                          ) -> tuple[tuple[float, float], ...]:
    require_binary_same_target(candidate_mask, target_image)
    points = correspondence.target_to_query(
        target_image, query_image, candidate_mask)
    return canonical_finite_points(points)
```

Overlay schema must bind a canonical contained prediction directory and reject
it for training modes. Existing invocations without the field retain their
current output bytes.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: writer/cycle pass and existing metrics remain exact.

- [ ] **Step 5: Commit Task 5**

```bash
git add v2sam_joint/pccs_cycle.py v2sam_joint/seg_metric_full.py \
  v2sam_joint/eval_runtime_overlay.py tests/test_pccs_cycle.py \
  tests/test_seg_metric_full.py tests/test_eval_runtime_overlay.py
git commit -m "feat: publish expert masks and cycle evidence"
```

---

### Task 6: ABC Job Spec, Sequential Coordinator, and Offline Scoring

**Files:**
- Modify: `v2sam_joint/crossview_job_spec.py`
- Modify: `scripts/eval_expert_matrix.sh`
- Modify: `scripts/eval_crossview_zero_shot.sh`
- Modify: `tests/test_eval_expert_matrix.py`
- Modify: `tests/test_crossview_benchmark_contract.py`

**Interfaces:**
- Produces: execution plans for `A` and `ABC`, sequential A/B/C subjobs,
  immutable selection, offline scoring, and formal job receipts.
- Consumes: all Task 1-5 APIs.

- [ ] **Step 1: Write plan/preflight RED tests**

```python
@pytest.mark.parametrize("experts", ["A", "ABC"])
def test_crossview_plan_accepts_strict_multiexpert_modes(tmp_path, experts):
    result = run_script("--print-plan", "--experts", experts,
                        "--datasets", "DROID", "--stages", "smoke")
    assert result.returncode == 0
    plan = json.loads(result.stdout)
    assert plan["experts"] == [experts]
    assert plan["selector"] == (
        "single-expert" if experts == "A" else "pccs_cycle_points_strict")
```

Add duplicate/unknown selectors, exact asset roles, B-no-DINO, A/C-DINO,
inference/metric manifest split, smoke-before-full, batch-size-1, output-root,
and no GT oracle in formal ranking.

- [ ] **Step 2: Write real formal-chain RED fixture**

The fake executor must emit deterministic A/B/C masks and reverse points, run
the production `pccs_job select`, run production offline score, finalize,
atomically publish, and aggregate. Assert candidate and selected hashes,
fallback counts, all five metrics, accounting zero, and no staging path in
structured JSON.

- [ ] **Step 3: Run integration RED**

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/test_eval_expert_matrix.py \
  tests/test_crossview_benchmark_contract.py \
  -k 'pccs or anchor or multiexpert'
```

Expected: scripts reject A/ABC and `preflight_triple` remains blocked.

- [ ] **Step 4: Implement plan and sequential execution**

Update accepted selectors and replace hard blocking with exact checks. The
formal ABC shell sequence is fixed:

```bash
run_candidate A "$job_root/candidates/A"
run_candidate B "$job_root/candidates/B"
run_candidate C "$job_root/candidates/C"
python -m v2sam_joint.pccs_job select \
  --job-root "$job_root" --inference "$inference_manifest" \
  --output "$job_root/selection.json"
python -m v2sam_joint.pccs_job score \
  --job-root "$job_root" --metric-manifest "$metric_manifest" \
  --output "$job_root/raw_metrics.json"
python -m v2sam_joint.pccs_job validate \
  --job-root "$job_root" --spec "$frozen_spec"
```

All arguments are canonical absolute paths validated before the first writer.

- [ ] **Step 5: Run integration tests and verify GREEN**

Run the Step 3 command. Expected: exact A and ABC fake-formal chains pass.

- [ ] **Step 6: Commit Task 6**

```bash
git add v2sam_joint/crossview_job_spec.py scripts/eval_expert_matrix.sh \
  scripts/eval_crossview_zero_shot.sh tests/test_eval_expert_matrix.py \
  tests/test_crossview_benchmark_contract.py
git commit -m "feat: orchestrate strict ABC PCCS jobs"
```

---

### Task 7: Published Artifact and Aggregate Closure

**Files:**
- Modify: `v2sam_joint/crossview_job_spec.py`
- Modify: `tests/test_eval_expert_matrix.py`
- Modify: `tests/test_pccs_job.py`

**Interfaces:**
- Produces: exact published artifact sets and aggregate documents containing A/B/C/PCCS summaries and claim labels.
- Consumes: Task 6 job tree and receipts.

- [ ] **Step 1: Write publication-tamper RED matrix**

Create a fully valid fake formal output, then independently mutate:

```python
@pytest.mark.parametrize("mutation", [
    "missing_candidate", "extra_candidate", "changed_candidate",
    "changed_selected", "changed_cycle", "changed_selection",
    "receipt_only", "manifest_only", "symlink", "special",
    "self_consistent_selected_tamper", "gt_field_in_selection",
])
def test_published_pccs_closure_rejects_mutation(valid_run, mutation):
    mutate(valid_run, mutation)
    with pytest.raises(CrossviewContractError):
        aggregate_run(valid_run, ["ABC"], valid_run.parent,
                      datasets=["DROID"], stages=["smoke"])
```

- [ ] **Step 2: Run tamper tests and record RED**

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/test_pccs_job.py tests/test_eval_expert_matrix.py \
  -k 'published and pccs'
```

Expected: current aggregate ignores at least one new artifact class.

- [ ] **Step 3: Implement exact manifest/receipt/aggregate sets**

Each artifact entry is exact `{path, sha256, bytes, role, record_id,
object_id}`. Receipt artifacts equal manifest artifacts plus the job manifest.
Aggregate recomputes all bytes/hashes and asserts:

```python
selected_sha == candidate_sha[decision["selected_expert"]]
summary["claim_level"] == "paper_faithful_reconstruction"
summary["jf_claim_level"] == "davis_style_auxiliary"
```

- [ ] **Step 4: Run tamper and existing aggregate tests**

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/test_pccs_job.py tests/test_eval_expert_matrix.py
```

Expected: all new and existing B/C artifact cases pass.

- [ ] **Step 5: Commit Task 7**

```bash
git add v2sam_joint/crossview_job_spec.py \
  tests/test_eval_expert_matrix.py tests/test_pccs_job.py
git commit -m "feat: close published PCCS artifact provenance"
```

---

### Task 8: Full CPU Regression and Independent Review Gate

**Files:**
- Modify only files required by findings from the checks below.

**Interfaces:**
- Produces: one clean reviewed commit ready for deployment.

- [ ] **Step 1: Run focused suites without CUDA**

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/test_pccs_selector.py tests/test_pccs_masks.py \
  tests/test_davis_metrics.py tests/test_pccs_cycle.py \
  tests/test_pccs_job.py tests/test_anchor_expert.py \
  tests/test_eval_runtime_overlay.py
```

- [ ] **Step 2: Run cross-view and legacy matrix regression**

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/test_crossview_benchmark.py \
  tests/test_crossview_benchmark_contract.py \
  tests/test_eval_expert_matrix.py \
  tests/test_seg_metric_full.py
```

- [ ] **Step 3: Run static checks**

```bash
bash -n scripts/eval_expert_matrix.sh
bash -n scripts/eval_crossview_zero_shot.sh
python -m py_compile \
  v2sam_joint/pccs_selector.py v2sam_joint/pccs_masks.py \
  v2sam_joint/davis_metrics.py v2sam_joint/pccs_cycle.py \
  v2sam_joint/pccs_job.py projects/v2sam_anchor/models/v2sam.py
git diff --check
git status --short
```

Expected: tests and static checks pass; status contains only intentional files before commit.

- [ ] **Step 4: Commit final test-only adjustments**

```bash
git add v2sam_joint projects/v2sam_anchor third_party/davis2017 \
  scripts/eval_expert_matrix.sh scripts/eval_crossview_zero_shot.sh tests
git commit -m "test: verify strict multi-expert evaluation"
```

Do not create an empty commit if no adjustment was needed.

- [ ] **Step 5: Obtain independent CLEAN review**

Review scope is the base-to-head range and must include:

- GT not reachable from producer/selector APIs or inference manifest;
- strict A/B/C asset closure and frozen post-state;
- selector ranking/fallback determinism;
- selected-mask byte identity and tamper rejection;
- official DAVIS implementation parity;
- published receipt/manifest/aggregate closure;
- exact HEAD, clean status, bash/compile/diff checks.

Any Critical/Important finding returns to the relevant Task's RED/GREEN cycle.

---

### Task 9: Deploy and Run Formal Smoke Matrix

**Files:**
- No source edits after review; write only unique deployment, control, and formal artifact directories on `.166`.

**Interfaces:**
- Produces: four formal smoke receipts and independently verified masks/metrics.

- [ ] **Step 1: Deploy reviewed commit by prerequisite bundle**

Verify source/local/`.166` bundle SHA, create a new detached contained checkout,
and prove exact HEAD/parent/tree, clean status, `git fsck`, and `git diff --check`.

- [ ] **Step 2: Run no-GPU preflight**

```bash
CUDA_VISIBLE_DEVICES='' bash scripts/eval_crossview_zero_shot.sh \
  --print-plan --datasets DROID,BehaviorSim --experts A --stages smoke
CUDA_VISIBLE_DEVICES='' bash scripts/eval_crossview_zero_shot.sh \
  --print-plan --datasets DROID,BehaviorSim --experts ABC --stages smoke
```

Validate exact dataset revisions/counts, A/SAM2/DINO, B, C, runtime, selector,
DAVIS source, and absent output roots.

- [ ] **Step 3: Run fresh resource gate**

Require at least 15,000 MiB free GPU memory, no existing V2-SAM job, sufficient
RAM/disk, and unchanged protected Label Studio/EVA/Isaac processes.

- [ ] **Step 4: Run A-only smokes serially**

Run unique formal DROID A smoke then BehaviorSim A smoke. After each, require
terminal rc=0, final root present, staging=0, accounting=10/10/10 and zero
mismatch/missing/extra, strict load, immutable artifacts, and GPU recovery.

- [ ] **Step 5: Run ABC+PCCS smokes serially**

Run unique formal DROID ABC smoke then BehaviorSim ABC smoke. Validate A/B/C
candidate counts, PCCS decisions, fallback counts, selected/candidate identity,
IoU/Dice/J/F/J&F offline recomputation, aggregate bytes, and protected services.

- [ ] **Step 6: Stop for product failure or proceed after CLEAN audit**

No failed formal job is retried. A code/product failure returns to TDD and a
new review. Only four independently verified smoke receipts authorize full.

---

### Task 10: Run Full Matrix and Record Results

**Files:**
- Modify: `V2SAM_EXPERIMENT_MATRIX_LIVE.md`

**Interfaces:**
- Produces: DROID and BehaviorSim A/B/C/PCCS full comparison with immutable evidence references.

- [ ] **Step 1: Run DROID ABC+PCCS full**

Use the reviewed checkout and a unique run root. Require 2,162 evaluated
pairs/objects, zero accounting errors, atomic publication, official aggregate
validation, and independent mask/metric recomputation.

- [ ] **Step 2: Run BehaviorSim ABC+PCCS full**

Require 243 evaluated pairs/objects with the same formal gates.

- [ ] **Step 3: Independently compare all expert outputs**

For each dataset report A, B, C, and selected PCCS:

```text
stock IoU/Dice
per-object IoU/Dice
per-pair IoU/Dice
J_m / F_m / J&F_m
selected-expert counts and rates
fallback counts and rates
paired B->PCCS and C->PCCS deltas with 100,000 bootstrap replicates
```

- [ ] **Step 4: Update the live matrix with exact evidence**

Record commit, run roots, relation/checkpoint/code SHAs, run/receipt/raw/summary
SHAs, counts, metrics, claim labels, and any limitations. Preserve the
existing B/C historical rows and add the new mask-backed J/F comparison.

- [ ] **Step 5: Verify and commit documentation**

```bash
git diff --check -- V2SAM_EXPERIMENT_MATRIX_LIVE.md
git add V2SAM_EXPERIMENT_MATRIX_LIVE.md
git commit -m "docs: record strict PCCS multi-expert results"
```

Completion requires no live runner, zero staging directories, restored GPU
resources, protected services unchanged, clean checkouts, and all final hashes
recorded.
