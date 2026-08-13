# V2-SAM Strict GT-Free Multi-Experts and Boundary Metrics Design

Date: 2026-08-14

Status: Approved direction; written design awaiting final user review

## Decision Summary

Implement a paper-faithful, independently auditable reconstruction of
V2-SAM Multi-Experts on the existing strict cross-view evaluation base.
The evaluated expert set is A (Anchor), B (Visual), and C (Fusion). The
final Multi-Experts prediction is one of the three candidate masks selected
by a strictly ground-truth-free Post-hoc Cyclic Consistency Selector (PCCS).

The first formal matrix is:

- DROID: A, B, C, and A+B+C+PCCS;
- BehaviorSim: A, B, C, and A+B+C+PCCS;
- metrics: stock IoU/Dice, per-object IoU/Dice, per-pair IoU/Dice,
  DAVIS-style region similarity J, contour accuracy F, and J&F;
- smoke before full, with no automatic retry after a formal product failure.

The result must be labeled `paper_faithful_reconstruction`, not an exact run
of unreleased official Multi-Experts code. J/F results on DROID and
BehaviorSim must be labeled `davis_style_auxiliary`, not DAVIS benchmark
results.

## Current Evidence and Constraints

The public V2-SAM repository currently exposes only
`projects/v2sam_visual` and `projects/v2sam_fusion`. Its README says that
V2-Anchor requires no training and uses the official SAM2 decoder, but it
does not expose a TripleDecoder/PCCS execution entry point or an A+B+C
checkpoint. The public repository has only its default branch.

The reviewed strict cross-view base is the clean worktree at commit
`8649bc30e91c0feab00fa204637178e82605dbba`. Its plan already names A, AB,
ABC, PCCS, cycle-mask, and a GT oracle, but `preflight_triple()` deliberately
blocks all of them. Its B and C inference, strict-load, artifact, aggregate,
and post-publication contracts are already validated in formal DROID and
BehaviorSim runs.

The internal TripleDecoder candidate at commit
`8b65a9c2c6c3ceaedee4967d5da8b24981be840c` is an algorithm reference only.
It cannot be used as the formal runtime because:

- its model pops target `masks` during prediction and passes them into expert
  calls;
- its metric reads target masks before selection;
- an empty-query fallback selects the expert with the highest target-GT IoU;
- its B adapter has checkpoint shape/key incompatibilities;
- its B and C loaders use `strict=False` without a closed alias/schema proof;
- it does not publish a canonical selected-mask closure suitable for an
  independent offline metric.

The design therefore composes the already reviewed single-expert paths
instead of reviving the monolithic TripleDecoder runtime.

## Goals

1. Produce strictly loaded A, B, and C candidate masks for every selected
   object.
2. Select one candidate with PCCS without reading target GT, target-GT paths,
   or any metric derived from target GT.
3. Publish candidate masks, cycle evidence, selector decisions, and final
   selected masks atomically with exact hashes and byte sizes.
4. Compute all metrics only after the selected mask has been frozen.
5. Preserve the existing DROID/BehaviorSim dataset, checkpoint, relation,
   provenance, and aggregate contracts.
6. Provide enough evidence to rerun metric computation offline without
   rerunning model inference.
7. Make candidate/selector output invariant to target GT being real, random,
   zero, or absent.

## Non-Goals

- Training or fine-tuning A, B, C, or PCCS.
- Claiming that the reconstruction is byte-identical to unreleased official
  Multi-Experts code.
- Reporting DROID/BehaviorSim J&F as a DAVIS leaderboard score.
- Using a GT oracle for model selection or formal ranking.
- Optimizing throughput by sharing mutable model objects or simultaneously
  loading all experts in the first implementation.
- Changing the already recorded B/C formal IoU/Dice results.

## Terminology

- **A / Anchor**: DINOv3 cross-view anchor points plus the original official
  SAM2 mask decoder. A has no independently trained expert checkpoint.
- **B / Visual**: the released Visual expert checkpoint and its reviewed
  Visual model class.
- **C / Fusion**: the released Fusion expert checkpoint plus the exact
  external DINOv3 closure.
- **ABC candidates**: three independent binary candidate masks for the same
  dataset record and object.
- **PCCS**: the deterministic, parameter-free selector that ranks ABC
  candidates by target-to-query cyclic consistency.
- **Multi-Experts result**: the final selected mask from ABC+PCCS, not the
  unordered set of three candidates.
- **GT oracle**: a diagnostic-only selector that reads target GT. It remains
  excluded from formal ranking and is outside the first implementation.

## Chosen Architecture

### Composition Instead of a Monolithic Triple Model

The formal job runs candidate producers sequentially at batch size 1:

```text
inference manifest (no target GT)
  -> A producer -> candidate A
  -> B producer -> candidate B
  -> C producer -> candidate C
  -> reverse-cycle evidence for A/B/C
  -> GT-free PCCS -> immutable selected mask
  -> offline metric process receives target GT
  -> raw metrics, summaries, receipts, aggregate
```

B and C reuse the reviewed model/config/load paths. Their only new output is a
canonical binary candidate mask plus selector evidence inputs. A is a narrow
new producer built from the already bound official SAM2 and DINOv3 assets.

PCCS runs after the three candidate masks exist. It uses the same pinned DINO
asset to project evidence from each predicted target mask back into the query
view. This avoids the incompatible internal TripleDecoder model classes and
keeps the amount of model code changed small.

Sequential execution deliberately trades runtime for lower peak memory and a
smaller proof surface. It also preserves the protected services on `.166`.

### Plan and Job Contract

The plan adds a formal expert selection `ABC` with selector
`pccs_cycle_points_strict`. It freezes:

- dataset ID, direction, scope, relation revision, and record count;
- code commit and exact runtime;
- A/B/C role mapping;
- every checkpoint/config path, SHA-256, byte size, and expected schema;
- selector version and deterministic expert tie order `A,B,C`;
- candidate and selected-mask encoding;
- DAVIS metric source revision and source-file SHA;
- expected artifact set and aggregate schema.

The job must fail before GPU use if any selected asset or plan field is
missing, aliased unexpectedly, outside the trusted root, noncanonical,
nonregular, or hash-inexact.

## Asset and Load Closure

There is no ABC checkpoint. The formal closure binds independent assets:

- A: official SAM2 checkpoint/config and official DINOv3 checkpoint;
- B: official Visual checkpoint and official SAM2 checkpoint/config; DINO is
  not a B model dependency;
- C: official Fusion checkpoint, official SAM2 checkpoint/config, and exact
  external DINOv3 checkpoint;
- PCCS: no learned checkpoint.

Known released schemas are retained as explicit preflight expectations:

- original SAM2 model state: 900 tensors;
- B checkpoint/model: 964 tensors, strict missing/unexpected/shape = 0;
- C checkpoint: 1,335 tensors plus the exact 368-tensor external DINO closure;
- DINO and all frozen shared inputs must have identical pre/post hashes.

The implementation must use the reviewed B and C classes rather than load
their checkpoints into the incompatible internal TripleDecoder classes. No
`strict=False` result is accepted without an explicit, exact alias expansion
whose final merged model load is strict and has zero missing, unexpected, or
shape-mismatched keys.

## Ground-Truth Isolation

The staged dataset contract is split into two documents:

1. `inference_manifest`: query image, query/reference mask, target image,
   object IDs, image dimensions, and non-GT metadata. It contains no target
   mask or target-mask path.
2. `metric_manifest`: the frozen selected-mask identity and target GT. Only
   the offline metric process receives it.

Candidate producers and PCCS receive only the inference manifest. Target
output size is derived from the decoded target image, never from a target
mask. PCCS must finish and the selected-mask file must be atomically published
before the metric process starts.

Tests replace target GT with random masks, all-zero masks, and no mask at all.
Candidate bytes, cycle evidence, selected expert, reason, and selected-mask
bytes must remain identical. Any difference is a product failure.

## PCCS Contract

For each record/object:

1. Validate that A, B, and C candidate masks are canonical binary masks with
   the exact target image dimensions.
2. Extract a deterministic set of representative points from the query mask
   using a fixed center-and-contour policy.
3. For each candidate mask, run the pinned reverse DINO correspondence from
   target to query and record returned points.
4. Rank valid candidates by:
   - descending number of reverse points inside the query mask;
   - ascending mean L2 cycle distance to the deterministic query-mask points;
   - fixed expert order `A,B,C` as the final exact tie break.
5. If exactly one expert has valid cycle evidence, select it.
6. If none has valid cycle evidence, use the declared GT-free fallback order
   `C,B,A`, selecting the first candidate whose mask is valid. Record
   `fallback_no_cycle` rather than silently treating it as a PCCS win.
7. If the query mask is empty, use the same declared fallback and record
   `fallback_empty_query`.

NaN/Inf coordinates, missing candidates, dimension disagreements, duplicate
object IDs, nonbinary masks, or an inability to choose from the declared
fallback order fail the job. Candidate input order must not affect the result.

Each decision records the selected expert, reason, candidate hashes, in-mask
counts, cycle distances, valid flags, query evidence hash, and selector
version. It contains no target metric.

## Prediction Artifacts

COCO RLE JSON is the canonical lossless mask representation. PNG is not a
second source of truth. The published job tree contains:

```text
candidates/<record>/<object>.A.rle.json
candidates/<record>/<object>.B.rle.json
candidates/<record>/<object>.C.rle.json
cycle/<record>/<object>.json
selected/<record>/<object>.rle.json
selection.jsonl
raw_metrics.json
metric_summary.json
strict_load.json
job_manifest.json
receipt.json
```

Every manifest entry binds relative path, SHA-256, byte size, role, record ID,
and object ID. The selected mask must be byte-identical to the selected
candidate after canonical serialization. The publisher rejects missing,
extra, duplicate, symlink, special, outside-root, staging-string, or
self-consistently tampered artifacts.

## Metrics

The offline metric process recomputes, for A, B, C, and PCCS-selected masks:

- stock IoU and Dice;
- per-object IoU and Dice;
- per-pair IoU and Dice;
- region similarity `J_m`;
- contour accuracy `F_m`;
- `J&F_m = (J_m + F_m) / 2`.

J/F uses a vendored, revision-pinned DAVIS evaluation implementation rather
than a new boundary formula. Its source commit and file hashes are part of the
job spec and receipt. Golden tests must match the pinned implementation for
identical, empty, disjoint, one-pixel, shifted-boundary, and resized masks.

For DROID and BehaviorSim, each pair currently has exactly one object. The
contract therefore expects stock, per-object, and per-pair IoU/Dice to be
equal. `J_m` should equal region IoU under the same object averaging, while
`F_m` measures contour agreement and is not Dice. J/F/J&F results are reported
as DAVIS-style auxiliary metrics only.

Metric output must be reproducible from published RLE and GT artifacts without
model execution. Online and independent offline summaries must be byte-exact
canonical JSON.

## Error Handling and Publication

The existing hidden staging and atomic final-directory publication rules are
retained. A formal failure produces no final run root and is not automatically
retried. Diagnostic logs may remain in a separate evidence directory, but
their partial metrics are not formal results.

Failures include:

- any asset, code, relation, or runtime identity mismatch;
- non-strict A/B/C state closure;
- target GT visible to inference or selector contracts;
- candidate/evidence/selection nonfinite or incomplete;
- selected mask not byte-identical to the selected candidate;
- metric disagreement or artifact closure mismatch;
- post-run checkpoint/DINO mutation;
- protected-service or resource-gate violation.

## Verification Strategy

### Unit and Contract Tests

- exact A/B/C asset roles and strict schemas;
- selector ranking, ties, candidate permutation, invalid points, empty query,
  no-cycle fallback, and NaN/Inf rejection;
- real/random/zero/absent target-GT invariance;
- canonical RLE round-trip and binary/dimension validation;
- J/F golden masks against the pinned DAVIS implementation;
- stock/per-object/per-pair equality for one-object pairs and intentional
  inequality for multi-object fixtures;
- missing/extra/changed/aliased/symlink/special artifact rejection;
- selected-mask/candidate exact identity and offline metric parity.

### CPU Preflight

- bash syntax, Python compile/import, exact plan, and validate-only;
- real A/SAM2/DINO schema load;
- real B 964-key strict load with no DINO role;
- real C 1,335-key plus external-DINO strict closure;
- selector execution on small synthetic canonical masks without GT;
- source and checkpoint pre/post SHA equality.

### GPU Execution Order

1. DROID A-only smoke, 10 pairs.
2. BehaviorSim A-only smoke, 10 pairs.
3. DROID ABC+PCCS smoke, 10 pairs.
4. BehaviorSim ABC+PCCS smoke, 10 pairs.
5. Independent review of all smoke receipts and masks.
6. DROID ABC+PCCS full, 2,162 pairs.
7. BehaviorSim ABC+PCCS full, 243 pairs.

The ABC jobs also publish and score A/B/C candidates, so no additional B/C
rerun is needed solely to obtain J/F metrics. Existing B/C formal IoU/Dice
remain the historical baseline; the new candidate scores must agree within
the frozen deterministic inference contract before comparisons are accepted.

All GPU jobs run serially at batch size 1 with fresh resource gates and the
existing protected processes untouched.

## Claim and Reporting Rules

Formal tables must distinguish:

- `official_paper_reported`: numbers copied from the paper;
- `official_checkpoint_single_expert_measured`: reviewed B/C runs;
- `paper_faithful_reconstruction`: A and ABC+PCCS from this design;
- `davis_style_auxiliary`: J/F/J&F on DROID or BehaviorSim;
- `oracle_diagnostic`: any future GT oracle, excluded from ranking.

The report may state whether reconstructed PCCS improves over A/B/C on the
same frozen dataset and metric. It must not state that DROID/BehaviorSim
results reproduce the paper's Ego-Exo4D, HANDAL-X, or DAVIS scores.

## Alternatives Considered

### Wait for Official Multi-Experts Code

This gives the strongest code provenance but has no known release date and
blocks current evaluation. It remains useful as a future cross-check.

### Revive the Internal Monolithic TripleDecoder

This appears shorter initially, but it carries GT coupling, incompatible B
weights, permissive loads, and higher simultaneous memory ownership. Fixing
those issues would change more model code and enlarge the review surface.

### Selected: Compose Strict Single-Expert Producers

This reuses validated B/C execution, adds a narrow A producer, makes PCCS a
pure postprocessor, and isolates metrics after immutable mask publication. It
requires more orchestration but less risky model modification and produces the
strongest audit trail.

## Completion Criteria

The work is complete only when:

1. all focused and existing cross-view CPU tests pass;
2. an independent review returns CLEAN for GT isolation, checkpoint closure,
   selector determinism, mask publication, and J/F parity;
3. all four smoke jobs complete with formal receipts;
4. both full ABC+PCCS jobs publish atomically with accounting zero;
5. independent offline mask/metric recomputation exactly matches each formal
   summary;
6. final documentation records run roots, commits, input/artifact hashes,
   metrics, fallback counts, and claim labels.
