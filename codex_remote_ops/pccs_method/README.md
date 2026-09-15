# PCCS + O-MaMa + learned replacement gate

This branch records the evaluated implementation and experiment evidence. The primary downstream objective is **Exo→Exo improvement**. Exo→Ego is the source of the currently available expert/matcher weights and gate training data; that does not make Exo→Exo evaluation an Exo→Ego task.

## Results and status

| Experiment | Original PCCS frame IoU | Learned gate | Difference | Status |
|---|---:|---:|---:|---|
| Exo→Ego independent 512-pair holdout | 70.9472% | 74.9235% | +3.9762pp | Complete; 95% take-bootstrap CI [+2.5743,+5.4232] |
| Exo→Ego full 46,515 pairs / 109,253 objects | 53.6119% | 57.2772% | +3.6653pp | Complete; CI [+2.6849,+4.8846] |
| Exo→Exo 1,094 pairs / 20 takes, using Exo→Ego weights | 38.4343% | 39.8194% | +1.3851pp | Complete; CI [−0.0288,+2.7435], crosses zero |

The full Exo→Ego benchmark includes earlier exploratory/confirmation test samples. It is a frozen full-benchmark evaluation, not an entirely unseen cohort. Exo→Exo was evaluated separately: fixed O-MaMa cosine-margin0.05 achieves40.0936% (+1.6593pp; CI [−0.4837,+3.6114]). Both transfer estimates are positive but intervals cross zero; the learned gate does not outperform the fixed gate on this benchmark. No Ego→Exo result exists for the newly learned gate yet.

Detailed Chinese method description, equations, feature table, data boundaries and all metrics: [FINAL_REPORT.md](../pccs_gain_full_20260915/FINAL_REPORT.md).

## Code layout

- `../pccs_gain_full_20260915/`: exact evaluated inference adapter, 65-feature extractor, candidate reader, scorer, sharded runner and aggregation; full run logs and receipts.
- `../pccs_gain_20260915/`: gate training/calibration code, input records sufficient to refit the small gate, frozen holdout scorer, configuration and reports.
- `../pccs_exoexo_transfer_20260916/`: direct-transfer Exo→Exo experiment; task direction and weight direction explicitly separated.
- `v2sam_pccs_gain.patch`: one patch against V2-SAM upstream revision `0e3bc33dec3e202ffbb86cec01038e60b18c162a`. Includes context diagnostics, rank completion receipts and immutable candidate-bank hooks. Formal learned-gate experiments set `PCCS_CONTEXT=0`.
- `EXPERIMENT_LOG.md`: dated experiment history, including negative findings and the occupied-4090 preflight.
- `EXPORT_MANIFEST.json`: source artifact hashes. Historical absolute Qizhi paths remain in raw logs/configurations for auditability; browser storage and credentials are excluded.

## Reproduction boundaries

These are actual **site-specific research scripts**, not an installer that provisions a clean machine. They assume the validated V2-SAM/SAM2 environment, expert weights and authorized Ego-Exo4D imagery. Historical Qizhi paths must be configured or recreated before using `prepare.py`/shell launchers. Large image/mask banks and foundation-model weights are deliberately not committed. `training_records.jsonl` contains small allowlisted prediction features and training supervision only, not raw images or masks.

To obtain the evaluated candidate-generation code, clone `https://github.com/jaychempan/V2-SAM-O`, checkout the pinned revision above, then run `git apply --check <absolute-path>/v2sam_pccs_gain.patch` followed by `git apply`. Do not apply the patch to an arbitrary current branch. The patch has been checked against a clean checkout of its exact base revision.

To refit only the gate without GPU inference, use Python with numpy/scipy, scikit-learn **1.5.2**, joblib **1.4.2** and threadpoolctl **3.5.0**, then run:

```bash
python codex_remote_ops/pccs_gain_20260915/fit_gain_gate.py
```

It uses the committed 901-object training records, 384 fit pairs and 128 disjoint calibration pairs; chooses only among the preregistered models/thresholds; and writes `selected_gate.joblib` and `fit_results.json`. Use a fresh copy when preserving original receipts. It never reads the test reports for fitting. The original selected gate SHA256 is `f79543baa14e81ebc83fa3bb7458e0940e405afe31a556fb612281c6aa0d2678`; serialization on another platform need not be byte-identical, so also verify predictions and feature-column order.

For matching, `aligned_model.py` expects `reference/O-MaMa`, `reference/dinov2` and `weights/` alongside it. Pin O-MaMa to `0f187c65cb9d8f8df1d8b5f445edbb0485956d88` and DINOv2 to `7764ea0f912e53c92e82eb78a2a1631e92725fc8`; obtain the public Exo→Ego matching checkpoint from the original O-MaMa project and DINOv2 ViT-B/14-reg4 weights from the official release. Expected hashes are listed in `ASSETS.json`. Inference must retain canonical geometry and gate threshold0.03 to reproduce this configuration.

The O-MaMa adapter reuses/adapts AGPL-3.0 model arithmetic. The original license is preserved in `third_party/O-MaMa/LICENSE`; see `THIRD_PARTY.md`. This is not an end-to-end reproduction of O-MaMa's original candidate-generation/training pipeline.
