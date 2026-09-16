# PCCS candidate union and O-MaMa adaptation — 2026-09-16

Goal: improve final segmentation, prioritizing Exo→Exo, while preserving every original candidate and the original O-MaMa two-geometry consensus output as fallback.

## Ordered experiments

1. **Cached multipoint union.** Add reliable MNN multipoint masks to the original three experts; keep original masks and scores. Fixed cosine margin 0.05, both geometries must agree. Completed offline: screen final −0.3227 pp / oracle +0.0386 pp; calibration final −0.2310 pp / oracle +0.1038 pp. This is a negative final-result control, not an improvement.
2. **Small residual pooling.** Predicted coarse-region features become `(1−α) legacy + α probability_weighted`, α ∈ {0.05, 0.10, 0.25}. The original random sampler is still called once, preserving RNG progression. Original source-prompt pooling and anchor path are unchanged. Add masks to the union; never replace original candidates wholesale.
3. **Quality-gain ranker.** Fit ridge α=10/100 and a small boosted-tree regressor only on TRAIN screen labels. Inputs: 29 prediction-only matching, source-family and mask-geometry features. Targets: candidate IoU minus original consensus IoU. Evaluate replacement thresholds 0.01/0.03/0.05 on calibration.
4. **O-MaMa metric adapter.** Freeze DINOv2, published O-MaMa cross-attention/context head and V2-SAM. Train a shared rank-8 residual 768→8→768 projection on frozen O-MaMa embeddings; zero-initialized output gives identity. Screen-only soft quality targets, two geometries, AdamW, 40 epochs; examine epochs 5/15/40 and margins 0.02/0.05/0.10. This is NOT V2-SAM prompt-projection finetuning, and stages 3/4 are NOT training-free.

## Data and selection

Reuse prior official-TRAIN screen (128 pairs / 243 objects / 16 takes) and calibration (128 / 225 / 16 disjoint takes). Experts have seen official TRAIN; these splits and Exo→Exo benchmark have also been used in earlier explorations. Neither constitutes a new blind validation set. Calibration selects configurations but does not optimize parameters. Only a method with positive final frame-IoU changes on BOTH screen and calibration qualifies; choose maximum calibration IoU, ties favor earlier/simpler methods. Freeze the checkpoint/hash and rule before any new Exo→Exo confirmation. If none qualifies, stop before target inference.

Primary target comparison is selected union versus newly rerun original O-MaMa geometry consensus, same pair seeds and candidate bank. Also report original PCCS. Prior historical scores used another RNG protocol and must not be substituted for the paired current baseline. Exo→Exo: 1094 pairs, 20 takes, corrected Exo→Ego Visual e19 / Fusion e20 weights. Frame metric averages objects inside each pair then averages pairs equally, including zero-IoU objects. Report take-cluster paired bootstrap 95% CI (10000 draws); never promise universal positive gains.

## Validation and execution

Run local syntax/routing checks, remote analytic residual/RNG and zero-adapter tests, 8-pair smoke, then screening/calibration. Require exact identity coverage, input-record hashes, unchanged anchor masks, union oracle ≥ original oracle, finite optimization gradients and loss. Reconstructed O-MaMa cosine must match unmodified arithmetic within 1e-6 and cached scores within 1e-4. Stop on failure; retain logs. Routing functions accept no target ground truth.

Use one task, at most four H100 GPUs. Residual arms occupy three GPUs; embedding extraction and tiny adaptation use one. Conditional target candidate generation uses four plus one waves. No shared notebook is restarted. Release the job after completion/failure with one-minute retention. Dynamic supervision: next check = round(0.8 × remaining stage ETA / 60) minutes, minimum one; initialization/stage transition/unknown ETA uses five minutes, no arbitrary upper cap.

All files live in this experiment directory. Summarize results and negative controls in `../pccs_gain_full_20260915/FINAL_REPORT.md`; publish source, metrics and logs to branch `pccs-omama-gain-20260916`, excluding images, masks, embeddings, private credentials and large weights.
