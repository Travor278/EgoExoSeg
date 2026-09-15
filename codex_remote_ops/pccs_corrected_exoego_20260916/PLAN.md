# Corrected Exo→Ego full rerun

Requested to complement corrected-weight Exo→Exo. The user's later instruction designates **`../pccs_gain_full_20260915/FINAL_REPORT.md` as the single master report**; this supersedes the earlier per-experiment report destination. **46,515 pairs,109,253 objects,295 takes**; corrected Visual e19/Fusion e20, no expert retraining.32-pair four-H100 smoke precedes full inference and four-way scoring.

Primary comparison is geometric consensus vs the same-run corrected PCCS baseline: canonical and native-aspect O-MaMa must both favor the same mask by raw cosine advantage>0.05. Secondary comparisons are canonical-margin, native-margin and the frozen old learned gate0.03. Thresholds are fixed before results. The old gate was trained using author-checkpoint predictions and remains a transfer control; do not claim it was refitted for corrected experts.

All methods use one freshly generated immutable candidate bank. Full object/pair coverage, zero skipped ranks, checkpoint/feature hashes, per-mask metric parity and finite values are required. Bootstrap resamples takes while preserving frame weighting for unequal take sizes. All methods report frame and object IoU/Dice/ContA/LocE. Prior test cohorts are included: this is a frozen full benchmark, not an entirely unseen cohort. Do not subtract previous author-weight scores to claim method gain.

Report status is pending until results and resource-release receipts are complete. Further research proposals are documented in `EXPLORATION_PLAN.md`; this job tests existing frozen rules only.
