**Completed 2026-09-15 23:45 CST.** Full frame IoU: PCCS 53.6119%, learned gate 57.2772% (+3.6653pp; 95% CI +2.6849 to +4.8846), fixed O-MaMa 56.3418%. Exact coverage; platform succeeded and occupied nodes 0. See [FINAL_REPORT.md](FINAL_REPORT.md).

# Frozen PCCS gain gate — full Exo2Ego confirmation

The preceding independent 512-pair/965-object/64-take holdout improved frame IoU from 70.9472% to 74.9235% (+3.9762 percentage points; paired take-bootstrap 95% interval +2.5743 to +5.4232). Fixed O-MaMa margin0.05 achieved 74.2185% (+3.2713 points). These subset scores are not full-benchmark scores.

This run freezes exactly the same HistGradientBoosting gate and predicted-gain threshold 0.03. The gate was fitted on 384 official-train pairs and calibrated on 128 other training pairs. V2-SAM and O-MaMa weights remain frozen. No new parameter selection is allowed.

Stages: a 32-pair 4090 smoke test; a 32-pair, four-H100 smoke test; full Exo2Ego evaluation of 46,515 pairs and 109,253 objects. Every stage generates its candidates once. All methods share that bank, including the actual original PCCS selection. Four scoring workers handle disjoint object rows. Every cached mask is checked against its recorded IoU, with exact identity coverage, zero skipped batches, finite predictions, strict checkpoint and feature hashes, and original O-MaMa forward equivalence.

The full benchmark contains earlier exploratory/confirmation frames. Report its aggregate performance as a frozen full benchmark, and the previous holdout separately as independent confirmation. Frame-weighted bootstrap resamples takes and preserves unequal take lengths.

All files are isolated here. The portable Python/OpenCV runtime, model assets, and private sklearn packages reference validated sibling experiment directories. Do not alter or restart older experiments. Do not stop the shared 4090 instance. The H100 job runs smoke4 then full, fails on incomplete coverage, and releases its allocated GPUs on exit. Monitor using the latest stage ETA and schedule the next check at 0.8×ETA, rounded to a minute with a one-minute minimum; use five minutes during initialization or stage changes.
