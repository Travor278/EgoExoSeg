# PCCS gain exploration — 2026-09-15

Goal: improve segmentation accuracy. All new files and outputs stay here; prior production runs remain frozen.

The first new experiment confirms two preselected O-MaMa rules on 512 Exo→Ego image pairs (871 objects) from 64 takes. All takes appearing in the previous exploratory 64-pair O-MaMa bank are excluded. Pair and take selection use fixed hashes, without looking at target outcomes. See `confirmation_plan.json` for the exact protocol and annotation checksum.

Primary rules: published full O-MaMa score with cosine replacement margin 0.05; and top-1 from the published head with context and cross-attention inputs zeroed. Full-score top-1 is secondary. The latter zeroing variant is an inference intervention, not a separately trained object-only model. Nothing is fitted on this cohort.

V2-SAM candidates are generated once under the upstream seed 0 and packed to immutable, rank-specific NPZ files. Every comparison uses the same masks and the same original PCCS baseline. O-MaMa inference uses seed 42. Scoring checks cached masks against annotation IoU, exact object coverage, finite scores, and equivalence to original O-MaMa forward for the complete head.

The result reports frame/object IoU, Dice, boundary and location metrics. Paired bootstrap resamples 64 takes, with 10,000 replicates. JSON also contains 97.5% intervals for the two prespecified primary comparisons.

Official train/validation/test annotations contain 754/200/295 disjoint takes. Official validation images are currently missing from the previously extracted locations; empty extraction placeholders were found. Therefore this run is an independent confirmation cohort within the official test split, **not validation tuning or a full-test result**. Training a new selector is deferred until suitable imagery is available.

Remote root: `/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/pccs_gain_20260915`.

`launch_confirmation.sh` starts the guarded pipeline on the existing 4090. `status.json` reports candidate generation; during O-MaMa scoring use `scoring_status.json`. `confirmation_results.json` and `REPORT.md` are created only after exact coverage and metric integrity checks pass. Do not rerun into existing outputs. `transfer.py` moves only experiment files through Jupyter; `run_command.py` targets the isolated browser tab marked `codex_context=pccs_gain_20260915`.

Supervision follows the user's rule: next check in 0.8 × current remaining ETA, rounded to a minute with a one-minute minimum; use five minutes during initialization/stage changes. Never stop the shared 4090 instance or unrelated tasks.
