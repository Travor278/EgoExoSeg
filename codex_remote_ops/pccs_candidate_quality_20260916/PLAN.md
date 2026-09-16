# Candidate-quality 2×2 ablation

Four fixed arms: baseline, predicted-coarse-mask probability-weighted pooling, reliable multipoint correspondence, and both. Corrected Visuale19/Fusione20 and all DINO/SAM/O-MaMa weights are frozen. No expert training in this first round.

Screen128 pairs from16 official TRAIN takes; calibration128 pairs from16 different TRAIN takes, reusing the already extracted training imagery. Eight screen pairs form the runtime smoke. Experts previously saw official TRAIN; these scores are not independent test results. No thresholds are chosen from Exo2Exo outcomes.

Pool: source-mask pooling unchanged. Only the predicted coarse-mask pooling in visual/fusion changes to area-resized probability weights over the dense SAM grid. The discarded original sampling call remains to preserve RNG advancement in other branches; this ablation makes no speedup claim. No per-mask contrast enhancement or gamma tuning.

Points: up to3 strict global mutual nearest-neighbor matches inside the source foreground; require top1−top2 target similarity margin≥0.01 and separation≥2 patches in both views. Multipoint is used only if at least2 pass; otherwise use the exact legacy single-point result. Target GT is never supplied. Original point-to-SAM coordinate conversion and backward PCCS routing remain unchanged.

All arms reset the same hash-derived RNG per pair. Runtime checks require pool-only to preserve anchor masks and points-only to preserve visual masks; both-arm branch invariants are also checked. Each arm caches its own generated candidates. The fixed canonical/native O-MaMa consensus chooses among that arm's candidates; compare consensus-vs-consensus across arms, not changed-arm method vs an unrelated old baseline.

Selection rule fixed before execution: positive candidate-oracle gain on both screen and calibration, plus positive final consensus gain on calibration; among qualifying arms maximize calibration consensus IoU, ties favor earlier/simpler arm. If no arm qualifies, report the negative result and do not launch target test. If qualified, freeze the selected arm and compare it with a fresh identically seeded baseline on the1094-pair Exo2Exo benchmark. Report candidate oracle, each expert, PCCS, consensus, frame/object metrics, paired take-bootstrap intervals and point acceptance/fallback counts. The target benchmark was previously observed; it is not a new wholly blind dataset.

All outcomes, including negative ones, go into the single master report `../pccs_gain_full_20260915/FINAL_REPORT.md` and the existing `pccs-omama-gain-20260916` GitHub branch. Never overwrite earlier experiment banks. Stop on failed coverage or isolation checks. Supervise at0.8×current ETA, rounded to a minute,min1; initialization/stage transitions use5min.
