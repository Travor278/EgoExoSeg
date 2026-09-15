# Frozen PCCS gain gate — full

full Exo2Ego benchmark including prior exploratory/confirmation cohorts

| Method | Frame IoU (%) | Delta (pp) | 95% take bootstrap CI (pp) |
|---|---:|---:|---|
| baseline | 53.6119 | 0.0000 | - |
| learned_gate | 57.2772 | 3.6653 | [2.6848649125606374, 4.884601729247951] |
| omama_margin005 | 56.3418 | 2.7299 | [1.8716373836524278, 3.7992474576010684] |

No model or threshold changes. All methods use the same freshly frozen V2-SAM candidate bank. Bootstrap resamples takes while preserving frame weighting for unequal take lengths. The full benchmark is not an entirely blind cohort: it includes earlier exploratory/confirmation test frames. See prior independent holdout for the separate confirmation.