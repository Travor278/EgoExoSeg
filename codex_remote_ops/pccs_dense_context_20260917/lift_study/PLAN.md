# Area-prior correction for native dense PCCS cycles

Source-only follow-up, declared before reading completed current target results. Motivation follows the TRAIN gate diagnosis and probability algebra, not target tuning: uninformative row-uniform correspondences give forward_mass=q_area, backward_mass=s_area, and cycle_mass=q_area*s_area. Raw mass therefore contains a candidate-size prior and a fixed .01 floor can exclude small objects.

Add log(forward_mass/q_area), log(backward_mass/s_area), log(cycle_mass/(q_area*s_area)), and log source/target occupancy to PCCS confidence inputs. Clamp log features to[-12,12], reject empty masks. Under a uniform kernel the three lift statistics are exactly0 for arbitrary positive mask areas. These are native correspondence statistics, not imported O-MaMa features or weights.

Three input groups: original cycle control; cycle+lift; cycle+lift+context. Three native admission rules: original dominance reproduction control; positive forward/backward lift; relative cycle lift exceeding the baseline by0.1 while forward/backward lift are positive. Keep source validity, original candidate-quality gate, nonduplicate candidate checks and original PCCS fallback. All configurations use the same three calibrators, .03/.05 thresholds, exact persisted R1 take folds, fit384 pairs and calibration128 pairs. Every fit is source-only; preserve no-context controls and all negative outcomes.

Freeze the best configuration with positive OOF and calibration before target replay. Same candidate bank and repaired external reference as R1/R2; do not change already-frozen R1/R2 choices. This is an exploratory extension of PCCS confidence; novel or effective is not assumed from the formulation alone.
