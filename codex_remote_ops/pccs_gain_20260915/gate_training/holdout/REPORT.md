# Frozen gain-gate independent test

Independent512-pair965-object64-take Exo2Ego test cohort; all earlier exploratory/confirmation takes excluded; not full test

Frozen hist_gradient_boosting, predicted-gain threshold 0.03; fitted only on official train, calibrated on disjoint train takes.

| Method | Frame IoU (%) | Delta (pp) | 95% take bootstrap CI (pp) |
|---|---:|---:|---|
| baseline | 70.9472 | 0.0000 | - |
| learned_gate | 74.9235 | 3.9762 | [2.5743315645650853, 5.42324589992175] |
| omama_margin005 | 74.2185 | 3.2713 | [1.7367203303349725, 4.902276967397092] |

Primary comparison: learned gate vs PCCS. Fixed O-MaMa margin0.05 is secondary. No test-time parameter selection. Training-internal calibration gains must not be reported as test gains.