# O-MaMa gain confirmation

512-pair,64-take confirmation; excludes every exploratory probe take; not full-test result

| Method | Frame IoU (%) | Delta (pp) | 95% take bootstrap CI (pp) |
|---|---:|---:|---|
| baseline | 71.8812 | 0.0000 | - |
| learned_object_only_top1 | 71.5770 | -0.3042 | [-1.8695437059483129, 1.2331055476054893] |
| omama_margin005 | 72.9935 | 1.1123 | [-0.882391887690033, 2.7574021252771486] |
| omama_top1 | 72.3434 | 0.4622 | [-1.5306439807209762, 2.132863885626924] |

Two primary rules were fixed before this sample was scored. Check multiplicity-adjusted intervals in JSON. Learned object-only means test-time context/cross-attention zeroing of the published head, not a newly trained object-only model. Official validation imagery was unavailable; this is an independent confirmation cohort within the official test split, not validation tuning.