# Frozen PCCS gain gate — full

Exo2Ego full benchmark using corrected Visual e19/Fusion e20; primary geometry-consensus gate; other methods exploratory

| Method | Frame IoU (%) | Delta (pp) | 95% take bootstrap CI (pp) |
|---|---:|---:|---|
| baseline | 53.8997 | 0.0000 | - |
| learned_gate | 57.1015 | 3.2018 | [2.243963333121287, 4.338743778864178] |
| omama_margin005 | 56.6352 | 2.7355 | [1.9524975082199914, 3.667664504185982] |
| native_margin005 | 56.6329 | 2.7332 | [1.9494767670863766, 3.6668535256106707] |
| geometry_consensus | 56.6385 | 2.7388 | [1.9555938636055405, 3.6699759293045227] |

No model or threshold changes. All methods use the same freshly frozen V2-SAM candidate bank. Bootstrap resamples takes while preserving frame weighting for unequal take lengths. No fitting or threshold selection on these test outcomes. This full Exo2Ego benchmark was evaluated previously using author weights; report this as a frozen corrected-weight comparison, not an entirely unseen cohort.