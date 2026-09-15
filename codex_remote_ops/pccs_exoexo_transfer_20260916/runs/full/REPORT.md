# Frozen PCCS gain gate — full

Exo2Exo full benchmark, using frozen Exo2Ego experts and gate; zero-shot transfer

| Method | Frame IoU (%) | Delta (pp) | 95% take bootstrap CI (pp) |
|---|---:|---:|---|
| baseline | 38.4343 | 0.0000 | - |
| learned_gate | 39.8194 | 1.3851 | [-0.028819579624421433, 2.743472248387379] |
| omama_margin005 | 40.0936 | 1.6593 | [-0.48368507270911903, 3.611411148987061] |

No model or threshold changes. All methods use the same freshly frozen V2-SAM candidate bank. Bootstrap resamples takes while preserving frame weighting for unequal take lengths. Exo2Exo outcomes were not used to refit the gate or choose its threshold. Earlier exploratory context/O-MaMa tests did inspect subsets of this benchmark; this is a frozen transfer evaluation, not a claim that the benchmark was never observed.