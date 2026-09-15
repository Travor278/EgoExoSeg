# Frozen PCCS gain gate — full

Exo2Exo full benchmark using corrected Visual e19/Fusion e20; primary native-geometry margin0.05; other methods exploratory

| Method | Frame IoU (%) | Delta (pp) | 95% take bootstrap CI (pp) |
|---|---:|---:|---|
| baseline | 38.5257 | 0.0000 | - |
| learned_gate | 41.1004 | 2.5747 | [0.2836138861735996, 4.462856708249998] |
| omama_margin005 | 42.3595 | 3.8338 | [1.3202864590497192, 5.984226796387022] |
| native_margin005 | 42.2665 | 3.7409 | [1.579173028333684, 5.79007904615121] |
| geometry_consensus | 42.5829 | 4.0572 | [2.1297264191910115, 5.80953357091458] |

No model or threshold changes. All methods use the same freshly frozen V2-SAM candidate bank. Bootstrap resamples takes while preserving frame weighting for unequal take lengths. Exo2Exo outcomes were not used to refit the gate or choose its threshold. Earlier exploratory context/O-MaMa tests did inspect subsets of this benchmark; this is a frozen transfer evaluation, not a claim that the benchmark was never observed.