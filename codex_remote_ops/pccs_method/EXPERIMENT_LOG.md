# Experiment log

Times refer to Asia/Shanghai unless the original log explicitly uses UTC. Raw training logs are retained under each experiment's `runs` / `gate_training` directories. No numbers below are fitted using the evaluated target labels.

| Date | Experiment | Finding / action | Evidence |
|---|---|---|---|
| 2026-09-14–15 | Original context PCCS; three-direction full evaluation | Expanded-bbox-minus-mask context did not provide consistent gains across directions. Exo→Exo mean context38.4343→39.0209; Ego→Exo and Exo→Ego degraded. | `../context_pccs_20260914/ablations_20260915/FINAL_RESULTS.md` |
| 2026-09-15 | O-MaMa aligned exploration | Published learned matching head improved some Exo→Ego64-pair variants; this small-sample gain was not generalized to other directions. | `../omama_aligned_20260915/REPORT.md` |
| 2026-09-15 | 512-pair / 64-new-take confirmation | Fixed full O-MaMa margin0.05 +1.1123pp, but CI crossed zero. Context/cross-zeroed learned head −0.3042pp. | `../pccs_gain_20260915/REPORT.md` |
| 2026-09-15 | Gate fitting | 384 official-train pairs fit /128 other train pairs calibrate; selected HistGradientBoosting, predicted-gain threshold0.03. Calibration +3.97575pp is not a test result. | `../pccs_gain_20260915/gate_training/fit_results.json` and `gate_fit.log` |
| 2026-09-15 | Independent gate test | 512 pairs965 objects64 unseen-to-exploration/confirmation takes; +3.9762pp with positive take-bootstrap CI. Frozen checkpoint calibration replay error0. | `../pccs_gain_20260915/gate_training/holdout/REPORT.md` |
| 2026-09-15 17:45–23:45 | Four-H100 full Exo→Ego | 46,515 pairs109,253 objects; baseline53.6119%, learned57.2772%, fixed O-MaMa56.3418%; exact coverage and all ranks zero skipped. Nodes released. | `../pccs_gain_full_20260915/FINAL_REPORT.md`, `runs/full/*.log`, `runs/full/scored/*receipt.json` |
| 2026-09-16 | Scope correction | User clarified Exo→Exo is the primary final target. Exo→Ego weight availability is a transfer choice, not justification to omit Exo→Exo evaluation. | Method section1 in the full report |
| 2026-09-16 | Exo→Exo preparation /4090 attempt | 1,094 pairs20 takes validated. GPU busy guard stopped before generating candidates; existing processes were left running. | `../pccs_exoexo_transfer_20260916/pipeline.log` and `runs/smoke4090/preflight_gpu_processes.txt` |
| 2026-09-16 00:18–00:29 | Exo→Exo four-H100 evaluation | Job `job-34b1c8a3-18b2-409f-9a1b-db3644048c4c`;1094 pairs exact, zero skipped. Baseline38.4343%, learned39.8194%(+1.3851pp), fixed O-MaMa40.0936%(+1.6593pp). Both95% take-bootstrap intervals cross zero. No Exo→Exo label fitting. Task succeeded; occupied nodes0. | `../pccs_exoexo_transfer_20260916/runs/full/REPORT.md`, raw logs, `resource_release.json` |

Implementation checks: Python syntax compilation; four-way aggregation parity against965 real holdout objects; feature invariance under target/oracle metric corruption on128 candidates; exact baseline candidate parity; all foundation/checkpoint loads and full-rank receipts recorded. Scoring checks every cached mask against recorded IoU with tolerance1e-5; full maximum observed error3.29e-8. These checks establish arithmetic and coverage, not cross-dataset generalization.

2026-09-16 pipeline audit: updated frame/object arithmetic independently recomputed; original and corrected weights both pass actual initialization and GT-content counterfactual probes. Corrected e19/e20 assets match the user's Hub revision. Full evidence: `../pccs_pipeline_audit_20260916/AUDIT_REPORT.md`.

2026-09-16 01:19–01:30 corrected-weight Exo→Exo: baseline38.5257%, prespecified native-margin primary42.2665% (+3.7409pp,95%CI[1.5792,5.7901]); canonical-margin42.3595%, geometric-consensus42.5829%, old learned-gate transfer41.1004%. Secondary comparisons remain exploratory.1094 pairs exact,4ranks zero skipped, taskjob-0572f367-dcd4-48c9-9cf8-0996c502b5eb succeeded, occupied nodes0. Code and raw logs are in `../pccs_corrected_exoexo_20260916/`.

The source code's random region sampler can change candidates across reruns/sharding. All reported deltas use the same frozen bank within the comparison. Never subtract a new run's method score from an unrelated prior-run baseline.
