# Source-only follow-up: calibrate native cycle confidence instead of requiring raw-cycle dominance

Declared after collecting TRAIN/calibration features, before reading the current Exo2Exo outcome. The first batch and its frozen primary remain unchanged. No new image/model inference is required: all experiments below use the same native-DINOv3 evidence and original candidate masks.

Reason: TRAIN has228 objects with a better quality-valid alternative, but the raw-cycle gate admits35; calibration has82, admitted7. These counts use TRAIN/calibration labels only for oracle diagnosis. The hard gate can prevent context from correcting a confident but wrong cycle.

Keep the original mask-quality/source-valid/nonduplicate requirements. Compare three fixed admission rules:

1. `dominance`: original R1 rule, soft_cycle > max(.01, baseline+.005), exact reproduction control.
2. `present`: require nontrivial soft-cycle support >.001; whether the correspondence is more trustworthy is decided by the calibrated native PCCS confidence, with the original PCCS output as fallback.
3. `context_override`: allow original dominance, OR require soft_cycle>.001 and ≥half baseline support, object cosine no worse than baseline−.03, and positive foreground/background contrast differences ≥.03 at BOTH1.5 and2 scales. Empty original candidates may be rescued by a valid alternative with nontrivial support. This uses positive relative support rather than low-context veto.

Use the SAME fit/calibration takes and confidence inputs/model families/thresholds as R1 (four input groups ×three calibrators ×two thresholds), plus all three admission rules. Each model fits the same quality-valid alternatives, so it can be shared across rules; recompute take-held-out OOF scores, do not evaluate alternatives merely on in-sample predictions. R1 dominance scores must reproduce within1e-6 pp before selecting follow-up configurations. All training and policy logic excludes O-MaMa scores.

Select positive OOF and calibration, maximum calibration. Freeze one primary and same-capacity feature controls BEFORE reading new target records. Infer through the same native confidence policy on cached features; report this as cached decision replay, not a fresh segmentation-network pass. Verify strength0 and label-counterfactual invariance. Retain original/PCCS and R1 results, every negative configuration, model SHA, dependency versions and native mask identities. This is additional source-side exploration; it does not create a new blind benchmark.
