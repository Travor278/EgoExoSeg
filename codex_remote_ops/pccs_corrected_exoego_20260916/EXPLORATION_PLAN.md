# Further exploration, ordered by the current evidence

## 1. Probability-weighted regional pooling

The audited upstream RegionSampler samples `nonzero()` positions. Predicted coarse masks are sigmoid probabilities, so almost the whole grid can be nonzero. A weighted average, or foreground-weighted spatial sampling, could make the language prompt reflect the proposed object instead of broad background. This changes candidate generation, whereas current O-MaMa routing only selects existing masks. It may improve the candidate oracle ceiling.

First compare frozen soft-weighted pooling, hard threshold pooling, and current sampler on disjoint TRAIN fit/calibration takes, preserving checkpoint and point prompts. It is a distribution shift for the trained matcher; improvement is a hypothesis, not a guaranteed bug fix. If necessary finetune the small matcher/prompt projection while keeping SAM/DINO frozen. Do not tune the pooling threshold on Exo→Exo test outcomes.

## 2. Reliable multi-point cross-view prompting

Current sparse correspondence sends one top match per object to SAM. Test reciprocal matches, spatially separated top-k points and confidence filtering; compare both candidate quality and final selector performance. Keep source-mask prompting and geometric coordinate conversion fixed and verify point placement. Any new confidence or k thresholds must be chosen using training/calibration imagery, not the20-take Exo→Exo test.

## 3. Object and context reliability at multiple scales

O-MaMa's fixed100px full-box descriptor includes foreground. A target-dependent projected scale or a foreground/exterior split can separate object identity from surrounding scene appearance. More compatible integration: fit a small reliability head using O-MaMa object/context/cross-attention scores plus V2-SAM point agreement, rather than hand-selecting a new scale on test. Compare against the complete pretrained head and the present two-geometry consensus. Preserve negative results: earlier untrained ring-context variants were inconsistent.

## 4. Quality-aware matching adaptation

The published matcher optimizes identity correspondence, while segmentation quality depends on boundary accuracy. Fit a compact ranking or gain model with candidate IoU differences on corrected-weight TRAIN candidates; include distorted-view and scale augmentation. The existing gain gate used author-weight candidates, so refitting on corrected candidates is a justified controlled comparison. Keep separate calibration takes and freeze before independent evaluation. Cross-view instance training pairs are needed to claim true Exo→Exo adaptation; same-image augmentation alone is only a proxy.

## 5. Larger independent confirmation and attribution

Current Exo→Exo has only20 takes. Positive bootstrap intervals support that benchmark, not universal stability. Freeze one selected method and evaluate additional unseen external-camera pairs/takes when authorized data is available. For attribution, run controlled object-only/context-only/cross-attention and quality-feature ablations, ideally refitting comparable small heads rather than treating test-time zeroing as independently trained ablation.

Recommended next concrete step after this bidirectional rerun: run the pooling audit on a fixed training/calibration bank and assess whether candidate oracle quality improves. If not, focus on reciprocal multi-point prompts. Avoid large test-threshold sweeps or changing several modules together.
