# Corrected-weight Exo→Exo: geometry adaptation and conservative replacement

Start only after the pipeline audit and actual metric regression pass. Use the user's corrected Visual e19 and Fusion e20, SHA verified against `Travor278/V2-SAM@8b8310e9ee13023cb8711c2417e18cc7361dc3aa`. Keep the historical author-weight experiment unchanged.

One new frozen bank on1094 pairs20takes. All comparisons use this corrected-weight baseline. Primary: original O-MaMa head with **native aspect-ratio geometry**, position-table interpolation, and the already specified raw cosine margin0.05. This tests whether forcing an external target into the700-square ego grid is an unnecessary transfer mismatch.

Secondary: canonical O-MaMa margin0.05; geometric-consensus gate that replaces only if canonical and native rules agree; previously trained gate0.03 as an explicitly labeled source-distribution transfer control. No threshold search on Exo→Exo outcomes. Same checkpoint, same masks, native/canonical only change the matcher geometry. All five methods are scored in one pass over one bank.

Report corrected baseline, each method's absolute IoU/Dice/ContA/LocE, paired take-bootstrap95CI. Only native_margin005 is primary; other intervals are descriptive. A positive mean does not guarantee stable gain, particularly with20takes. Preserve failures and non-improvements. Do not compare against the old author-weight baseline to claim method gain.
