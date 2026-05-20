# INVALID ABLATION OUTPUTS - DO NOT USE FOR PAPER

Generated: 2026-04-18 (all files)  
Archived: 2026-05-20 (audit FIX_SESSION_07)

## Why invalid

These outputs were generated with the following known code defects,
all fixed in FIX_SESSION_01 through FIX_SESSION_06:

| Defect | Fixed in |
|---|---|
| url_count, FB_only_flag, language_mismatch all-zero | FIX_SESSION_01 |
| Metadata features unscaled (RobustScaler not applied) | FIX_SESSION_02 |
| language_location_mismatch dropped, 17->16 cascade | FIX_SESSION_02 |
| Cross-attention K/V self-reference reverted | FIX_SESSION_03 |
| Residual + learnable gates + proj init missing | FIX_SESSION_04 |
| InfoNCE valid_mask image-only; text dropout active | FIX_SESSION_05 |
| Trainer-level compound dropout | FIX_SESSION_06 |

## Do not use for:
- Paper result tables
- Ablation section comparisons
- Any reported metric

## Regenerate after:
1. Full model retrain with fixed code
2. scripts/run_ablations.py re-run (fixed in FIX_SESSION_07 Part B)
