# Ablation Outputs - Awaiting Post-Retrain Generation

Status: EMPTY - ready for post-retrain ablation run.

Previous outputs archived at: outputs/ablations_INVALID_prefixed_2026-04-18/  
Reason: generated with pre-fix code (see archived README_INVALID.md).

## Generation sequence
1. Complete full model retrain: `python scripts/train.py --config configs/base.yaml`
2. Run ablations: `python scripts/run_ablations.py --config configs/base.yaml`
3. Verify outputs appear in modality/ and component/ subdirectories
