# 00 Â· CONTEXT â€” TMD Project State

> Single source of truth vá» project hiá»‡n táº¡i. Update sau Má»–I check.
> Äáº§u file: project intent (má»¥c tiÃªu gá»‘c). Cuá»‘i file: current state (snapshot má»›i nháº¥t).

---

## Project Intent (Original Scope â€” DO NOT MODIFY)

> ÄÃ¢y lÃ  design gá»‘c trÆ°á»›c khi drift. Fill xong, KHÃ”NG Ä‘Æ°á»£c sá»­a ná»¯a â€” Ä‘Ã¢y lÃ  north star.

**Project**: TMD â€” Tri-Modal Detection for Vietnamese Facebook ad misinformation

**Architecture (intent)**:
- Text encoder: PhoBERT-base (Vietnamese)
- Image encoder: ViT-B/16
- Metadata encoder: MLP over 17 engineered behavioral features
  - List 9 features: `ads_per_page`, `platform_count`, `FB_only_flag`, `all_targeted`, `burstiness`, `avg_ad_duration`, `launch_delay`, `num_countries`, `language_location_mismatch`, `emoji_count`, `text_length`, `ads_duration`, `repeated_text_ratio`, `exclamation_ratio`, `caps_word_ratio`, `repeated_punct_count`, `url_count`
- Fusion: dual bidirectional cross-attention â†’ nonlinear gated fusion â†’ residual
- Auxiliary loss: InfoNCE contrastive
- Training: two-phase (frozen â†’ selectively unfrozen)
- Optimizer: AdamW with cosine decay
- Modality dropout: p=0.15
- Early stop on val macro-F1

**Dataset**:
- ~16,679 labeled Vietnamese FB ads
- CSV with ~18 columns + image directory of PNGs keyed by ad ID
- Train/val/test splits: 70/15/15

**Evaluation**:
- Primary metric: macro-F1
- Test set held out, ablation done

**Original deliverables**:
- Trained model + checkpoint
- Ablation study with [list ablations intended]
- Error analysis section
- Paper draft (LaTeX)

---

## What Changed (Drift Log)

> Things báº¡n nhá»› lÃ  Ä‘Ã£ thay Ä‘á»•i nhÆ°ng khÃ´ng cháº¯c tÃ¡c Ä‘á»™ng.
> Fill vÃ o ngay khi nhá»› ra. ÄÃ¢y sáº½ lÃ  input cho audit checks.

| Date | Component | What changed | Source | Risk noted? |
|---|---|---|---|---|
| 2026-04-12 | Feature engineering | Selected 9 metadata features from correlation analysis: `ads_per_page`, `platform_count`, `FB_only_flag`, `all_targeted`, `burstiness`, `avg_ad_duration`, `launch_delay`, `num_countries`, `language_location_mismatch` | Claude Sonnet 4.6 | No |
| 2026-04-12 | Architecture | Designed V1 tri-modal architecture: PhoBERT + ViT-B/16 + MLP encoder with dual cross-attention, nonlinear gated fusion, InfoNCE contrastive loss | Claude Sonnet 4.6 | No |
| 2026-04-12 | Cross-attention | V1 cross-attention implemented as Q=Text, K/V=[Image, Metadata] only â€” text excluded from its own K/V; query modality fully replaced by attention-weighted noise | Claude Sonnet 4.6 | No |
| 2026-04-12 | Text encoder | PhoBERT used with default `pooler_output` (randomly initialized pooler head) instead of `last_hidden_state[:,0,:]` | Claude Sonnet 4.6 | No |
| 2026-04-28 | Dataset | Page-level split enforced (70/15/15 by `page_id`); leakage audit added after training run showed val F1=1.000 in early epochs | Claude Sonnet 4.6 | Yes |
| 2026-04-28 | Feature engineering | Features expanded from 9 to 13 then to 17: added `emoji_count`, `text_length`, `ads_duration`, `repeated_text_ratio`, `exclamation_ratio`, `caps_word_ratio`, `repeated_punct_count`, `url_count` | Claude Sonnet 4.6 | No |
| 2026-04-28 | Training | Contrastive loss collapse diagnosed: `con=4.843 â‰ˆ log(128)` constant across all epochs â€” embeddings collapsing to constant vector | User + Claude Sonnet 4.6 | Yes |
| 2026-04-28 | Training | Checkpoint tracking bug found: `best_checkpoint_path` remained `None` throughout training; wrong checkpoint loaded for test eval | User + Claude Sonnet 4.6 | Yes |
| 2026-04-28 | Training | Phase 2 LR oscillation: `lr_encoders=1e-5` too high for `batch_size=128`; val F1 oscillated Â±0.07 across epochs | User + Claude Sonnet 4.6 | Yes |
| 2026-04-28 | Training | Early stopping patience=5 too short given oscillation; best epoch (6, F1=0.699) missed, stopped at epoch 11 (F1=0.661) | User + Claude Sonnet 4.6 | Yes |
| 2026-05-05 | Cross-attention | **CRITICAL FIX**: Self-reference added to K/V. Before: `K=[i,m]` only. After: `K=[t,i,m]` â€” text now attends to itself plus other modalities | Claude Sonnet 4.6 | Yes |
| 2026-05-05 | Cross-attention | Strong residual added: `output = LayerNorm(input + attn_output)` â€” guarantees output quality â‰¥ input quality | Claude Sonnet 4.6 | Yes |
| 2026-05-05 | Cross-attention | Learnable per-dimension gates added: `Ïƒ(Î¸_t) âˆˆ â„Â²âµâ¶`, initialized at âˆ’2.0 (sigmoidâ‰ˆ0.12) so attention starts near-identity | Claude Sonnet 4.6 | No |
| 2026-05-05 | Cross-attention | Attention output projection scaled Ã—0.1 at init â€” cross-attention starts as near-identity transformation | Claude Sonnet 4.6 | No |
| 2026-05-05 | Text encoder | Fixed: switched from `pooler_output` (randomly initialized) to `last_hidden_state[:,0,:]` via `add_pooling_layer=False` | Claude Sonnet 4.6 | Yes |
| 2026-05-05 | Optimizer | Phase 1 param groups fixed to exclude frozen encoder params â€” prevents wasted momentum buffers (~1.3 GB GPU waste) | Claude Sonnet 4.6 | Yes |
| 2026-05-05 | Contrastive loss | L2 normalization added before cosine similarity; `valid_mask` added to exclude modality-dropout-zeroed samples from InfoNCE | Claude Sonnet 4.6 | Yes |
| 2026-05-05 | Early stopping | Patience increased from 5 to 8; EMA smoothing (Î±=0.7) added to metric before stopping decision | Claude Sonnet 4.6 | No |
| 2026-05-05 | Training | `lr_encoders` reduced from 1e-5 to 3e-6; Phase 2 warmup added separately from Phase 1 warmup | Claude Sonnet 4.6 | No |
| 2026-05-05 | Loss | Label smoothing added (Îµ=0.05): targets softened from hard 0/1 to 0.025/0.975 for noisy label handling | Claude Sonnet 4.6 | No |
| 2026-05-05 | Metadata encoder | Architecture changed from 9â†’256â†’256â†’256 (19.7Ã— first-layer jump) to gradual 9â†’64â†’128â†’256 with residual projection | Claude Sonnet 4.6 | No |
| 2026-05-11 | Validation | After cross-attention fix: F1=0.7931, AUC=0.8604, threshold=0.84 â€” significant improvement from 0.6879 baseline | User observation | â€” |
| 2026-05-11 | Calibration | Classifier bias init planned: `log(0.609/0.391) = 0.443` to match class prior â€” STRATEGY 2, not yet applied | Claude Sonnet 4.6 | No |
| 2026-05-11 | Architecture | Auxiliary modality heads planned: 3 lightweight heads (256â†’64â†’1), `aux_lambda=0.1`, auto-disabled for single-modality ablations â€” STRATEGY 3, not yet applied | Claude Sonnet 4.6 | No |
| 2026-05-11 | Architecture | SelectiveCrossAttention planned: per-sample gate `Ïƒ(MLP(concat(t,i,m)))` init at âˆ’1.0 â€” STRATEGY 1, not yet applied | Claude Sonnet 4.6 | No |
| 2026-05-17 | Baseline | GossipCop run with PhoBERTâ†’RoBERTa substitution during project refactor; F1=0.76 likely wrong due to silent refactoring bugs | User + Claude Sonnet 4.6 | Yes |
---

## Current State (Updated Each Cycle)

> Snapshot of what's TRULY in the codebase right now. Updated after each check based on findings.

### Active code path (per C02 run trace)
Entry: scripts/train.py
â†’ src.data.dataset.create_datasets / AdDataset
â†’ src.data.preprocessing (compute_class_weights/compute_pos_weight; _fixed used only in tests)
â†’ src.models.factory.build_model â†’ src.models.full_model / unimodal / bimodal
   â†’ text_encoder, image_encoder, metadata_encoder
   â†’ projection, cross_attention, gated_fusion
   â†’ classification_head
â†’ src.training.trainer
   â†’ src.losses.combined_loss â†’ src.losses.contrastive
   â†’ src.evaluation.metrics
   â†’ src.training.early_stopping, optim, scheduler

### Known ambiguities (postâ€‘C02)
- Active configs (training/*.yaml, ablation_*.yaml usage)
- Canonical preprocessing path (preprocessing_fixed is testâ€‘only variant)
- Stale entry point: scripts/preprocess.py appears to import missing functions

### Repository structure
./configs
./configs\ablation
./configs\model
./configs\training
./data
./data\embeddings
./data\processed
./data\processed\splits
./data\raw
./data\raw\ad_images
./experiments
./experiments\dry_run
./experiments\dry_run\logs
./experiments\dry_run\results
./experiments\exp_20260427_001306
./experiments\exp_20260427_001306\logs
./experiments\exp_20260427_001306\results
./experiments\exp_20260428_164231
./experiments\exp_20260428_164231\logs
./experiments\exp_20260428_164231\results
./experiments\test_full_mode
./experiments\test_full_mode\logs
./experiments\test_full_mode\results
./experiments\text_only
./experiments\text_only\logs
./experiments\text_only\results
./notebooks
./outputs
./outputs\ablations
./outputs\ablations\modality
./outputs\ablations\modality\full_model
./outputs\ablations\modality\image_metadata
./outputs\ablations\modality\image_only
./outputs\ablations\modality\metadata_only
./outputs\ablations\modality\text_image
./outputs\ablations\modality\text_metadata
./outputs\ablations\modality\text_only
./outputs\figures
./outputs\figures\feature_selection
./outputs\logs
./outputs\reports
./outputs\results
./outputs\tables
./scripts
./scripts\outputs
./scripts\outputs\logs
./src
./src\ablation
./src\config
./src\data
./src\evaluation
./src\losses
./src\models
./src\models\encoders
./src\models\fusion
./src\training
./src\utils

### Data pipeline
- Raw CSV: `data/raw/ads_vietnam_clean.csv` (utfâ€‘8), 16,678 rows Ã— 18 columns
- Columns: id, page_id, page_name, ad_creation_time, ad_delivery_start_time, ad_delivery_stop_time,
  ad_creative_bodies, ad_creative_link_titles, currency, impressions, spend, target_gender,
  target_ages, target_locations, languages, publisher_platforms, ad_snapshot_url, misinformation
- Label: `misinformation` (0/1) with 1 null; class balance ~0.6089 / 0.3911
- IDs: `id` is string; 16,672 unique IDs (6 duplicates)
- Pages: `page_id` string; 2,554 pages; ads/page mean 6.53, median 2, max 467; pages with 1 ad = 1,112
- Text: `ad_creative_bodies` nulls=436; `ad_creative_link_titles` nulls=2,889; values stored as listâ€‘like strings
- Dates: creation/start range 2024â€‘12â€‘12 â†’ 2026â€‘02â€‘08; stop_time nulls=3,045
- Images: `data/raw/ad_images` contains 16,678 PNGs; match coverage ~93.8% (1,037 ads missing images), 1,043 orphan images
- Splits present: `data/processed/splits/{train,val,test}.csv`
- C06 artifact verification: page overlap is clean (`trainâˆ©val=0`, `trainâˆ©test=0`, `valâˆ©test=0`), total unique pages=2,554
- Split sizes (artifact): train=10,874; val=3,296; test=2,501; total=16,671 (= raw 16,678 minus 1 null label minus 6 duplicate IDs)
- Image coverage (artifact): train=93.711%, val=94.086%, test=93.563%
- Stratification (artifact pos_rate): train=0.6140, val=0.5890, test=0.6130 (val drift ~2.5pp)
- Training data source (code): `scripts/train.py` loads `processed_dir/splits/{train,val,test}.csv` via `create_datasets`; `AdDataset` does not apply metadata scaling at runtime
- Metadata scaling: fixed in FIX_SESSION_02. Canonical path (`src/data/preprocessing.py`) now fits on train and applies to train/val/test; scaler artifact verified loadable.
- Rawâ€text url_count precompute and listâ€like parsing remain active; `language_location_mismatch` was dropped by decision (ISSUE-012 Option C).
- Effective metadata feature count from artifacts: 16/16
- Postâ€regen check (2026-05-19): train=10,874, val=3,296, test=2,501

### Model architecture (as-coded, not as-intended)
- Entry model: `src/models/full_model.py::MultimodalMisinfoDetector` (full modes) via `src/models/factory.py`
- Text encoder: `TextEncoder` wraps HF `AutoModel` with `add_pooling_layer=False`; uses `last_hidden_state[:,0,:]`; default model `vinai/phobert-base-v2`
- Image encoder: `ImageEncoder` wraps timm `vit_base_patch16_224`; uses `forward_features(... )[:,0,:]` CLS extraction
- Metadata encoder: `MetadataEncoder` is `16â†’64â†’128â†’256` MLP (BatchNorm + GELU + Dropout) with residual projection `Linear(16,256)`
- Projection space: all modalities projected/aligned to `projection_dim=256` (`text_proj`, `image_proj`, `meta_proj`)
- Missing-image handling: dataset substitutes zero image tensor `(3,224,224)` and model zero-masks projected image embedding via `missing_image` flag
- Two-phase training:
  - Phase 1: text/image encoders frozen; optimizer excludes frozen encoder params (empty placeholder groups)
  - Phase 2: at `epoch == freeze_encoder_epochs`, unfreeze top-k text/image blocks and inject params into encoder groups with `lr_encoders`
- Fusion (CHECK_08/C09): active `DualCrossAttention` now contains all 4 Drift Log 2026-05-05 stabilization fixes in main path: self-inclusive K/V, strong residual, per-dimension learnable gates (init -2.0), and output projection init scaling x0.1.
- Noted architectural concern from C07: `full_model.py` mode validator allows only `full*` modes while forward still contains unimodal branches (tracked as ISSUE-018)

### Training behavior
- Combined loss: `L_total = L_cls + lambda_con * L_con` in multimodal modes (`CombinedLoss`)
- Active main loss config: `cls_loss_type=bce` with `label_smoothing=0.05` and train-derived `pos_weight=n_neg/n_pos`
- Contrastive: InfoNCE with L2 normalization on both modalities and learnable temperature (`logit_scale`)
- Contrastive masking status: resolved in FIX_SESSION_05 via norm-based joint mask in `InfoNCELoss` (`||text||>1e-3 && ||image||>1e-3`) ANDed with upstream mask (ISSUE-021 closed)
- Optimizer/scheduler (C10): AdamW with Phase-1 empty encoder groups and epoch-triggered Phase-2 encoder param injection (`lr_encoders=2.0e-6`) is active
- Scheduler warmup (C10): single warmup+cosine LambdaLR is active; separate executable Phase-2 warmup is not implemented (tracked as ISSUE-022)
- Modality dropout (C11): trainer-level dropout call removed (FIX_SESSION_06); model-level `ModalityDropout` is sole active path with text dropout off and image/metadata dropout active

### Evaluation setup
_Filled by C13â€“C14_

---

## Reconciliation Status

> After audit, list each Intent component vs Current state.
> Filled progressively.

| Component | Intent | Current | Status | Action |
|---|---|---|---|---|
| Metadata feature count | 9 (original) â†’ 17 (post-expansion) | 16 active features after ISSUE-012 Option C | âœ… MATCHES (audit decision) | Document in paper as explicit scope change |
| Active preprocessing module | preprocessing.py | preprocessing.py (per C02) | âœ… MATCHES | None |
| Active write path (splits) | single canonical writer | `src/data/preprocessing.py::run_preprocessing_pipeline` (confirmed in FIX_SESSION_02 A.0) | âœ… MATCHES | Keep prepare_data main as orchestrator |
| Cross-attention K/V | K=[t,i,m] (after 2026-05-05 fix) | Restored in active full path (`m_proj` present): both text and image query arms use `K/V=[t,i,m]` | âœ… MATCHES (C08) | Keep regression test for K/V construction |
| Cross-attention strong residual | `LayerNorm(input + attn_output)` | Implemented in active `DualCrossAttention` via `norm_text(t_proj + sigmoid(gate_text)*t_attn)` and `norm_image(i_proj + sigmoid(gate_image)*i_attn)` | ✅ MATCHES (FIX_SESSION_04) | Keep verification script `verify_issue020.py` |
| Cross-attention learnable gates | per-dim gate `σ(θ)` with init `-2.0` | `gate_text/gate_image` are `nn.Parameter((embed_dim,), -2.0)` and used with sigmoid in forward | ✅ MATCHES (FIX_SESSION_04) | Keep gate-init check in `verify_issue020.py` |
| Cross-attention output-proj init | output projection scaled `x0.1` at init | `attn_text_to_image.out_proj.weight` and `attn_image_to_text.out_proj.weight` scaled by 0.1 in `__init__` | ✅ MATCHES (FIX_SESSION_04) | Keep out-proj scale check in `verify_issue020.py` |
| PhoBERT pooler | last_hidden_state[:,0,:] | Confirmed in code (`TextEncoder` uses `last_hidden_state[:,0,:]`, pooler disabled) | âœ… MATCHES | None |
| Image encoder backbone | ViT-B/16 class | `timm` `vit_base_patch16_224` with CLS extraction | âœ… MATCHES | None |
| Metadata encoder topology | gradual + residual | `16→64→128→256` + residual `Linear(16,256)` | ✅ MATCHES | None |
| Two-phase freezing | freeze→selective unfreeze | Phase 1 frozen encoders + empty optimizer groups; Phase 2 epoch-triggered top-k unfreeze + encoder param injection | ✅ MATCHES | Verified in C10 |
| Main loss type | configurable BCE/Focal/Asymmetric | Active config: BCEWithLogitsLoss (`cls_loss_type=bce`) | ✅ MATCHES | Re-check if config switches in experiments |
| Label smoothing | ε=0.05 | Applied in BCE branch: `targets*(1-ε)+0.5*ε` | ✅ MATCHES | None |
| pos_weight chain | n_neg/n_pos -> BCE pos_weight | `compute_pos_weight` -> `scripts/train.py` -> `CombinedLoss` BCE instantiation | ✅ MATCHES | None |
| InfoNCE L2 normalization | normalize both embeddings before similarity | `F.normalize(text_emb)` and `F.normalize(image_emb)` in InfoNCELoss | ✅ MATCHES | None |
| InfoNCE valid_mask | exclude modality-dropout-zeroed samples | Joint norm-based filtering now active in `InfoNCELoss` and ANDed with upstream mask (FIX_SESSION_05) | ✅ MATCHES | Keep regression check for zero-embedding samples |
| Learnable temperature | trainable temperature parameter | `logit_scale` is `nn.Parameter`, clamped each forward, added to optimizer temperature group | ✅ MATCHES | Re-check behavior in C10 |
| Phase 2 warmup | separate warmup after encoder unfreeze | Transition records `phase2_start_step`/`phase2_warmup_steps` metadata, but no separate warmup LR schedule is applied | ⚠️ PARTIAL (C10) | Track ISSUE-022 |
| Dataset size | ~16,679 ads | 16,678 rows | ⚠️ DRIFT (minor) | Check missing row / duplicates in C05 |
| Splits | 70/15/15 page-level | Artifact verified page-isolated (0/0/0 overlaps), sizes 10877/3299/2501, pos_rate drift in val (~0.589); post‑regen train rows 10,874 (val/test not rechecked) | ✅ MATCHES (with minor drift) | Track stratification drift; not critical |
| Class balance | ~0.609/0.391 (implied by Strategy 2) | 0.6089 / 0.3911 | ✅ MATCHES | None |
| Modality dropout | p=0.15 | Model-level image/metadata dropout p=0.15; text dropout disabled in active path; no trainer-level call-site compounding | ✅ MATCHES (C11 rerun) | Keep ISSUE-023 regression check (call-site absence) |
| Encoder/fusion subdirs | not in intent | DEAD code present | âš ï¸ DRIFT (cosmetic) | Cleanup after audit |
| Feature count computed | 17 features | 16 active features (language_location_mismatch dropped by Option C) | âœ… MATCHES (audit decision) | Document as paper limitation |
| Metadata scaling | RobustScaler applied | Train/val/test transformed from train-fitted scaler; scaler artifact loadable | âœ… MATCHES | Monitor val/test degenerate-IQR features separately |
| Leakage status (preprocess) | Split â†’ trainâ€‘fit â†’ apply | Code matches | âœ… MATCHES | None |
| Dataset size | ~16,679 ads | 16,678 rows (1 missing or off-by-one) |âš ï¸ DRIFT (minor; need to resolve in C05 whether duplicates count)||
| Dataset size (final) | ~16,679 ads (intent) | 16,671 effective post-fixes (raw 16,678 âˆ’ 1 null âˆ’ 6 dups) | âœ… MATCHES (within rounding tolerance) | Supersedes earlier rows |

Status legend:
- âœ… MATCHES â€” current matches intent
- âš ï¸ DRIFT â€” current differs from intent, evaluate impact
- âŒ BROKEN â€” current is buggy
- ðŸ¤” UNKNOWN â€” not yet audited
- ðŸ†• IMPROVEMENT â€” drift, but better than intent â†’ keep





