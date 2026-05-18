# 00 · CONTEXT — TMD Project State

> Single source of truth về project hiện tại. Update sau MỖI check.
> Đầu file: project intent (mục tiêu gốc). Cuối file: current state (snapshot mới nhất).

---

## Project Intent (Original Scope — DO NOT MODIFY)

> Đây là design gốc trước khi drift. Fill xong, KHÔNG được sửa nữa — đây là north star.

**Project**: TMD — Tri-Modal Detection for Vietnamese Facebook ad misinformation

**Architecture (intent)**:
- Text encoder: PhoBERT-base (Vietnamese)
- Image encoder: ViT-B/16
- Metadata encoder: MLP over 17 engineered behavioral features
  - List 9 features: `ads_per_page`, `platform_count`, `FB_only_flag`, `all_targeted`, `burstiness`, `avg_ad_duration`, `launch_delay`, `num_countries`, `language_location_mismatch`, `emoji_count`, `text_length`, `ads_duration`, `repeated_text_ratio`, `exclamation_ratio`, `caps_word_ratio`, `repeated_punct_count`, `url_count`
- Fusion: dual bidirectional cross-attention → nonlinear gated fusion → residual
- Auxiliary loss: InfoNCE contrastive
- Training: two-phase (frozen → selectively unfrozen)
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

> Things bạn nhớ là đã thay đổi nhưng không chắc tác động.
> Fill vào ngay khi nhớ ra. Đây sẽ là input cho audit checks.

| Date | Component | What changed | Source | Risk noted? |
|---|---|---|---|---|
| 2026-04-12 | Feature engineering | Selected 9 metadata features from correlation analysis: `ads_per_page`, `platform_count`, `FB_only_flag`, `all_targeted`, `burstiness`, `avg_ad_duration`, `launch_delay`, `num_countries`, `language_location_mismatch` | Claude Sonnet 4.6 | No |
| 2026-04-12 | Architecture | Designed V1 tri-modal architecture: PhoBERT + ViT-B/16 + MLP encoder with dual cross-attention, nonlinear gated fusion, InfoNCE contrastive loss | Claude Sonnet 4.6 | No |
| 2026-04-12 | Cross-attention | V1 cross-attention implemented as Q=Text, K/V=[Image, Metadata] only — text excluded from its own K/V; query modality fully replaced by attention-weighted noise | Claude Sonnet 4.6 | No |
| 2026-04-12 | Text encoder | PhoBERT used with default `pooler_output` (randomly initialized pooler head) instead of `last_hidden_state[:,0,:]` | Claude Sonnet 4.6 | No |
| 2026-04-28 | Dataset | Page-level split enforced (70/15/15 by `page_id`); leakage audit added after training run showed val F1=1.000 in early epochs | Claude Sonnet 4.6 | Yes |
| 2026-04-28 | Feature engineering | Features expanded from 9 to 13 then to 17: added `emoji_count`, `text_length`, `ads_duration`, `repeated_text_ratio`, `exclamation_ratio`, `caps_word_ratio`, `repeated_punct_count`, `url_count` | Claude Sonnet 4.6 | No |
| 2026-04-28 | Training | Contrastive loss collapse diagnosed: `con=4.843 ≈ log(128)` constant across all epochs — embeddings collapsing to constant vector | User + Claude Sonnet 4.6 | Yes |
| 2026-04-28 | Training | Checkpoint tracking bug found: `best_checkpoint_path` remained `None` throughout training; wrong checkpoint loaded for test eval | User + Claude Sonnet 4.6 | Yes |
| 2026-04-28 | Training | Phase 2 LR oscillation: `lr_encoders=1e-5` too high for `batch_size=128`; val F1 oscillated ±0.07 across epochs | User + Claude Sonnet 4.6 | Yes |
| 2026-04-28 | Training | Early stopping patience=5 too short given oscillation; best epoch (6, F1=0.699) missed, stopped at epoch 11 (F1=0.661) | User + Claude Sonnet 4.6 | Yes |
| 2026-05-05 | Cross-attention | **CRITICAL FIX**: Self-reference added to K/V. Before: `K=[i,m]` only. After: `K=[t,i,m]` — text now attends to itself plus other modalities | Claude Sonnet 4.6 | Yes |
| 2026-05-05 | Cross-attention | Strong residual added: `output = LayerNorm(input + attn_output)` — guarantees output quality ≥ input quality | Claude Sonnet 4.6 | Yes |
| 2026-05-05 | Cross-attention | Learnable per-dimension gates added: `σ(θ_t) ∈ ℝ²⁵⁶`, initialized at −2.0 (sigmoid≈0.12) so attention starts near-identity | Claude Sonnet 4.6 | No |
| 2026-05-05 | Cross-attention | Attention output projection scaled ×0.1 at init — cross-attention starts as near-identity transformation | Claude Sonnet 4.6 | No |
| 2026-05-05 | Text encoder | Fixed: switched from `pooler_output` (randomly initialized) to `last_hidden_state[:,0,:]` via `add_pooling_layer=False` | Claude Sonnet 4.6 | Yes |
| 2026-05-05 | Optimizer | Phase 1 param groups fixed to exclude frozen encoder params — prevents wasted momentum buffers (~1.3 GB GPU waste) | Claude Sonnet 4.6 | Yes |
| 2026-05-05 | Contrastive loss | L2 normalization added before cosine similarity; `valid_mask` added to exclude modality-dropout-zeroed samples from InfoNCE | Claude Sonnet 4.6 | Yes |
| 2026-05-05 | Early stopping | Patience increased from 5 to 8; EMA smoothing (α=0.7) added to metric before stopping decision | Claude Sonnet 4.6 | No |
| 2026-05-05 | Training | `lr_encoders` reduced from 1e-5 to 3e-6; Phase 2 warmup added separately from Phase 1 warmup | Claude Sonnet 4.6 | No |
| 2026-05-05 | Loss | Label smoothing added (ε=0.05): targets softened from hard 0/1 to 0.025/0.975 for noisy label handling | Claude Sonnet 4.6 | No |
| 2026-05-05 | Metadata encoder | Architecture changed from 9→256→256→256 (19.7× first-layer jump) to gradual 9→64→128→256 with residual projection | Claude Sonnet 4.6 | No |
| 2026-05-11 | Validation | After cross-attention fix: F1=0.7931, AUC=0.8604, threshold=0.84 — significant improvement from 0.6879 baseline | User observation | — |
| 2026-05-11 | Calibration | Classifier bias init planned: `log(0.609/0.391) = 0.443` to match class prior — STRATEGY 2, not yet applied | Claude Sonnet 4.6 | No |
| 2026-05-11 | Architecture | Auxiliary modality heads planned: 3 lightweight heads (256→64→1), `aux_lambda=0.1`, auto-disabled for single-modality ablations — STRATEGY 3, not yet applied | Claude Sonnet 4.6 | No |
| 2026-05-11 | Architecture | SelectiveCrossAttention planned: per-sample gate `σ(MLP(concat(t,i,m)))` init at −1.0 — STRATEGY 1, not yet applied | Claude Sonnet 4.6 | No |
| 2026-05-17 | Baseline | GossipCop run with PhoBERT→RoBERTa substitution during project refactor; F1=0.76 likely wrong due to silent refactoring bugs | User + Claude Sonnet 4.6 | Yes |
---

## Current State (Updated Each Cycle)

> Snapshot of what's TRULY in the codebase right now. Updated after each check based on findings.

### Active code path (per C02 run trace)
Entry: scripts/train.py
→ src.data.dataset.create_datasets / AdDataset
→ src.data.preprocessing (compute_class_weights/compute_pos_weight; _fixed used only in tests)
→ src.models.factory.build_model → src.models.full_model / unimodal / bimodal
   → text_encoder, image_encoder, metadata_encoder
   → projection, cross_attention, gated_fusion
   → classification_head
→ src.training.trainer
   → src.losses.combined_loss → src.losses.contrastive
   → src.evaluation.metrics
   → src.training.early_stopping, optim, scheduler

### Known ambiguities (post‑C02)
- Active configs (training/*.yaml, ablation_*.yaml usage)
- Canonical preprocessing path (preprocessing_fixed is test‑only variant)
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
- Raw CSV: `data/raw/ads_vietnam_clean.csv` (utf‑8), 16,678 rows × 18 columns
- Columns: id, page_id, page_name, ad_creation_time, ad_delivery_start_time, ad_delivery_stop_time,
  ad_creative_bodies, ad_creative_link_titles, currency, impressions, spend, target_gender,
  target_ages, target_locations, languages, publisher_platforms, ad_snapshot_url, misinformation
- Label: `misinformation` (0/1) with 1 null; class balance ~0.6089 / 0.3911
- IDs: `id` is string; 16,672 unique IDs (6 duplicates)
- Pages: `page_id` string; 2,554 pages; ads/page mean 6.53, median 2, max 467; pages with 1 ad = 1,112
- Text: `ad_creative_bodies` nulls=436; `ad_creative_link_titles` nulls=2,889; values stored as list‑like strings
- Dates: creation/start range 2024‑12‑12 → 2026‑02‑08; stop_time nulls=3,045
- Images: `data/raw/ad_images` contains 16,678 PNGs; match coverage ~93.8% (1,037 ads missing images), 1,043 orphan images
- Splits present: `data/processed/splits/{train,val,test}.csv`
- C06 artifact verification: page overlap is clean (`train∩val=0`, `train∩test=0`, `val∩test=0`), total unique pages=2,554
- Split sizes (artifact): train=10,874; val=3,296; test=2,501; total=16,671 (= raw 16,678 minus 1 null label minus 6 duplicate IDs)
- Image coverage (artifact): train=93.711%, val=94.086%, test=93.563%
- Stratification (artifact pos_rate): train=0.6140, val=0.5890, test=0.6130 (val drift ~2.5pp)
- Training data source (code): `scripts/train.py` loads `processed_dir/splits/{train,val,test}.csv` via `create_datasets`; `AdDataset` does not apply metadata scaling at runtime
- Metadata scaling: fixed in FIX_SESSION_02. Canonical path (`src/data/preprocessing.py`) now fits on train and applies to train/val/test; scaler artifact verified loadable.
- Raw‐text url_count precompute and list‐like parsing remain active; `language_location_mismatch` was dropped by decision (ISSUE-012 Option C).
- Effective metadata feature count from artifacts: 16/16
- Post‐regen check (2026-05-19): train=10,874, val=3,296, test=2,501

### Model architecture (as-coded, not as-intended)
_Filled by C07–C09_

### Training behavior
_Filled by C10–C12_

### Evaluation setup
_Filled by C13–C14_

---

## Reconciliation Status

> After audit, list each Intent component vs Current state.
> Filled progressively.

| Component | Intent | Current | Status | Action |
|---|---|---|---|---|
| Metadata feature count | 9 (original) → 17 (post-expansion) | 17 confirmed (C02 base.yaml) | ⚠️ INTENT-DRIFT | Decide canonical # in C06; document rationale for expansion |
| Active preprocessing module | preprocessing.py | preprocessing.py (per C02) | ✅ MATCHES | None |
| Active write path (splits) | single canonical writer | `src/data/preprocessing.py::run_preprocessing_pipeline` (confirmed in FIX_SESSION_02 A.0) | ✅ MATCHES | Keep prepare_data main as orchestrator |
| Cross-attention K/V | K=[t,i,m] (after 2026-05-05 fix) | confirmed in code via Drift Log | ✅ MATCHES (verify in C08) | C08 |
| PhoBERT pooler | last_hidden_state[:,0,:] | confirmed (per Drift Log 2026-05-05) | ✅ MATCHES (verify in C07) | C07 |
| Dataset size | ~16,679 ads | 16,678 rows | ⚠️ DRIFT (minor) | Check missing row / duplicates in C05 |
| Splits | 70/15/15 page-level | Artifact verified page-isolated (0/0/0 overlaps), sizes 10877/3299/2501, pos_rate drift in val (~0.589); post‑regen train rows 10,874 (val/test not rechecked) | ✅ MATCHES (with minor drift) | Track stratification drift; not critical |
| Class balance | ~0.609/0.391 (implied by Strategy 2) | 0.6089 / 0.3911 | ✅ MATCHES | None |
| Modality dropout | p=0.15 | UNKNOWN | 🤔 UNKNOWN | C11 |
| Encoder/fusion subdirs | not in intent | DEAD code present | ⚠️ DRIFT (cosmetic) | Cleanup after audit |
| Feature count computed | 17 features | 16 active features (language_location_mismatch dropped by Option C) | ✅ MATCHES (audit decision) | Document as paper limitation |
| Metadata scaling | RobustScaler applied | Train/val/test transformed from train-fitted scaler; scaler artifact loadable | ✅ MATCHES | Monitor val/test degenerate-IQR features separately |
| Leakage status (preprocess) | Split → train‑fit → apply | Code matches | ✅ MATCHES | None |
| Dataset size | ~16,679 ads | 16,678 rows (1 missing or off-by-one) |⚠️ DRIFT (minor; need to resolve in C05 whether duplicates count)||

Status legend:
- ✅ MATCHES — current matches intent
- ⚠️ DRIFT — current differs from intent, evaluate impact
- ❌ BROKEN — current is buggy
- 🤔 UNKNOWN — not yet audited
- 🆕 IMPROVEMENT — drift, but better than intent → keep
