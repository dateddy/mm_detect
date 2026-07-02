# 01 Â· OPEN ISSUES â€” Discovered Problems

> Append-only log of issues found during audit. Move to FIXES.md when resolved.
> Newest issues at top.

---

## Severity Legend

- **CRITICAL** â€” affects validity of results / paper claims. Must fix before paper submission.
- **HIGH** â€” affects correctness of a major component but may not invalidate results.
- **MEDIUM** â€” code quality / maintainability / minor drift from intent.
- **LOW** â€” cosmetic, optimization, nice-to-have.

## Status Legend

- `OPEN` â€” discovered, not yet addressed
- `INVESTIGATING` â€” currently being analyzed in a check
- `DECISION_PENDING` â€” need to choose fix strategy + accept risks
- `FIX_IN_PROGRESS` â€” fix being applied
- `RESOLVED` â€” moved to FIXES.md

---

## Active Issues

### ISSUE-024 · Early stopping monitors AUC-ROC, not macro-F1 as stated in Project Intent

- **Discovered in**: CHECK_12
- **Severity**: MEDIUM
- **Status**: OPEN
- **Component**: training / config
- **Description**:
  Project Intent states "Early stop on val macro-F1" (immutable north star).
  Active config: `training.early_stopping_metric: auc_roc`.
  Code selects best checkpoint based on AUC-ROC improvement, not F1.
- **Impact**:
  Best checkpoint may differ from the one that optimized F1. Paper claim
  "early stop on val macro-F1" is inaccurate if left unchanged.
- **Options**:
  A — Change config to `early_stopping_metric: macro_f1` (matches intent)
  B — Accept AUC-ROC, update paper to say "early stop on val AUC-ROC"
- **Decision needed**: before retrain

### ISSUE-026 · scripts/run_ablations.py API drift from current Trainer/Evaluate interfaces

- **Discovered in**: CHECK_14
- **Severity**: HIGH
- **Status**: RESOLVED
- **Component**: ablation / training orchestration
- **Description**:
  `scripts/run_ablations.py` appears out-of-sync with current `Trainer` interfaces.
  It constructs `Trainer` with positional args that do not match the current
  `Trainer.__init__` signature and calls `trainer.evaluate(..., load_best=True)`,
  while active `Trainer.evaluate` has no `load_best` parameter.

- **Impact**:
  Post-fix ablation reruns via this script are likely to fail or be unreliable,
  blocking regeneration of paper-grade ablation results.

- **Evidence**:
  - Current trainer signature: `src/training/trainer.py:37-47`
  - Current evaluate signature: `src/training/trainer.py:538`
  - Drifted calls in ablation runner:
    - `scripts/run_ablations.py:288-294`
    - `scripts/run_ablations.py:302`

- **Reproducer**:
  ```bash
  rg -n "Trainer\\(|evaluate\\(.*load_best" scripts/run_ablations.py src/training/trainer.py
  ```

- **Resolution (FIX_SESSION_07, 2026-05-20)**:
  - Updated `scripts/run_ablations.py` call sites to current APIs:
    - fixed `Trainer(...)` constructor usage to pass `config`, `model`, loaders, and `loss_fn`
    - removed deprecated `load_best=True` from `trainer.evaluate(...)`
    - removed reliance on `trainer.train()` return payload (current train returns `None`)
    - aligned dataset/offline embedding and worker config access with active config schema
  - Verification:
    - `python -m py_compile scripts/run_ablations.py` passes
    - AST parse/import trace passes
    - `python scripts/run_ablations.py --help` runs successfully
  - Commit:
    - `e457296` — `fix(ISSUE-026): update run_ablations.py API to match current interfaces`

---

### ISSUE-025 · Existing modality ablation artifacts are pre-fix and invalid for final claims

- **Discovered in**: CHECK_14
- **Severity**: CRITICAL
- **Status**: RESOLVED
- **Component**: evaluation / ablation / paper validity
- **Description**:
  All files under `outputs/ablations/modality/*` are timestamped `2026-04-18`,
  predating all critical fix sessions (`2026-05-18` onward). These artifacts were
  produced before scaling/fusion/dropout repairs and cannot support final claims.

- **Impact**:
  Any paper table using these artifacts would be methodologically invalid relative
  to the current fixed codebase.

- **Evidence**:
  - Per-directory earliest/latest timestamps: `2026-04-18 04:31:43`
  - Fix commits begin `2026-05-19` (`13cb755`, `caf7a4b`, `bd5f4ab`, `9c27c8c`, ...)
  - `outputs/ablations/modality/summary.csv` currently has empty metric fields

- **Reproducer**:
  ```bash
  Get-ChildItem -Path outputs/ablations/modality -Recurse -File | Sort-Object LastWriteTime
  git log --oneline --format="%ci %h %s" | Select-String -Pattern "fix\\(|FIX_SESSION"
  ```

- **Resolution (FIX_SESSION_07, 2026-05-20)**:
  - Archived legacy outputs to:
    - `outputs/ablations_INVALID_prefixed_2026-04-18/`
  - Added archive guardrail note:
    - `outputs/ablations_INVALID_prefixed_2026-04-18/README_INVALID.md`
  - Recreated clean empty ablation output structure with `.gitkeep` and new run instructions:
    - `outputs/ablations/README.md`
  - Verification:
    - archive README exists
    - clean `outputs/ablations/` structure exists
    - stale `.json/.csv/.pt` count in new ablations tree = `0`
  - Commit:
    - `ed8620a` — `audit(ISSUE-025): archive pre-fix ablation outputs, create clean structure`

---

### ISSUE-023 · Trainer-level modality dropout still drops text and compounds with model dropout

- **Discovered in**: CHECK_11
- **Severity**: HIGH
- **Status**: RESOLVED
- **Component**: training / modality dropout
- **Description**:
  FIX_SESSION_05 Option D disabled text dropout inside `MultimodalMisinfoDetector`, but the
  active training path still calls `Trainer.apply_modality_dropout()` before model forward.
  That trainer-level dropout can choose `"text"` and zero raw `input_ids` / `attention_mask`.
  The model then applies a second embedding-level `ModalityDropout` to image and metadata.

- **Impact**:
  The full training pipeline does not satisfy the intended "text dropout disabled" behavior.
  Image dropout can also compound across raw trainer dropout and model embedding dropout.
  In full mode, random raw image dropout is approximately `0.15 * 1/2 = 0.075`, then model
  image dropout applies at `0.15`, for roughly `21.4%` effective random image zeroing before
  structural missing images are considered.

- **Evidence**:
  - Trainer applies raw modality dropout before forward: `src/training/trainer.py:391-392`
  - Trainer still includes text as droppable: `src/training/trainer.py:213-215`
  - Trainer text zeroing path: `src/training/trainer.py:232-238`
  - Model-level text dropout is disabled: `src/models/full_model.py:313-315`
  - Model-level image/metadata dropout remains active: `src/models/full_model.py:315`

- **Reproducer**:
  ```bash
  rg -n "apply_modality_dropout|droppable.append\\(\"text\"\\)|input_ids\\]\\[mask_i\\]|self.modality_dropout\\(i_proj, m_proj\\)" src/training/trainer.py src/models/full_model.py
  ```

- **Resolution (FIX_SESSION_06, 2026-05-20)**:
  - Removed trainer-loop call site so `Trainer.apply_modality_dropout()` is no longer executed in active training:
    - deleted call in `src/training/trainer.py` train loop (line previously at ~392)
  - Kept model-level dropout as the single active mechanism:
    - `self.modality_dropout(i_proj, m_proj)` remains in `src/models/full_model.py`
  - Verification:
    - `rg` shows `apply_modality_dropout(` appears only at method definition
    - AST static check confirms zero call sites
    - import sanity: `src.training.trainer` and `src.models.full_model` import successfully
  - Commit:
    - `d8a6ad4` — `fix(ISSUE-023 B): remove trainer-level apply_modality_dropout call from train loop`

---

### ISSUE-022 · Phase 2 warmup is metadata-only (no effective LR warmup)

- **Discovered in**: CHECK_10
- **Severity**: HIGH
- **Status**: OPEN (DEFERRED)
- **Component**: training / scheduler
- **Description**:
  Drift Log claims a separate Phase 2 warmup was added, but active trainer/scheduler path
  does not apply a second warmup ramp after encoder unfreeze. At transition, the code sets
  `phase2_start_step` and `phase2_warmup_steps` metadata and updates `scheduler.base_lrs`
  for encoder groups, but does not instantiate/apply any separate warmup schedule.

- **Impact**:
  Encoder LR may jump directly into the ongoing global LambdaLR trajectory after unfreeze,
  which can differ materially from intended "Phase 2 warmup" behavior and affect stability
  and reproducibility versus the reported fixed pipeline.

- **Evidence**:
  - Transition metadata/log only: `src/training/trainer.py:344-353`
  - No use of `phase2_start_step` / `phase2_warmup_steps` in LR computation thereafter.
  - Single scheduler initialized once in `Trainer.__init__`: `src/training/trainer.py:82-92`
  - Scheduler implementation has one warmup window only: `src/training/scheduler.py:21-83`

- **Reproducer**:
  ```bash
  rg -n "phase2_start_step|phase2_warmup_steps|warmup_steps|get_scheduler|scheduler.step" src/training/trainer.py src/training/scheduler.py
  ```

- **Deferral note (FIX_SESSION_06)**:
  - No code-path warmup fix applied in this session by design.
  - Added explicit config note near Phase 2 settings in `configs/base.yaml`.
  - Conservative `lr_encoders` set to `2.0e-6` to reduce transition shock until proper Phase 2 warmup is implemented.

---
### ISSUE-021 · InfoNCE valid_mask is image-only in active training path

- **Discovered in**: CHECK_09
- **Severity**: HIGH
- **Status**: RESOLVED
- **Component**: losses / training
- **Description**:
  InfoNCE supports `valid_mask` filtering, but the active trainer path prioritizes
  `output["image_valid"]` from the model, and model output exposes only `i_valid`
  (image modality validity). This means samples with dropped text embeddings can still
  be treated as valid for contrastive loss if image remains valid.

- **Impact**:
  The 2026-05-05 intent was to exclude modality-dropout-zeroed samples from InfoNCE.
  Current behavior appears partial, potentially injecting noisy contrastive pairs and
  weakening alignment signal.

- **Evidence**:
  - `src/models/full_model.py:298-300` computes `(t_valid, i_valid, m_valid)` but returns only `"image_valid": i_valid` at `src/models/full_model.py:371`.
  - `src/training/trainer.py:407-411` and `426-430` pass `output["image_valid"]` preferentially to loss, falling back to batch `valid_mask` only when image_valid is None.
  - `src/losses/contrastive.py:103-107` then filters using that mask.

- **Reproducer**:
  ```bash
  rg -n "image_valid|valid_mask|modality_dropout" src/models/full_model.py src/training/trainer.py src/losses/contrastive.py
  ```

- **Resolution (FIX_SESSION_05, 2026-05-19)**:
  - Option B (`src/losses/contrastive.py`): added norm-based joint validity inside `InfoNCELoss.forward()`:
    - `_norm_mask = (||text_emb|| > 1e-3) & (||image_emb|| > 1e-3)`
    - `valid_mask` is ANDed with `_norm_mask` (or replaced by it when None)
  - Option D (`configs/base.yaml`, `src/models/full_model.py`): disabled text modality dropout while preserving image/metadata dropout:
    - `training.text_modality_dropout_p: 0.0`
    - `training.image_modality_dropout_p: 0.15`
    - full model now applies modality dropout to image/metadata only.
  - Commits:
    - `c944daa` fix(ISSUE-021 B)
    - `b0f7119` fix(ISSUE-021 D)
  - Verified by CPU-only synthetic checks (Option B, Option D, combined).

---
### ISSUE-020 Â· Fusion stabilization trio missing in active DualCrossAttention

- **Discovered in**: CHECK_08 (post-FIX_SESSION_03 rerun)
- **Severity**: HIGH
- **Status**: RESOLVED
- **Component**: model / fusion
- **Description**:
  After ISSUE-019 was fixed, active `src/models/cross_attention.py` still lacks 3 of the 4
  2026-05-05 fusion fixes claimed in Drift Log:
  - strong residual (`LayerNorm(input + attn_output)`) is absent
  - per-dimension learnable gates (`theta` init `-2.0`) are absent
  - explicit output-projection init scaling (`Ã—0.1`) is absent

- **Impact**:
  Reported architecture claims for the 2026-05-05 F1 jump are only partially represented in active code.
  This is a reproducibility/validity risk for future retraining and paper-method consistency.

- **Reproducer**:
  ```bash
  rg -n "LayerNorm|gate|sigmoid|Parameter|out_proj|0\\.1|xavier|kaiming" src/models/cross_attention.py
  ```

- **Resolution (FIX_SESSION_04, 2026-05-19)**:
  - Added per-arm strong residual in `DualCrossAttention`:
    - `t_prime = norm_text(t_proj + sigmoid(gate_text) * t_attn)`
    - `i_prime = norm_image(i_proj + sigmoid(gate_image) * i_attn)`
  - Added per-dimension learnable gates:
    - `gate_text`, `gate_image` as `nn.Parameter(torch.full((embed_dim,), -2.0))`
  - Added output projection stabilization init:
    - `attn_text_to_image.out_proj.weight *= 0.1`
    - `attn_image_to_text.out_proj.weight *= 0.1`
  - Commit: `9c27c8c`
  - Verified by: `python verify_issue020.py` (all 6 checks passed on CPU synthetic forward)

---

### ISSUE-019 Â· CRITICAL FIX REVERTED â€” cross-attention K/V self-reference missing

- **Discovered in**: CHECK_08 (abort at Step 2)
- **Severity**: CRITICAL
- **Status**: RESOLVED
- **Component**: model / cross-attention
- **Description**:
  Active `DualCrossAttention.forward()` had reverted to:
  - Text query: K/V = `torch.stack([i_proj, m_proj], dim=1)` (K=[i,m])
  - Image query: K/V = `torch.stack([t_proj, m_proj], dim=1)` (K=[t,m])
  which removed self-reference.

- **Impact**: CRITICAL
  - Was a direct architecture regression risk for retraining reproducibility.

- **Hypothesized cause**:
  High-churn commits on `src/models/cross_attention.py` (part of ISSUE-007 pattern).
  A coding agent likely reverted the self-reference change while making other edits,
  without recognizing this was the CRITICAL architectural fix.

- **Reproducer**:
```bash
  # read-only â€” confirm K/V construction:
  grep -A 5 "kv_image_metadata\|kv_text_metadata" src/models/cross_attention.py
  # Should show: torch.stack([i_proj, m_proj]...)  â† WRONG
  # Should be:   torch.stack([t_proj, i_proj, m_proj]...) â† CORRECT
```

- **Risks if not fixed before retrain**:
  - `validity_threat` â€” paper F1 unreproducible
  - `cascade` â€” all downstream ablation results potentially invalid

- **Resolution (FIX_SESSION_03, 2026-05-19)**:
  - Restored self-inclusive K/V in active code path:
    - text query arm now uses `kv_all = torch.stack([t_proj, i_proj, m_proj], dim=1)`
    - image query arm now uses `kv_all = torch.stack([t_proj, i_proj, m_proj], dim=1)`
  - Commit: `bd5f4ab` (`fix(ISSUE-019): restore K/V self-reference in DualCrossAttention`)
  - Verified via CPU forward shape checks and zero-modal self-reference behavior.

---

### ISSUE-018 Â· full_model ablation mode guard conflicts with unimodal branches/tests

- **Discovered in**: CHECK_07
- **Severity**: LOW
- **Status**: OPEN
- **Component**: model / ablation / repro
- **Description**:
  `MultimodalMisinfoDetector` validates `ablation_mode` against only `full*` modes (`src/models/full_model.py:66-73`), but `forward()` still includes `text_only/image_only/metadata_only` branches (`src/models/full_model.py:333-343`), and `scripts/test_ablation_mode.py` directly instantiates `MultimodalMisinfoDetector` with unimodal modes.

- **Impact**:
  Canonical factory ablations are not blocked, but stale/unreachable unimodal branches
  in `full_model.py` and direct-instantiation tests remain confusing and can mislead
  contributors about the supported entry path.

- **Reproducer**:
  ```bash
  # read-only static evidence
  rg -n "valid_modes|text_only|image_only|metadata_only" src/models/full_model.py scripts/test_ablation_mode.py
  ```

- **C14 update**:
  Factory-based unimodal instantiation passed for `text_only`, `image_only`,
  and `metadata_only` (dedicated model classes). This issue remains a cleanup/
  clarity concern rather than a runtime blocker.

---

### ISSUE-016 Â· List-like text fields not parsed

- **Discovered in**: CHECK_04
- **Severity**: MEDIUM
- **Status**: RESOLVED
- **Component**: data / text
- **Description**:
  `ad_creative_bodies` and related fields are stored as list-like strings (e.g., `['text1', 'text2']`) but preprocessing does not parse them. Text features operate on bracketed strings.

- **Impact**:
  Text length and regex-based features may be skewed; tokenization sees artifacts.

- **Reproducer**:
  ```bash
  # read-only: search for literal_eval usage (none found)
  ```

---

### ISSUE-015 Â· Split CSV artifacts are unscaled (confirmed)

- **Discovered in**: CHECK_04
- **Severity**: CRITICAL
- **Status**: RESOLVED
- **Component**: data / preprocessing
- **Description**:
  C06 artifact-level verification confirms `data/processed/splits/*.csv` are unscaled. RobustScaler signatures are absent (e.g., `ads_per_page` median=20, IQR=53; `platform_count` median=4, IQR=3). Although preprocessing code contains scaler logic, the active split artifacts consumed by training are not scaled.

- **Impact**:
  Reported training/evaluation metrics reflect unscaled metadata artifacts. This is a paper-validity and reproducibility risk until artifact generation path is reconciled and rerun.

- **C06 evidence (train split)**:
  - `ads_per_page`: median 20.0, IQR 53.0 (unscaled)
  - `platform_count`: median 4.0, IQR 3.0 (unscaled)
  - Most metadata features flagged `UNSCALED?` by the RobustScaler sanity check.

- **Mini-check (2026-05-18, post regen)**:
  - `ads_per_page`: median 20.000, IQR 53.000 (unscaled)
  - `platform_count`: median 4.000, IQR 3.000 (unscaled)
  - `avg_ad_duration`: median 225.763, IQR 621.246 (unscaled)
  - `text_length`: median 373.000, IQR 491.000 (unscaled)
  - Verdict: still unscaled after regen

- **Reproducer**:
  ```bash
  # read-only
  python - <<'PY'
  import pandas as pd
  train = pd.read_csv('data/processed/splits/train.csv')
  for f in ['ads_per_page','platform_count','avg_ad_duration','num_countries']:
      q1, q3 = train[f].quantile([0.25, 0.75])
      print(f, 'median=', train[f].median(), 'iqr=', q3-q1)
  PY
  ```

- **C06 update**:
  Escalated from INVESTIGATING/HIGH to OPEN/CRITICAL based on direct artifact checks.

- **Resolution (FIX_SESSION_02, 2026-05-19)**:
  - Canonical write path confirmed: `src/data/preprocessing.py::run_preprocessing_pipeline`
  - Scaler now applied to train/val/test (train-fitted)
  - Legacy `float_format='%.0f'` removed from `scripts/prepare_data.py` helper path
  - `metadata_scaler.pkl` persisted via `pickle.dump(..., 'wb')` and verified loadable

---

### ISSUE-014 Â· FB_only_flag degenerate (all zeros)

- **Discovered in**: CHECK_04
- **Severity**: MEDIUM
- **Status**: RESOLVED
- **Component**: data / features
- **Description**:
  `FB_only_flag` is always 0 in train split. `publisher_platforms` is stored as listâ€‘like strings with single quotes (invalid JSON), so `json.loads` fails and fallback logic does not detect facebookâ€‘only pages.

- **Impact**:
  Feature provides no signal; drift from intent.

- **C06 quantification**:
  - train unique values: 1
  - train % zeros: 100.0%
  - Cohen's d: 0.000
  - Mutual information: N/A (excluded from working features due to constant value)

- **Mini-check (2026-05-18)**:
  - train unique values: 2
  - train % non-zero: 16.7%

- **Reproducer**:
  ```bash
  # read-only
  python - <<'PY'
  import pandas as pd
  train = pd.read_csv('data/processed/splits/train.csv')
  print(train['FB_only_flag'].value_counts())
  PY
  ```

---

### ISSUE-013 Â· url_count always zero (clean_text order)

- **Discovered in**: CHECK_04
- **Severity**: HIGH
- **Status**: RESOLVED
- **Component**: data / features
- **Description**:
  `clean_text` removes URLs before `engineer_row_features` computes `url_count`, so URL counts are zero across the dataset.

- **Impact**:
  One of the 17 intended features is effectively broken.

- **C06 quantification**:
  - train unique values: 1
  - train % zeros: 100.0%
  - Cohen's d: 0.000
  - Mutual information: N/A (excluded from working features due to constant value)

- **Mini-check (2026-05-18)**:
  - train unique values: 14
  - train % non-zero: 3.8%

- **Reproducer**:
  ```bash
  # read-only
  python - <<'PY'
  import pandas as pd
  train = pd.read_csv('data/processed/splits/train.csv')
  print(train['url_count'].value_counts())
  PY
  ```

---

### ISSUE-012 Â· language_location_mismatch uses wrong column name

- **Discovered in**: CHECK_04
- **Severity**: HIGH
- **Status**: RESOLVED
- **Component**: data / features
- **Description**:
  `compute_language_location_mismatch` expects `ad_languages`, but raw data column is `languages`. The function returns 0 for all rows.

- **Impact**:
  Feature is always zero; drift from intent.

- **C06 quantification**:
  - train unique values: 1
  - train % zeros: 100.0%
  - Cohen's d: 0.000
  - Mutual information: N/A (excluded from working features due to constant value)

- **Mini-check (2026-05-18)**:
  - train unique values: 1
  - train % non-zero: 0.0%
  - Note: likely `target_locations` not ISO codes; mapping mismatch with COUNTRY_LANGUAGES

- **Reproducer**:
  ```bash
  # read-only
  python - <<'PY'
  import pandas as pd
  train = pd.read_csv('data/processed/splits/train.csv')
  print(train['language_location_mismatch'].value_counts())
  PY
  ```

- **Resolution (FIX_SESSION_02, 2026-05-19)**:
  Option C applied. `language_location_mismatch` was dropped from the active metadata feature set and cascaded from 17 to 16 features in config/model/test paths.

---

### ISSUE-017 Â· Significant null rates in feature-relevant columns (renumbered from duplicate ISSUE-012)

- **Discovered in**: CHECK_03
- **Severity**: MEDIUM
- **Status**: RESOLVED
- **Component**: data / features
- **Description**:
  4 columns used as inputs to the 17 features have non-trivial null rates: target_locations (6%), languages (5%), target_gender/ages (3%), ad_delivery_stop_time (18.2%).

- **Resolution (CHECK_04 Section 8 audit)**:
  Null handling verified explicitly:
  - target_gender null â†’ all_targeted=1 fallback
  - target_locations null â†’ 0 for num_countries / language_location_mismatch (latter now dropped per Option C)
  - languages null â†’ unused after ISSUE-012 Option C
  - ad_delivery_stop_time null (18%) â†’ fills with current date in compute_avg_ad_duration; ads_duration returns 0
  - ad_creative_bodies null (2.6%) â†’ clean_text fills then restores NA; text features return 0

- **Note**: renumbered from duplicate ISSUE-012 number to maintain append-only discipline per AUDIT_RULES R7.

### ISSUE-011 Â· Null label present in raw CSV

- **Discovered in**: CHECK_03
- **Severity**: LOW
- **Status**: RESOLVED
- **Component**: data
- **Description**:
  `misinformation` has 1 null value in `data/raw/ads_vietnam_clean.csv`.

- **Impact**:
  Minor, but should be zero after `fix_null_labels.py`. Needs cleanup to avoid edge-case failures.

- **Reproducer**:
  ```bash
  # read-only
  python - <<'PY'
  import pandas as pd
  df = pd.read_csv('data/raw/ads_vietnam_clean.csv', encoding='utf-8')
  print(df['misinformation'].isna().sum())
  PY
  ```

---

### ISSUE-010 Â· Image coverage below 95%

- **Discovered in**: CHECK_03
- **Severity**: MEDIUM
- **Status**: OPEN
- **Component**: data
- **Description**:
  Only ~93.8% of unique ad IDs have matching image files (1,037 missing images). There are also 1,043 orphan images.

- **Impact**:
  Image modality is absent for a non-trivial subset; may bias multimodal training.

- **Reproducer**:
  ```bash
  # read-only
  python - <<'PY'
  import os, pandas as pd
  df = pd.read_csv('data/raw/ads_vietnam_clean.csv', encoding='utf-8')
  image_files = set(os.listdir('data/raw/ad_images'))
  stems = set(os.path.splitext(f)[0] for f in image_files)
  ids = set(df['id'].astype(str))
  print(len(ids & stems), len(ids - stems), len(stems - ids))
  PY
  ```

---

### ISSUE-009 Â· ID duplication + scientific-notation mismatch

- **Discovered in**: CHECK_03
- **Severity**: HIGH
- **Status**: RESOLVED
- **Component**: data
- **Description**:
  `id` has 6 duplicate values (unique IDs < total rows). Preprocessing reads CSV with default dtype (no explicit string enforcement), and `fix_csv_ids.py` is a manual script not integrated into the pipeline. Scientificâ€‘notation IDs remain possible.

- **Resolution (FIX_SESSION_01 + 02, 2026-05-18/19)**:
  - dtype={'id': str, 'page_id': str} enforced in preprocessing read_csv
  - drop_duplicates(subset='id', keep='first') added immediately after load
  - Post-regen: train=10,874, val=3,296, test=2,501 (raw 16,678 âˆ’ 1 null âˆ’ 6 dups = 16,671 âœ“)

- **Impact**:
  Resolved. ID type drift eliminated; duplicate rows removed.

- **Reproducer** (for verification):
```bash
  python -c "
  import pandas as pd
  train = pd.read_csv('data/processed/splits/train.csv', dtype={'id':str})
  assert train['id'].nunique() == len(train), 'No dups expected'
  print('Train id unique check passed')
  "
```

---

### ISSUE-008 Â· scripts/preprocess.py appears stale/broken

- **Discovered in**: CHECK_02
- **Severity**: MEDIUM
- **Status**: OPEN
- **Component**: data / repro
- **Description**:
  `scripts/preprocess.py` imports functions (`load_raw_data`, `clean_text_features`, etc.) that are not present in `src.data.preprocessing`. Likely an outdated entry point.

- **Impact**:
  If used, preprocessing would fail. Confuses which preprocessing path is canonical.

- **Reproducer**:
  ```bash
  # read-only: compare scripts/preprocess.py imports vs src/data/preprocessing.py
  ```

---

### ISSUE-001 Â· Undocumented __main__ entry points in src modules

- **Discovered in**: CHECK_01
- **Severity**: HIGH
- **Status**: OPEN
- **Component**: data / training
- **Description**:
  `src/data/feature_engineering.py`, `src/data/preprocessing.py`, `src/data/preprocessing_fixed.py`, and `src/training/trainer.py` expose `__main__` entry points but are not referenced in scripts/README. Their intended usage is unclear.

- **Impact**:
  Unknown entry points may indicate legacy or alternate execution paths that diverge from the main training scripts.

- **Reproducer**:
  ```bash
  # read-only: inspect __main__ blocks in the listed files
  ```

- **Hypothesized cause** (optional):
  Legacy CLI/debug entry points left in place during iterative development.

---

### ISSUE-002 Â· Potential duplicate/legacy encoder modules under src/models/encoders

- **Discovered in**: CHECK_01
- **Severity**: MEDIUM
- **Status**: OPEN
- **Component**: model
- **Description**:
  Encoder implementations exist both in `src/models/` and `src/models/encoders/` with overlapping names. The latter appear unused by fully-qualified imports and may be legacy copies.

- **Impact**:
  Increases drift risk and confusion about which modules are authoritative.

- **Reproducer**:
  ```bash
  # read-only: compare encoder files under src/models/ and src/models/encoders/
  ```

### ISSUE-003 Â· Duplicate preprocessing files: preprocessing.py vs preprocessing_fixed.py

- **Discovered in**: CHECK_01
- **Severity**: HIGH
- **Status**: DECISION_PENDING
- **Component**: data
- **Description**:
  `train.py` and most scripts import `src.data.preprocessing`, while `preprocessing_fixed.py` is used only in tests (`test_metadata_scaling.py`, `test_learnable_temperature.py`). Two distinct pipelines exist with different scaling logic.

- **Impact**:
  If the â€œfixedâ€ scaling is the intended path, current training results may not reflect it.

- **Reproducer**:
  ```bash
  # read-only
  Select-String -Path .\**\*.py -Pattern "preprocessing_fixed"
  Select-String -Path .\**\*.py -Pattern "from src.data.preprocessing" | findstr /v preprocessing_fixed
  ```

- **Hypothesized cause** (optional):
  A lateâ€‘April refactor introduced a separate scaling pipeline for tests.

### ISSUE-004 Â· "FIX N" remediation history not documented
- Discovered in: CHECK_01
- Severity: HIGH
- Status: OPEN
- Component: process/repro
- Description: scripts/test_focal_loss.py (refs FIX 4), test_metadata_scaling.py (refs FIX 3), test_optimizer_fix.py, test_learnable_temperature.py exist as verification tests. Numbered FIX references imply prior fixes were tracked, but no FIX_LOG.md or similar exists in repo. Cannot reconstruct what was originally broken or whether all FIX N are still applied.
- Impact: reproducibility threat; may have lost track of architectural decisions.
- Risk categories: reproducibility, validity_threat

### ISSUE-005 Â· Encoder/fusion subdirs appear unused (legacy)

- **Discovered in**: CHECK_01
- **Severity**: MEDIUM
- **Status**: DECISION_PENDING
- **Component**: model
- **Description**:
  `src/models/encoders/` and `src/models/fusion/` have modules but are not imported anywhere. `src/models/fusion/fusion_block.py` explicitly states it is deprecated.

- **Impact**:
  Drift/maintenance risk; unclear authoritative implementation.

- **Reproducer**:
  ```bash
  # read-only
  Select-String -Path .\**\*.py -Pattern "from src.models.encoders|from src.models.fusion"
  ```

### ISSUE-006 Â· Unreferenced config files (possible dead or dynamic)
- Discovered in: CHECK_01
- Severity: MEDIUM
- Status: OPEN
- Component: config
- Description: configs/training/default.yaml, configs/training/fast_debug.yaml, configs/ablation_*.yaml (top-level) not referenced by any entry point per grep. May be loaded via dynamic mechanism (Hydra, importlib), or dead.
- Risk categories: validity_threat (if active with wrong values), maintenance

### ISSUE-007 Â· High-churn files = drift risk concentration
- Discovered in: CHECK_01
- Severity: HIGH (informational; concrete issues to be discovered in later checks)
- Status: INVESTIGATING
- Component: training, losses, configs
- Description: trainer.py (6 commits), train.py (5), requirements.txt (5), base.yaml (5), combined_loss.py (5) dominate last 30 commits. These are exactly the components most prone to drift. Specific issues will be discovered in C09, C10, C11.
- Risk categories: validity_threat, cascade

---

<!-- Template:

### ISSUE-NNN Â· [Short title]

- **Discovered in**: CHECK_NN
- **Severity**: CRITICAL | HIGH | MEDIUM | LOW
- **Status**: OPEN
- **Component**: data / model / training / eval / ablation / repro
- **Description**:
  What was found, where, evidence (file:line, error message, anomalous metric)

- **Impact**:
  What this means for the project (validity, results, paper claim affected)

- **Reproducer**:
  ```bash
  # command to reproduce or observe the issue
  ```

- **Hypothesized cause** (optional):
  Why this might have happened (e.g., coding agent edit on YYYY-MM-DD)

- **Linked to**: ISSUE-XXX (if related)

---
-->

_(empty â€” will be populated by audit checks)_



### ISSUE-027 · Merge 5359227 committed with unresolved conflict markers in 3 source files

- **Discovered in**: FIX_SESSION_08
- **Severity**: CRITICAL
- **Status**: RESOLVED
- **Component**: repo/merge
- **Description**:
  HEAD:scripts/run_ablations.py, src/models/cross_attention.py, src/training/trainer.py contain literal <<<<<<< HEAD / ======= / >>>>>>> audit/fix-session-07; committed code does not parse
- **Impact**:
  Any checkout of 5359227 fails to import all 3 files; branch is dead on arrival
- **Reproducer**:
  ```bash
  for f in scripts/run_ablations.py src/models/cross_attention.py src/training/trainer.py; do git show HEAD:$f>/tmp/c.py; python -m py_compile /tmp/c.py; done  # all FAIL
  ```
- **Hypothesized cause**:
  merge conflicts staged+committed without resolution; no conflict-marker guard

<!-- emitted 2026-07-02 -->

---

### ISSUE-028 · Botched working-tree resolution produced duplicate config= kwarg (SyntaxError)

- **Discovered in**: FIX_SESSION_08
- **Severity**: CRITICAL
- **Status**: RESOLVED
- **Component**: scripts/run_ablations
- **Description**:
  scripts/run_ablations.py Trainer(...) call had config=config twice (kept from both conflict sides)
- **Impact**:
  SyntaxError: keyword argument repeated: config -> run_ablations un-runnable
- **Reproducer**:
  ```bash
  python -m py_compile scripts/run_ablations.py  # before fix: SyntaxError line ~365
  ```
- **Linked to**: ISSUE-027

<!-- emitted 2026-07-02 -->

---

### ISSUE-029 · Test-set evaluate() call deleted during botched resolution

- **Discovered in**: FIX_SESSION_08
- **Severity**: HIGH
- **Status**: RESOLVED
- **Component**: scripts/run_ablations
- **Description**:
  scripts/run_ablations.py: active line 'test_metrics = trainer.evaluate(test_loader, split="test")' removed, only '# OLD:' comment left
- **Impact**:
  Ablations would compute no test metrics; Phase-4 results empty/invalid
- **Reproducer**:
  ```bash
  grep -nE 'trainer\.evaluate\(' scripts/run_ablations.py  # before fix: 0 live calls
  ```
- **Linked to**: ISSUE-027

<!-- emitted 2026-07-02 -->

---

### ISSUE-030 · output_dict=output dropped in both loss calls -> aux + InfoNCE silently broken

- **Discovered in**: FIX_SESSION_08
- **Severity**: HIGH
- **Status**: RESOLVED
- **Component**: src/training/trainer
- **Description**:
  src/training/trainer.py:~524 & ~543 lost output_dict=output; CombinedLoss.forward (combined_loss.py:345,372,420) needs it for embedding extraction + aux gate
- **Impact**:
  aux_* losses hard-zero and InfoNCE embeddings fall back to defaults; no crash, silent training-signal loss
- **Reproducer**:
  ```bash
  grep -nE 'output_dict=output' src/training/trainer.py  # before fix: 0 hits; after: 2
  ```
- **Linked to**: ISSUE-027

<!-- emitted 2026-07-02 -->

---

### ISSUE-031 · Stray one-off monkey-patch fix.py at repo root

- **Discovered in**: FIX_SESSION_08
- **Severity**: LOW
- **Status**: RESOLVED
- **Component**: repo/root
- **Description**:
  fix.py: ad-hoc script rewriting run_ablations.py DataLoader call; unrelated to conflicts, clutter
- **Impact**:
  Encourages patch-without-risk-control pattern the audit route forbids
- **Reproducer**:
  ```bash
  test -f fix.py && echo present  # before fix: present
  ```

<!-- emitted 2026-07-02 -->

---
