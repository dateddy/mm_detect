# 01 · OPEN ISSUES — Discovered Problems

> Append-only log of issues found during audit. Move to FIXES.md when resolved.
> Newest issues at top.

---

## Severity Legend

- **CRITICAL** — affects validity of results / paper claims. Must fix before paper submission.
- **HIGH** — affects correctness of a major component but may not invalidate results.
- **MEDIUM** — code quality / maintainability / minor drift from intent.
- **LOW** — cosmetic, optimization, nice-to-have.

## Status Legend

- `OPEN` — discovered, not yet addressed
- `INVESTIGATING` — currently being analyzed in a check
- `DECISION_PENDING` — need to choose fix strategy + accept risks
- `FIX_IN_PROGRESS` — fix being applied
- `RESOLVED` — moved to FIXES.md

---

## Active Issues

### ISSUE-020 · Fusion stabilization trio missing in active DualCrossAttention

- **Discovered in**: CHECK_08 (post-FIX_SESSION_03 rerun)
- **Severity**: HIGH
- **Status**: RESOLVED
- **Component**: model / fusion
- **Description**:
  After ISSUE-019 was fixed, active `src/models/cross_attention.py` still lacks 3 of the 4
  2026-05-05 fusion fixes claimed in Drift Log:
  - strong residual (`LayerNorm(input + attn_output)`) is absent
  - per-dimension learnable gates (`theta` init `-2.0`) are absent
  - explicit output-projection init scaling (`×0.1`) is absent

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

### ISSUE-019 · CRITICAL FIX REVERTED — cross-attention K/V self-reference missing

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
  # read-only — confirm K/V construction:
  grep -A 5 "kv_image_metadata\|kv_text_metadata" src/models/cross_attention.py
  # Should show: torch.stack([i_proj, m_proj]...)  ← WRONG
  # Should be:   torch.stack([t_proj, i_proj, m_proj]...) ← CORRECT
```

- **Risks if not fixed before retrain**:
  - `validity_threat` — paper F1 unreproducible
  - `cascade` — all downstream ablation results potentially invalid

- **Resolution (FIX_SESSION_03, 2026-05-19)**:
  - Restored self-inclusive K/V in active code path:
    - text query arm now uses `kv_all = torch.stack([t_proj, i_proj, m_proj], dim=1)`
    - image query arm now uses `kv_all = torch.stack([t_proj, i_proj, m_proj], dim=1)`
  - Commit: `bd5f4ab` (`fix(ISSUE-019): restore K/V self-reference in DualCrossAttention`)
  - Verified via CPU forward shape checks and zero-modal self-reference behavior.

---

### ISSUE-018 · full_model ablation mode guard conflicts with unimodal branches/tests

- **Discovered in**: CHECK_07
- **Severity**: MEDIUM
- **Status**: OPEN
- **Component**: model / ablation / repro
- **Description**:
  `MultimodalMisinfoDetector` validates `ablation_mode` against only `full*` modes (`src/models/full_model.py:66-73`), but `forward()` still includes `text_only/image_only/metadata_only` branches (`src/models/full_model.py:333-343`), and `scripts/test_ablation_mode.py` directly instantiates `MultimodalMisinfoDetector` with unimodal modes.

- **Impact**:
  Direct full-model unimodal ablation tests can fail with mode-validation errors or rely on stale/unreachable branches. This increases confusion about canonical ablation entry points (factory vs direct class).

- **Reproducer**:
  ```bash
  # read-only static evidence
  rg -n "valid_modes|text_only|image_only|metadata_only" src/models/full_model.py scripts/test_ablation_mode.py
  ```

---

### ISSUE-016 · List-like text fields not parsed

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

### ISSUE-015 · Split CSV artifacts are unscaled (confirmed)

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

### ISSUE-014 · FB_only_flag degenerate (all zeros)

- **Discovered in**: CHECK_04
- **Severity**: MEDIUM
- **Status**: RESOLVED
- **Component**: data / features
- **Description**:
  `FB_only_flag` is always 0 in train split. `publisher_platforms` is stored as list‑like strings with single quotes (invalid JSON), so `json.loads` fails and fallback logic does not detect facebook‑only pages.

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

### ISSUE-013 · url_count always zero (clean_text order)

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

### ISSUE-012 · language_location_mismatch uses wrong column name

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

### ISSUE-017 · Significant null rates in feature-relevant columns (renumbered from duplicate ISSUE-012)

- **Discovered in**: CHECK_03
- **Severity**: MEDIUM
- **Status**: RESOLVED
- **Component**: data / features
- **Description**:
  4 columns used as inputs to the 17 features have non-trivial null rates: target_locations (6%), languages (5%), target_gender/ages (3%), ad_delivery_stop_time (18.2%).

- **Resolution (CHECK_04 Section 8 audit)**:
  Null handling verified explicitly:
  - target_gender null → all_targeted=1 fallback
  - target_locations null → 0 for num_countries / language_location_mismatch (latter now dropped per Option C)
  - languages null → unused after ISSUE-012 Option C
  - ad_delivery_stop_time null (18%) → fills with current date in compute_avg_ad_duration; ads_duration returns 0
  - ad_creative_bodies null (2.6%) → clean_text fills then restores NA; text features return 0

- **Note**: renumbered from duplicate ISSUE-012 number to maintain append-only discipline per AUDIT_RULES R7.

### ISSUE-011 · Null label present in raw CSV

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

### ISSUE-010 · Image coverage below 95%

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

### ISSUE-009 · ID duplication + scientific-notation mismatch

- **Discovered in**: CHECK_03
- **Severity**: HIGH
- **Status**: RESOLVED
- **Component**: data
- **Description**:
  `id` has 6 duplicate values (unique IDs < total rows). Preprocessing reads CSV with default dtype (no explicit string enforcement), and `fix_csv_ids.py` is a manual script not integrated into the pipeline. Scientific‑notation IDs remain possible.

- **Resolution (FIX_SESSION_01 + 02, 2026-05-18/19)**:
  - dtype={'id': str, 'page_id': str} enforced in preprocessing read_csv
  - drop_duplicates(subset='id', keep='first') added immediately after load
  - Post-regen: train=10,874, val=3,296, test=2,501 (raw 16,678 − 1 null − 6 dups = 16,671 ✓)

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

### ISSUE-008 · scripts/preprocess.py appears stale/broken

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

### ISSUE-001 · Undocumented __main__ entry points in src modules

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

### ISSUE-002 · Potential duplicate/legacy encoder modules under src/models/encoders

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

### ISSUE-003 · Duplicate preprocessing files: preprocessing.py vs preprocessing_fixed.py

- **Discovered in**: CHECK_01
- **Severity**: HIGH
- **Status**: DECISION_PENDING
- **Component**: data
- **Description**:
  `train.py` and most scripts import `src.data.preprocessing`, while `preprocessing_fixed.py` is used only in tests (`test_metadata_scaling.py`, `test_learnable_temperature.py`). Two distinct pipelines exist with different scaling logic.

- **Impact**:
  If the “fixed” scaling is the intended path, current training results may not reflect it.

- **Reproducer**:
  ```bash
  # read-only
  Select-String -Path .\**\*.py -Pattern "preprocessing_fixed"
  Select-String -Path .\**\*.py -Pattern "from src.data.preprocessing" | findstr /v preprocessing_fixed
  ```

- **Hypothesized cause** (optional):
  A late‑April refactor introduced a separate scaling pipeline for tests.

### ISSUE-004 · "FIX N" remediation history not documented
- Discovered in: CHECK_01
- Severity: HIGH
- Status: OPEN
- Component: process/repro
- Description: scripts/test_focal_loss.py (refs FIX 4), test_metadata_scaling.py (refs FIX 3), test_optimizer_fix.py, test_learnable_temperature.py exist as verification tests. Numbered FIX references imply prior fixes were tracked, but no FIX_LOG.md or similar exists in repo. Cannot reconstruct what was originally broken or whether all FIX N are still applied.
- Impact: reproducibility threat; may have lost track of architectural decisions.
- Risk categories: reproducibility, validity_threat

### ISSUE-005 · Encoder/fusion subdirs appear unused (legacy)

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

### ISSUE-006 · Unreferenced config files (possible dead or dynamic)
- Discovered in: CHECK_01
- Severity: MEDIUM
- Status: OPEN
- Component: config
- Description: configs/training/default.yaml, configs/training/fast_debug.yaml, configs/ablation_*.yaml (top-level) not referenced by any entry point per grep. May be loaded via dynamic mechanism (Hydra, importlib), or dead.
- Risk categories: validity_threat (if active with wrong values), maintenance

### ISSUE-007 · High-churn files = drift risk concentration
- Discovered in: CHECK_01
- Severity: HIGH (informational; concrete issues to be discovered in later checks)
- Status: INVESTIGATING
- Component: training, losses, configs
- Description: trainer.py (6 commits), train.py (5), requirements.txt (5), base.yaml (5), combined_loss.py (5) dominate last 30 commits. These are exactly the components most prone to drift. Specific issues will be discovered in C09, C10, C11.
- Risk categories: validity_threat, cascade

---

<!-- Template:

### ISSUE-NNN · [Short title]

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

_(empty — will be populated by audit checks)_
