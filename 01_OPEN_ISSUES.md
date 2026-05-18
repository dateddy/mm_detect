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

### ISSUE-012 · Significant null rates in feature-relevant columns
- Discovered in: CHECK_03
- Severity: MEDIUM (depends on C04 finding)
- Status: INVESTIGATING
- Component: data / features
- Description: 4 columns used as inputs to the 17 features have non-trivial null rates: target_locations (6%), languages (5%), target_gender/ages (3%), ad_delivery_stop_time (18.2%). C04 must verify how preprocessing handles these nulls.
- Risk categories: correctness, validity_threat

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
- **Status**: FIX_IN_PROGRESS
- **Component**: data
- **Description**:
  `id` has 6 duplicate values (unique IDs < total rows). Preprocessing reads CSV with default dtype (no explicit string enforcement), and `fix_csv_ids.py` is a manual script not integrated into the pipeline. Scientific‑notation IDs remain possible.

- **Mini-check (2026-05-18)**:
  - dtype enforcement + dedup added in preprocessing
  - post‑regen train rows: 10,874

- **Impact**:
  Can cause incorrect image matching, orphan images, and sample duplication in training.

- **Reproducer**:
  ```bash
  # read-only
  python - <<'PY'
  import pandas as pd
  df = pd.read_csv('data/raw/ads_vietnam_clean.csv', encoding='utf-8')
  print('duplicates:', len(df) - df['id'].nunique())
  print(df['id'].astype(str).head())
  PY
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
