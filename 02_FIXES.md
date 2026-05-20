# 02 Â· FIXES â€” Applied Changes Log

> Chronological record of every change applied during audit recovery.
> Every entry must list risks accepted BEFORE the fix was made.
> This is the audit trail that prevents repeat of "agent fixes without risk control."

---

## Discipline

Before logging a fix here, the corresponding ISSUE must have:
1. Risks identified in OPEN_ISSUES.md
2. Explicit risk acceptance noted below
3. Reproducer command (so the fix can be verified)

---

## Fix Entries (newest at top)

### FIX-SESSION_06 · ISSUE-023 Option B + lr_encoders adjustment · 2026-05-20

- **Resolves**: ISSUE-023
- **Defers**: ISSUE-022 (documented only, no scheduler code change)
- **Component**: training + config
- **Files changed**:
  - `src/training/trainer.py` (remove trainer-level dropout call site in `train_epoch`)
  - `configs/base.yaml` (set `lr_encoders: 2.0e-6` + ISSUE-022 documentation note)

- **Change summary**:
  Removed trainer-level modality dropout execution so model-level `ModalityDropout` is the sole active dropout path. This eliminates accidental text dropout in training and removes dropout compounding. Also adjusted Phase 2 encoder LR conservatively to `2.0e-6` and documented missing standalone Phase 2 warmup behavior.

- **Diff snippet** (key changes only):
  ```python
  # src/training/trainer.py (train loop)
  # Removed:
  batch = self.apply_modality_dropout(batch)
  # Kept:
  # model.forward() still applies model-level self.modality_dropout(i_proj, m_proj)
  ```
  ```yaml
  # configs/base.yaml
  lr_encoders: 2.0e-6
  # NOTE (ISSUE-022...): separate Phase 2 warmup not implemented in active LR behavior
  ```

- **Risks accepted**:
  - Trainer-level raw augmentation is removed; regularization now relies on model-level dropout only.
  - Lower `lr_encoders` may slow Phase 2 adaptation, accepted as a stability tradeoff while Phase 2 warmup is deferred.

- **Verification**:
  - `rg -n "apply_modality_dropout\\(" src/training/trainer.py src -g "*.py"` -> definition only, no call sites
  - AST static check confirms zero `self.apply_modality_dropout(...)` call nodes
  - `python -c "import yaml; ..."` confirms `lr_encoders in config: 2e-06` and in safe range
  - `python -c "from src.training import trainer; from src.models.full_model import MultimodalMisinfoDetector"` imports OK

- **Commits**:
  - `d8a6ad4` — `fix(ISSUE-023 B): remove trainer-level apply_modality_dropout call from train loop`
  - `0cea923` — `config(lr_encoders): adjust Phase 2 encoder LR to 2.0e-6`

- **Rollback plan**:
  - `git revert d8a6ad4`
  - `git revert 0cea923`

### FIX-SESSION_05 · ISSUE-021 Options B + D · 2026-05-19

- **Resolves**: ISSUE-021
- **Component**: losses + model/config
- **Files changed**:
  - `src/losses/contrastive.py` (Option B)
  - `src/models/full_model.py` (Option D, Case 2)
  - `configs/base.yaml` (Option D config keys)

- **Change summary**:
  Implemented a two-part mitigation for contrastive-mask leakage:
  1. Added joint norm-based validity masking in `InfoNCELoss.forward()` so zeroed text/image embeddings are excluded even if upstream mask is incomplete.
  2. Disabled text modality dropout specifically while preserving image/metadata dropout via split config keys.

- **Diff snippet** (key changes only):
  ```python
  # Option B: src/losses/contrastive.py
  _norm_mask = (text_emb.norm(dim=-1) > 1e-3) & (image_emb.norm(dim=-1) > 1e-3)
  valid_mask = valid_mask & _norm_mask if valid_mask is not None else _norm_mask

  # Option D: src/models/full_model.py + configs/base.yaml
  text_modality_dropout_p = cfg_training.get("text_modality_dropout_p", 0.0)
  image_modality_dropout_p = cfg_training.get("image_modality_dropout_p", modality_dropout_p)
  self.modality_dropout = ModalityDropout(p=image_modality_dropout_p)
  t_valid = torch.ones(t_proj.shape[0], dtype=torch.bool, device=t_proj.device)  # text dropout off
  (i_proj, m_proj), (i_valid, m_valid) = self.modality_dropout(i_proj, m_proj)
  ```

- **Risks accepted**:
  - Fewer valid contrastive pairs per batch when zeroed embeddings are excluded (intended).
  - Minor reduction in text-path regularization from disabling text dropout (accepted for stability).

- **Verification**:
  - Option B CPU synthetic forward: finite losses with zeroed text embeddings; no NaN.
  - Option D config check: `training.text_modality_dropout_p == 0.0`, `training.image_modality_dropout_p == 0.15`.
  - Combined check: InfoNCE remains finite with partial zeroed text embeddings.

- **Commits**:
  - `c944daa` — `fix(ISSUE-021 B): add norm-based joint valid_mask in InfoNCELoss`
  - `b0f7119` — `fix(ISSUE-021 D): disable text modality dropout`

- **Rollback plan**:
  - `git revert c944daa`
  - `git revert b0f7119`
  
### FIX-SESSION_04 Â· Restore fusion stabilization trio (ISSUE-020) Â· 2026-05-19

- **Resolves**: ISSUE-020
- **Component**: model / fusion
- **Files changed**:
  - `src/models/cross_attention.py`

- **Change summary**:
  Restored the missing fusion stabilization trio in active `DualCrossAttention`: strong per-arm residual with LayerNorm, per-dimension learnable gates initialized to `-2.0`, and output projection weight scaling (`Ã—0.1`) at init.

- **Diff snippet** (key change only):
  ```python
  # Added in __init__
  self.norm_text = nn.LayerNorm(embed_dim)
  self.norm_image = nn.LayerNorm(embed_dim)
  self.gate_text = nn.Parameter(torch.full((embed_dim,), -2.0))
  self.gate_image = nn.Parameter(torch.full((embed_dim,), -2.0))
  self.attn_text_to_image.out_proj.weight.data *= 0.1
  self.attn_image_to_text.out_proj.weight.data *= 0.1

  # Added in forward
  t_prime = self.norm_text(t_proj + torch.sigmoid(self.gate_text) * t_attn)
  i_prime = self.norm_image(i_proj + torch.sigmoid(self.gate_image) * i_attn)
  ```

- **Risks accepted**:
  - Fusion behavior and training dynamics will change on next retrain (intended and required for architecture parity with Drift Log claims).

- **Verification**:
  ```bash
  python verify_issue020.py
  ```
  Result: all 6 checks passed (shape invariance, gate init values, residual behavior, output-proj scaling, LayerNorm presence, import sanity).

- **Side effects observed**:
  - Added verification artifact `verify_issue020.py` in repo root (not part of model runtime path).

- **Rollback plan**:
  `git revert 9c27c8c`

### FIX-SESSION_02 Â· Phase 1 closeout (ISSUE-015 + ISSUE-012) Â· 2026-05-19

- **Resolves**: ISSUE-015, ISSUE-012
- **Component**: data + metadata-dim cascade
- **Files changed**:
  - `src/data/preprocessing.py`
  - `scripts/prepare_data.py`
  - `src/data/feature_engineering.py`
  - `configs/base.yaml`
  - `src/models/metadata_encoder.py`
  - `scripts/test_ablation_mode.py`
  - `scripts/test_metadata_scaling.py`

- **Change summary**:
  Fixed scaling pipeline by making canonical preprocessing writes preserve float precision, applying scaler to train/val/test, and saving scaler with pickle binary mode. Dropped `language_location_mismatch` from active metadata features and cascaded metadata dimensionality from 17 to 16.

- **Risks accepted**:
  - Existing 17-dim checkpoints are incompatible with new 16-dim metadata input.
  - Train/val/test split artifacts regenerated (new canonical outputs).

- **Verification**:
  - `python scripts/prepare_data.py --config configs/base.yaml`
  - Train scaling check: `ads_per_page median=0.000, IQR=1.000`
  - Config check: `metadata_input_dim=16`, `len(metadata_features)=16`
  - Scaler check: `pickle.load(data/processed/metadata_scaler.pkl)` succeeds, `feature_names_in_` length = 16
  - Forward check: `MetadataEncoder()(torch.randn(8,16))` â†’ `(8, 256)`

- **Side effects observed**:
  - Windows console logging shows UnicodeEncodeError for some symbol characters in log messages, but preprocessing run completed successfully.

- **Rollback plan**:
  - `git revert caf7a4b`
  - `git revert 13cb755`

### FIX-SESSION_01 Â· Phase 1 data layer remediation Â· 2026-05-18

- **Resolves**: ISSUE-009, ISSUE-013, ISSUE-014, ISSUE-016
- **Partially addresses**: ISSUE-012 (column fixed; still zero due to location format)
- **Verifies**: ISSUE-011
- **Component**: data
- **Files changed**:
  - `src/data/preprocessing.py`
  - `src/data/feature_engineering.py`

- **Change summary**:
  Enforced `id/page_id` dtype + dedup; added list-like text parsing; precomputed `url_count` before `clean_text`;
  corrected language column reference; added robust parsing for `publisher_platforms` list strings.

- **Diff snippet** (key changes only):
  ```python
  # dtype + dedup
  df_raw = pd.read_csv(raw_csv, dtype={"id": str, "page_id": str})
  df_raw = df_raw.drop_duplicates(subset="id", keep="first").reset_index(drop=True)

  # precompute url_count on raw text
  df_raw["url_count"] = compute_url_count(df_raw)

  # list-like parsing in clean_text
  items = ast.literal_eval(stripped)
  text = " ".join(str(x) for x in items)

  # language column fix
  langs = row["languages"]

  # FB_only_flag parsing
  platforms = ast.literal_eval(s)
  ```

- **Risks accepted**:
  - 6 duplicate ID rows dropped
  - Splits regen may change distribution
  - Text length distribution shifts
  - Pre-fix reported F1 invalidated

- **Verification**:
  Inline terminal assertions (FIX_SESSION_01 verification) â€” all passed.

- **Side effects observed**:
  `language_location_mismatch` still zero (likely target_locations format). Scaling still unscaled after regen (ISSUEâ€‘015 persists).

- **Rollback plan**:
  `git reset --hard 61a50dfb80dbc64c163f1e5fbb16f93cce2d3d62`

<!-- Template:

### FIX-NNN Â· [Short title] Â· YYYY-MM-DD

- **Resolves**: ISSUE-NNN
- **Component**: data / model / training / eval / ablation / repro
- **Files changed**:
  - `path/to/file.py` (lines XXâ€“YY)
  - `path/to/config.yaml`

- **Change summary**:
  1-3 sentences describing what was changed, in plain language.

- **Diff snippet** (key change only, not full diff):
  ```python
  # Before
  old_line_or_block

  # After
  new_line_or_block
  ```

- **Risks accepted**:
  - [List the risks from OPEN_ISSUES.md that were knowingly accepted]
  - [If a risk was mitigated, describe how]

- **Verification**:
  How you confirmed the fix works (test command, metric change, manual inspection).
  ```bash
  # verification command
  ```
  Result: [what was observed]

- **Side effects observed**:
  Anything that changed beyond intent (e.g., metrics dropped, training time changed).
  Leave blank if none.

- **Rollback plan**:
  How to undo if needed (git commit hash, or "revert FIX-NNN").

---
-->

_(empty â€” will be populated as fixes are applied)_

---

## Summary Table (auto-update after each fix)

| Fix # | Date | Component | Severity resolved | Risks accepted |
|---|---|---|---|---|
| FIX-SESSION_05 | 2026-05-19 | losses + model/config | HIGH | fewer InfoNCE pairs, reduced text-dropout regularization |
| FIX-SESSION_04 | 2026-05-19 | model / fusion | HIGH | intended fusion-dynamics change on retrain |
| FIX-SESSION_02 | 2026-05-19 | data + config/model cascade | CRITICAL + HIGH | checkpoint incompatibility, splits regen |
| FIX-SESSION_01 | 2026-05-18 | data | HIGH | dup drop, split regen, text shift |
