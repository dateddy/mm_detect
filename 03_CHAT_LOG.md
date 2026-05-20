# 03 - CHAT LOG - Key Decision Snapshots

> Manually-curated snapshots of important Claude <-> User conversation moments.
> Not full transcript - only decision-shaping exchanges.
> Update at end of each working session.

---

## How to Use

After each session with Claude, paste relevant snippets here. Focus on:
- Why a particular check approach was chosen
- Trade-off discussions ("we picked X over Y because...")
- Risk acceptance reasoning
- Strategic pivots (scope reductions, priority changes)

Skip:
- Routine prompt/report exchanges (those live in CHECK_NN files)
- Generic code outputs

---

## Session Entries (newest at top)

### SESSION 2026-05-20 - FIX_SESSION_06 (ISSUE-023 closeout + lr_encoders tune)

**Context entering session**: CHECK_11 aborted with ISSUE-023 after finding two-level dropout (trainer raw dropout + model embedding dropout), which violated FIX_SESSION_05 intent to keep text dropout disabled.

**Decisions made**:
- D1: Remove only the trainer-loop call site `self.apply_modality_dropout(batch)` and keep the method definition intact for compatibility/traceability.
- D2: Keep model-level `ModalityDropout` as the sole active mechanism (`full_model.py`), preserving text off (`0.0`) and image/meta dropout (`0.15`).
- D3: Keep ISSUE-022 deferred (no scheduler rewrite in this session), but document it in `base.yaml` and reduce `lr_encoders` conservatively to `2.0e-6`.
- D4: Re-run C11 evidence checks immediately after patch and close ISSUE-023.

**Linked artifacts**:
- Commit `d8a6ad4` (remove trainer-level dropout call site)
- Commit `0cea923` (`lr_encoders` change + ISSUE-022 note)
- `CHECK_11_report.md` remains as prior abort record; rerun evidence captured in FIX_SESSION_06 verification + issue/fix logs

---

### SESSION 2026-05-19 - FIX_SESSION_03 + CHECK_08 rerun

**Context entering session**: CHECK_08 had previously aborted on ISSUE-019 (K/V self-reference regression in active cross-attention).

**Decisions made**:
- D1: Apply narrowly scoped fix in `src/models/cross_attention.py` only, restoring self-inclusive `K/V=[t,i,m]` for both text-query and image-query branches.
- D2: Re-run CHECK_08 end-to-end after patch instead of continuing with stale aborted findings.
- D3: Mark ISSUE-019 resolved and open ISSUE-020 for the remaining missing fusion stabilization claims (strong residual, per-dim gate init, output-proj x0.1 init) that are not present in active code.

**Linked artifacts**:
- Commit `bd5f4ab` for ISSUE-019 fix
- `CHECK_08_report.md` replaced with post-fix full report
- `00_context.md`, `01_OPEN_ISSUES.md`, `04_AUDIT_ROUTE.md` updated

---

### SESSION 2026-05-19 - FIX_SESSION_02 Phase 1 closeout

**Context entering session**: ISSUE-015 remained CRITICAL (artifacts unscaled) and ISSUE-012 remained OPEN (feature always zero).

**Decisions made**:
- D1: A.0 confirmed canonical split writer is `src/data/preprocessing.py::run_preprocessing_pipeline`; `prepare_data.py main()` is orchestration-only.
- D2: Apply scaler to train/val/test and remove legacy float rounding path (`float_format='%.0f'`).
- D3: Persist scaler using pickle binary mode for compatibility with existing `pickle.load` diagnostics.
- D4: Accept ISSUE-012 Option C: deprecate and drop `language_location_mismatch`, cascade metadata dimensions 17 -> 16.

**Linked artifacts**:
- FIX-SESSION_02 logged in `02_FIXES.md`
- ISSUE-015 status: RESOLVED
- ISSUE-012 status: RESOLVED
- Updated context in `00_CONTEXT.md`

---

### SESSION 2026-05-16 - Audit framework setup

**Context entering session**: Project drifted after coding agent applied fixes without risk control. Codebase state uncertain.

**Key decisions**:
- D1: Adopt Meta-Cycle harness pattern, but inverted - Claude = controller, User = executor
- D2: 14-check route across 5 phases, starting with State of Truth (read-only)
- D3: Strict risk acceptance discipline - no fix without logged risks
- D4: 4 persistent memory files: CONTEXT, OPEN_ISSUES, FIXES, CHAT_LOG

**Linked artifacts**:
- All 6 setup files created (00-04 + CHECK_01)

---
