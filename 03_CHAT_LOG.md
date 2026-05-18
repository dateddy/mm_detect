# 03 · CHAT LOG — Key Decision Snapshots

> Manually-curated snapshots of important Claude ↔ User conversation moments.
> Not full transcript — only decision-shaping exchanges.
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

### SESSION 2026-05-19 · FIX_SESSION_02 Phase 1 closeout

**Context entering session**: ISSUE-015 remained CRITICAL (artifacts unscaled) and ISSUE-012 remained OPEN (feature always zero).

**Decisions made**:
- D1: A.0 confirmed canonical split writer is `src/data/preprocessing.py::run_preprocessing_pipeline`; `prepare_data.py main()` is orchestration-only.
- D2: Apply scaler to train/val/test and remove legacy float rounding path (`float_format='%.0f'`).
- D3: Persist scaler using pickle binary mode for compatibility with existing `pickle.load` diagnostics.
- D4: Accept ISSUE-012 Option C: deprecate and drop `language_location_mismatch`, cascade metadata dimensions 17 → 16.

**Linked artifacts**:
- FIX-SESSION_02 logged in `02_FIXES.md`
- ISSUE-015 status: RESOLVED
- ISSUE-012 status: RESOLVED
- Updated context in `00_CONTEXT.md`

---

<!-- Template:

### SESSION YYYY-MM-DD · [Topic]

**Context entering session**: What we already knew

**Key exchanges**:
> **Q (me)**: ...
> **A (Claude)**: ...

**Decisions made**:
- D1: ...
- D2: ...

**Linked artifacts**:
- CHECK_NN updated
- ISSUE-NNN created
- FIX-NNN applied

---
-->

### SESSION 2026-05-16 · Audit framework setup

**Context entering session**: Project drifted after coding agent applied fixes without risk control. Codebase state uncertain.

**Key decisions**:
- D1: Adopt Meta-Cycle harness pattern, but inverted — Claude = controller, User = executor
- D2: 14-check route across 5 phases, starting with State of Truth (read-only)
- D3: Strict risk acceptance discipline — no fix without logged risks
- D4: 4 persistent memory files: CONTEXT, OPEN_ISSUES, FIXES, CHAT_LOG

**Linked artifacts**:
- All 6 setup files created (00–04 + CHECK_01)

---
