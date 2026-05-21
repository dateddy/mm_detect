# 04 Â· AUDIT ROUTE â€” TMD Project Recovery

> Sequence of 14 audit checks, ordered by dependency. Each check is its own cycle.
> Do NOT skip ahead â€” earlier findings inform later checks.

---

## Workflow Per Check

Má»—i check thá»±c hiá»‡n nhÆ° má»™t full 7-step harness cycle:

| Step | Owner | Action |
|---|---|---|
| 1. Prompt Assembly | Claude | Assemble check prompt tá»« CONTEXT.md + previous findings |
| 2. LLM Inference | Claude | Generate check prompt + risk register cho check Ä‘Ã³ |
| 3. Classify Output | Claude | Decide: send check, hoáº·c ask clarifying question |
| 4. Tool Execution | **YOU** | Run check trÃªn codebase, KHÃ”NG apply fix |
| 5. Result Packaging | YOU | Write report theo template trong check file |
| 6. Context Update | Both | Update CONTEXT.md + OPEN_ISSUES.md + FIXES.md |
| 7. Loop / Exit | Claude | Generate next check, hoáº·c declare audit complete |

**Critical rule**: Step 4 lÃ  READ-ONLY máº·c Ä‘á»‹nh. Fix chá»‰ apply sau khi Step 6 Ä‘Ã£ update OPEN_ISSUES.md vÃ  báº¡n Ä‘Ã£ accept risks.

---

## The Route (14 checks Â· 5 phases)

### Phase 0 â€” State of Truth (foundation, READ-ONLY)
- [x] **C01** â€” Repository inventory & active code path
- [x] **C02** â€” Reproduce single end-to-end run trace (no actual execution, just trace)

### Phase 1 â€” Data Layer (highest risk after coding agent edits)
- [x] **C03** â€” Raw data integrity & schema
- [x] **C04** â€” Preprocessing pipeline correctness
- [x] **C05** â€” Train/val/test split logic + leakage re-verify *(code-path verified; artifact-level CSV overlap/count checks pending in data-available environment)*
- [x] **C06** â€” Metadata feature computation deep-dive (17 present, 14 effective; 3 broken constants confirmed; split artifacts verified)

### Phase 2 â€” Model Architecture
- [x] **C07** â€” Encoder configurations (PhoBERT, ViT-B/16, MLP) + freezing phases
- [x] **C08** — Fusion layer (dual bidirectional cross-attention + gated fusion) *(rerun complete post-FIX_SESSION_03; ISSUE-019 and ISSUE-020 resolved via FIX_SESSION_03/04)*
- [x] **C09** â€” Loss formulation (main + InfoNCE auxiliary) *(completed; L2 + label smoothing + learnable temperature confirmed, valid_mask path is partial and tracked as ISSUE-021)*

### Phase 3 â€” Training Pipeline
- [x] **C10** â€” Optimizer, scheduler, two-phase transitions *(completed; AdamW two-phase param-group injection and lr_encoders=3e-6 confirmed, separate Phase-2 warmup behavior missing and tracked as ISSUE-022)*
- [x] **C11** â€” Modality dropout (p=0.15) implementation *(rerun after FIX_SESSION_06; trainer-level dropout call removed, model-level dropout is sole active path, ISSUE-023 resolved)*
- [ ] **C12** â€” Early stopping + checkpoint saving

### Phase 4 â€” Evaluation & Ablation
- [ ] **C13** â€” Metric computation + test set isolation
- [ ] **C14** â€” Ablation logic & component isolation

### Phase 5 â€” Reproducibility (cross-cutting, run last)
- [ ] **C15** â€” Seeds, deterministic ops, config versioning *(optional, time-permitting)*

---

## Exit Conditions

Audit complete khi:
- All 14 checks marked done in this file
- OPEN_ISSUES.md has zero items with `severity: critical`
- FIXES.md log matches all critical issues fixed
- CONTEXT.md reflects current state accurately

OR

- You hit time budget exhaustion â†’ mark which checks remain, ship project with caveats

---

## Files in This Memory System

| File | Purpose | Updated by |
|---|---|---|
| `00_CONTEXT.md` | Snapshot of project intent + current state | Both, every cycle |
| `01_OPEN_ISSUES.md` | Issues discovered, with severity + status | After each check |
| `02_FIXES.md` | Chronological log of changes made, with risks accepted | When fix applied |
| `03_CHAT_LOG.md` | Key decision snapshots from conversation | Manually, end of session |
| `04_AUDIT_ROUTE.md` | This file â€” overall plan + check checklist | Update checkbox after each |
| `CHECK_NN_*.md` | One file per check: prompt + report | Generated each cycle |

---

## Risk Discipline

Before ANY fix:
1. Review the risks listed in the check's Risk Register section
2. Decide explicitly: ACCEPT this risk / MITIGATE / ABORT fix
3. Log decision in FIXES.md with `risk_accepted: [list]`
4. ONLY THEN apply the fix

The exact failure mode of "applying coding agent fixes without risk control" must NOT repeat in this recovery.

