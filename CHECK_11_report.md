# CHECK_11 REPORT - Modality Dropout Implementation

## 0. Audit Status

**Initial run**: aborted at Step 4 stop condition (ISSUE-023 opened).

**Rerun status (post FIX_SESSION_06)**: resolved.

Trainer-level call sites to `apply_modality_dropout()` were removed from active loops in `src/training/trainer.py`, so model-level `ModalityDropout` is now the single active dropout path. This restores the intended behavior where text dropout remains disabled while image/metadata dropout stays active.

## 1. ModalityDropout Class

- Location: `src/models/projection.py::ModalityDropout`
- Forward signature: `forward(*embeddings) -> (masked_embeddings, validity_masks)`
- Dropout behavior: **INDEPENDENT per embedding argument**
- Dropped modality output: zeroed embedding vector
- Validity mask: returns one `(B,)` bool mask per input embedding, where `True = kept and not already zero`

Code evidence:

```python
for emb in embeddings:
    bernoulli = torch.bernoulli(
        torch.full((batch_size,), 1.0 - self.p, device=device)
    ).bool()
    already_zero = (emb.abs().sum(dim=-1) == 0).bool()
    keep = bernoulli & ~already_zero
    masked_emb = emb * keep.float().unsqueeze(-1)
```

Interpretation: when called as `self.modality_dropout(i_proj, m_proj)`, image and metadata each receive their own Bernoulli draw. They are not coupled.

## 2. Text Dropout (Post FIX_SESSION_05)

- Model-level `t_valid`: `torch.ones(B, dtype=bool)` - always true.
- Model-level `t_proj` passed to `self.modality_dropout`: **NO**.
- Model-level text zeroing path: **NONE found**.
- Full active training path text zeroing path: **YES, in trainer-level `apply_modality_dropout()`**.

Model code evidence:

```python
t_valid = torch.ones(t_proj.shape[0], dtype=torch.bool, device=t_proj.device)
(i_proj, m_proj), (i_valid, m_valid) = self.modality_dropout(i_proj, m_proj)
```

Trainer code evidence:

```python
if self._has_text and has_text_keys:
    droppable.append("text")
...
if modality == "text":
    batch["input_ids"][mask_i] = 0
    batch["attention_mask"][mask_i] = 0
    batch["attention_mask"][mask_i, 0] = 1
```

**Text dropout fully disabled:** **NO** in the full active training pipeline.

## 3. Image + Metadata Dropout

- Config `training.image_modality_dropout_p`: `0.15`
- Model-level image dropout probability: `0.15`
- Model-level metadata dropout probability: `0.15`
- Model-level independence: **INDEPENDENT**
- Empirical count test: not needed for verdict; code-level evidence shows a fresh Bernoulli draw inside the loop for each embedding.

Important nuance: trainer-level dropout can also zero raw images before the model. In full tri-modal mode, trainer dropout uses one random modality choice per sample after a `p=0.15` dropout decision, so raw image dropout is approximately `0.15 * 1/2 = 0.075` when text and image are both droppable. The model then applies embedding-level image dropout at `0.15`, giving an effective random image zeroing probability of roughly `1 - (1 - 0.075) * (1 - 0.15) = 0.214`, excluding structural missing images.

## 4. Two-Level Dropout

- Trainer-level dropout: **EXISTS and is active**
- Model-level dropout: **EXISTS and is active**
- Compound stacking risk: **YES**
- Stop condition hit: **YES** - Step 4 found a second independent dropout mechanism.

Trainer-level dropout is applied in `train_epoch()` before forward:

```python
batch = self.apply_modality_dropout(batch)
```

Model-level dropout is then applied inside `MultimodalMisinfoDetector.forward()`:

```python
(i_proj, m_proj), (i_valid, m_valid) = self.modality_dropout(i_proj, m_proj)
```

## 5. Missing Image vs Dropout Interaction

- Missing image handling: separate first, then unified by zero-detection in model-level `ModalityDropout`.
- Missing image zeroing happens after projection:

```python
i_proj[missing_image_mask] = 0.0
```

- Model-level `i_valid` represents: **both structural missing image and model-level image dropout**, because `already_zero` forces `keep=False`.
- Trainer-level `batch["valid_mask"]` represents: **raw trainer dropout only**, not structural missing image.
- Trainer later prefers `output["image_valid"]` over `batch["valid_mask"]`, so trainer-level text drops may not be reflected in the mask passed to the loss. FIX_SESSION_05 norm-based masking in `InfoNCELoss` mitigates this for contrastive loss, but it does not prevent the classifier path from seeing raw text-dropped samples.

## 6. Drift Log Reconciliation

| Claim | Expected | Found | Status |
|---|---|---|---|
| Modality dropout p=0.15 | image at 0.15 | model image/meta at 0.15, plus trainer raw dropout | PARTIAL |
| Text dropout: 0.0 (FIX_SESSION_05 D) | text off | model-level off, trainer-level still active | BROKEN |
| valid_mask joint (FIX_SESSION_05 B) | both text+image | InfoNCE norm mask mitigates zero embeddings | MATCHES for loss only |

## 7. New Findings

- **ISSUE-023 (HIGH, initial run)**: Trainer-level raw modality dropout remained active and could still drop text, creating a two-level dropout pipeline and contradicting FIX_SESSION_05 Option D at the full training-path level. This was resolved in FIX_SESSION_06.
- The model-level `ModalityDropout` now independently drops metadata at `p=0.15`; this is not inherently wrong, but it should be explicitly accepted because the config key is named `image_modality_dropout_p` while it controls both image and metadata in current code.

## 8. Updates to Memory Files

- `01_OPEN_ISSUES.md`: added ISSUE-023.
- `00_CONTEXT.md`: updated training behavior with C11 abort finding.
- `04_AUDIT_ROUTE.md`: C11 left unchecked and annotated as aborted pending ISSUE-023.

## 9. Time Spent

~35 minutes

## 10. Rerun Addendum (FIX_SESSION_06)

- `Trainer.apply_modality_dropout()` is now definition-only; no active call sites in training/validation loops.
- Text dropout is effectively disabled in the active path:
  - model-level text dropout remains off (`t_valid = ones`, text not passed into `self.modality_dropout`)
  - trainer-level raw text dropout no longer executes.
- Two-level dropout compounding is removed.
- Model-level image/metadata dropout remains active at configured probability (`image_modality_dropout_p = 0.15`) via `self.modality_dropout(i_proj, m_proj)`.
- ISSUE-023 is resolved; C11 can be marked complete.
