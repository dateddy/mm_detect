# Multimodal Misinformation Detection on Vietnamese Facebook Ads

> A tri-modal deep learning system that jointly encodes **text** (PhoBERT), **images** (ViT-B/16), and **behavioral metadata** to detect misleading advertisements on Vietnamese Facebook.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Training](#training)
- [Evaluation & Ablation](#evaluation--ablation)
- [File Structure](#file-structure)
- [Results](#results)

---

## Overview

Existing misinformation detection systems treat the problem as a unimodal text classification task, ignoring the rich contextual signals available in ad creative images and the behavioral metadata of the distributing accounts. This repository addresses that gap.

**Three input streams:**

| Modality | Encoder | Output dim | Signal type |
|---|---|---|---|
| Ad caption text (Vietnamese) | PhoBERT (`vinai/phobert-base-v2`) | 768 | Linguistic |
| Ad creative image | ViT-B/16 (ImageNet-21k) | 768 | Visual |
| 9 engineered behavioral features | Custom MLP | 256 | Behavioral |

**Key architectural contributions:**

- **Dual bidirectional cross-attention** — text attends to image+metadata *and* image attends to text+metadata simultaneously
- **Nonlinear gated fusion** — input-conditioned per-modality weighting learned dynamically per sample
- **InfoNCE contrastive auxiliary loss** — aligns text and image embedding spaces before classification
- **Modality dropout** (p=0.15) — ensures graceful degradation when any modality is missing at inference

**Output:** Binary label — `1 = misleading`, `0 = not misleading`

---

## Architecture

```
Text (Vietnamese)         Image (224x224)         Metadata (9 features)
       │                        │                         │
  PhoBERT                   ViT-B/16                 MLP Encoder
  [CLS] 768d                [CLS] 768d               9→256→256d
       │                        │                         │
  Linear+LN               Linear+LN                Linear+LN
    768→256                  768→256                  256→256
       │                        │                         │
       └──────── Modality Dropout (p=0.15, train only) ───┘
                                │
              ┌─────────────────┼─────────────────┐
              │                                   │
    Cross-Attention A                   Cross-Attention B
    Q=Text, K/V=[Image,Meta]            Q=Image, K/V=[Text,Meta]
    → T'  (text+visual context)         → I'  (image+text context)
              │                                   │
              └─────────────────┬─────────────────┘
                                │
                    Concat [T', I', Meta]  →  768d
                                │
                    Gate = sigmoid(W·concat)
                    g1, g2, g3 = split(Gate)
                    Fused = g1·T' + g2·I' + g3·Meta
                                │
                    Residual + LayerNorm
                    out = LN(Fused + T + I + Meta)
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
         Classification Head          Contrastive Loss (aux)
         256→128→64→1                 InfoNCE(T_proj, I_proj)
         Dropout(0.3)                 λ = 0.1
                 │
             sigmoid
                 │
       Misleading / Not misleading
```

### Selected Metadata Features

Features were selected via Pearson correlation analysis against the misinformation label. Nine features are retained:

| Feature | Derived from | Correlation insight |
|---|---|---|
| `ads_per_page` | `COUNT(id) GROUP BY page_id` | High-volume pages correlate with coordinated inauthentic behavior |
| `platform_count` | `len(publisher_platforms)` | Multi-platform reach differs between organic and misinfo content |
| `FB_only_flag` | `1 if platforms == ['facebook']` | r = −0.771 with platform_count; misinfo often stays on a single platform |
| `all_targeted` | `1 if target_gender == 'all'` | Broad demographic targeting correlates with misinfo distribution |
| `burstiness` | `(max_daily − mean_daily) / std_daily` | Irregular publishing cadence signals coordinated campaigns |
| `avg_ad_duration` | `MEAN(stop − start) per page` | Short campaigns across many countries are a known misinfo pattern |
| `launch_delay` | `start_time − creation_time (hours)` | Near-instant scheduling may indicate automated ad deployment |
| `num_countries` | `len(unique countries in target_locations)` | Geographic spread; misinfo campaigns often target multiple countries |
| `language_location_mismatch` | Language vs. target country language | Deceptive cross-lingual targeting flag |

---

## Dataset

### Raw data structure

```
data/
├── raw/
│   ├── ads.csv          # one row per ad, 18 columns
│   └── images/
│       ├── id1.png      # ad creative for ad with id = id1
│       ├── id2.png
│       └── ...
```

### CSV schema

| Column | Type | Description |
|---|---|---|
| `id` | string | Unique ad identifier — maps directly to `images/{id}.png` |
| `page_id` | string | Facebook Page that published the ad |
| `page_name` | string | Human-readable page name |
| `ad_creation_time` | datetime | When the ad was created in Ads Manager |
| `ad_delivery_start_time` | datetime | When the ad began serving |
| `ad_delivery_stop_time` | datetime | When the ad stopped serving (null if active) |
| `ad_creative_bodies` | string | Primary Vietnamese text body of the ad |
| `ad_creative_link_titles` | string | Headline shown in link preview |
| `currency` | string | Spend currency code (e.g. `VND`) |
| `impressions` | string | Meta-reported impressions range (e.g. `1000-5000`) |
| `spend` | string | Meta-reported spend range (e.g. `100-499`) |
| `target_gender` | string | Audience gender setting (`male` / `female` / `all`) |
| `target_ages` | string | Age range(s) targeted |
| `target_locations` | string | Geographic locations targeted |
| `languages` | string | Language(s) targeted |
| `publisher_platforms` | string | Platforms where ad ran (`facebook`, `instagram`, etc.) |
| `ad_snapshot_url` | string | URL to Meta Ad Library snapshot |
| `misinformation` | int | **Label**: `1 = misleading`, `0 = not misleading` |

### Splits

Partitioned at **page level** (not ad level) to prevent data leakage — ads from the same page share behavioral metadata.

| Split | Proportion | Strategy |
|---|---|---|
| Train | 70% | Stratified by label at page level |
| Validation | 15% | Stratified by label at page level |
| Test | 15% | Held out — never used for any tuning decision |

---

## Installation

```bash
git clone https://github.com/your-username/multimodal-misinfo-detection.git
cd multimodal-misinfo-detection

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**Key dependencies:**

```
torch>=2.1.0
transformers>=4.38.0
timm>=0.9.12
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
Pillow>=10.0.0
tqdm>=4.65.0
pyyaml>=6.0
```

**GPU requirements:** Minimum 16GB VRAM for batch size 16 with full encoder fine-tuning. Tested on NVIDIA A100 40GB.

---

## Quickstart

### 1. Prepare data

```bash
# Feature engineering, train/val/test splits, scaler fitting
python scripts/prepare_data.py \
  --csv data/raw/ads.csv \
  --images data/raw/images/ \
  --output data/processed/
```

### 2. (Optional) Extract embeddings offline

Skip if you plan to fine-tune encoders end-to-end. Offline extraction is faster for initial fusion experiments.

```bash
python scripts/extract_embeddings.py \
  --split train val test \
  --data data/processed/ \
  --output data/embeddings/
```

### 3. Train

```bash
python scripts/train.py --config configs/base.yaml
```

### 4. Evaluate on test set

```bash
python scripts/evaluate.py \
  --checkpoint outputs/checkpoints/best_model.pt \
  --data data/processed/test.csv
```

### 5. Inference on new samples

```bash
python scripts/predict.py \
  --checkpoint outputs/checkpoints/best_model.pt \
  --csv path/to/new_ads.csv \
  --images path/to/images/
```

---

## Training

### Configuration

All hyperparameters are defined in `configs/base.yaml`:

```yaml
# Encoders
text_encoder: vinai/phobert-base-v2
image_encoder: vit_base_patch16_224
image_size: 224
max_text_len: 256

# Projection
proj_dim: 256

# Cross-attention
num_heads: 8
attn_dropout: 0.1

# Modality dropout
modality_dropout_p: 0.15

# Classification head
head_dropout: 0.3

# Optimizer
optimizer: adamw
lr_fusion: 3.0e-4
lr_encoders: 1.0e-5
weight_decay: 0.01
warmup_steps: 500
lr_schedule: cosine

# Training
batch_size: 32
max_epochs: 30
early_stopping_patience: 5
early_stopping_metric: macro_f1
gradient_clip: 1.0
mixed_precision: true

# Loss
contrastive_lambda: 0.1
contrastive_temperature: 0.07

# Two-phase protocol
freeze_encoders_epochs: 3
unfreeze_top_k_blocks: 4
```

### Two-phase training protocol

**Phase 1 — Fusion warmup (epochs 1–3)**
PhoBERT and ViT are fully frozen. Only the projection layers, metadata MLP, cross-attention modules, gating network, and classification head are trained at `lr=3e-4`. This stabilizes the randomly-initialized fusion components before gradients flow into the pre-trained encoders.

**Phase 2 — End-to-end fine-tuning (epochs 4–30)**
The top-4 transformer blocks of PhoBERT and ViT are unfrozen at `lr=1e-5` (10× lower than fusion components). Early stopping monitors validation macro-F1 with patience=5.

### Resuming from checkpoint

```bash
python scripts/train.py \
  --config configs/base.yaml \
  --resume outputs/checkpoints/epoch_10.pt
```

---

## Evaluation & Ablation

### Metrics

| Metric | Notes |
|---|---|
| Accuracy | Overall correctness |
| Precision | Fraction of predicted positives that are truly misleading |
| Recall | Fraction of actual misleading ads that are detected |
| F1 (macro) | Primary metric for model selection and early stopping |
| AUC-ROC | Threshold-independent discrimination |
| AUC-PR | Preferred over ROC for imbalanced datasets |

### Run all ablations

```bash
python scripts/run_ablations.py --output outputs/results/
```

This runs five ablation groups sequentially:

**Modality ablation** — establishes the contribution of each input stream

| Variant | Text | Image | Metadata |
|---|---|---|---|
| Text-only | ✓ | — | — |
| Image-only | — | ✓ | — |
| Metadata-only | — | — | ✓ |
| Text + Image | ✓ | ✓ | — |
| Text + Metadata | ✓ | — | ✓ |
| Image + Metadata | — | ✓ | ✓ |
| **Full model** | **✓** | **✓** | **✓** |

**Fusion mechanism ablation** — quantifies the benefit of dual cross-attention

| Variant | Method |
|---|---|
| Concat + MLP | Concatenate [T,I,M], pass to MLP — no cross-modal interaction |
| Simple average | Element-wise mean of T, I, M projections |
| Single cross-attn (baseline) | Q=T, K/V=[I,M] only — unidirectional |
| Dual cross-attn | Parallel Q=T and Q=I branches |
| Dual cross-attn + gating | + nonlinear input-conditioned gate |
| **Full model** | + residual + contrastive loss |

**Metadata feature ablation** — validates the correlation-guided feature selection

| Variant | Features |
|---|---|
| Top-3 only | `ads_per_page`, `platform_count`, `all_targeted` |
| Temporal only | `burstiness`, `launch_delay`, `avg_ad_duration` |
| Targeting only | `all_targeted`, `FB_only_flag`, `num_countries`, `language_location_mismatch` |
| **Selected-9 (proposed)** | All 9 selected features |
| All features (40+) | Full engineered feature set (upper bound) |

**Contrastive loss weight ablation**

| Variant | λ |
|---|---|
| No contrastive loss | 0 |
| Weak | 0.05 |
| **Proposed** | **0.1** |
| Strong | 0.5 |

**Encoder fine-tuning ablation**

| Variant | Strategy |
|---|---|
| Frozen (offline extraction) | Encoders frozen throughout |
| **Partial fine-tune (proposed)** | **Top-4 blocks unfrozen after epoch 3** |
| Full fine-tune from epoch 1 | All blocks unfrozen immediately |
| Full fine-tune after warmup | All blocks unfrozen after epoch 3 |

### Visualize attention weights

```bash
python src/evaluation/attention_viz.py \
  --checkpoint outputs/checkpoints/best_model.pt \
  --sample_id id1 \
  --output outputs/figures/
```

---

## File Structure

```
project/
│
├── data/
│   ├── raw/
│   │   ├── ads.csv                     # raw CSV, 18 columns
│   │   └── images/                     # ad creative images
│   │       ├── id1.png
│   │       └── ...
│   ├── processed/
│   │   ├── train.csv                   # page-level split
│   │   ├── val.csv
│   │   ├── test.csv
│   │   └── metadata_scaler.pkl         # fitted RobustScaler
│   └── embeddings/                     # optional offline embeddings
│       ├── phobert_train.npy           # (N, 768) float32
│       ├── vit_train.npy               # (N, 768) float32
│       └── ...
│
├── src/
│   ├── data/
│   │   ├── dataset.py                  # AdDataset — loads text + image + metadata
│   │   ├── collate.py                  # custom collate_fn
│   │   ├── feature_engineering.py      # derives 40+ features from raw CSV
│   │   └── preprocessing.py           # scaler fitting, image transforms
│   ├── models/
│   │   ├── text_encoder.py             # PhoBERT wrapper, [CLS] extraction
│   │   ├── image_encoder.py            # ViT-B/16 wrapper, [CLS] extraction
│   │   ├── metadata_encoder.py         # 3-layer MLP, BatchNorm + GELU
│   │   ├── projection.py              # modality-specific Linear + LayerNorm
│   │   ├── cross_attention.py          # multi-head cross-attention module
│   │   ├── gated_fusion.py             # nonlinear gate + weighted sum + residual
│   │   ├── classification_head.py      # 256→128→64→1 MLP + Dropout
│   │   └── full_model.py              # composes all components; full forward pass
│   ├── losses/
│   │   ├── contrastive.py              # InfoNCE loss
│   │   └── combined_loss.py           # L = L_cls + λ·L_con
│   ├── training/
│   │   ├── trainer.py                  # training loop, two-phase protocol
│   │   ├── early_stopping.py           # macro-F1 based early stopping
│   │   └── scheduler.py               # warmup + cosine decay
│   ├── evaluation/
│   │   ├── metrics.py                  # accuracy, F1, AUC-ROC, AUC-PR
│   │   ├── ablation.py                 # ablation variant runner
│   │   └── attention_viz.py           # cross-attention weight visualization
│   └── utils/
│       ├── seed.py                     # set_seed() for reproducibility
│       ├── logger.py                   # structured logging
│       └── checkpoint.py              # save/load checkpoints
│
├── configs/
│   ├── base.yaml                       # default hyperparameters
│   ├── ablation_modality.yaml
│   ├── ablation_fusion.yaml
│   ├── ablation_metadata.yaml
│   ├── ablation_loss.yaml
│   └── ablation_finetune.yaml
│
├── scripts/
│   ├── prepare_data.py                 # feature engineering + splits + scaling
│   ├── extract_embeddings.py           # offline embedding extraction
│   ├── train.py                        # main training entry point
│   ├── evaluate.py                     # evaluate checkpoint on test set
│   ├── run_ablations.py                # runs all ablation experiments
│   └── predict.py                      # inference on new samples
│
├── notebooks/
│   ├── 01_eda.ipynb                    # exploratory data analysis
│   ├── 02_feature_selection.ipynb      # correlation analysis + feature selection
│   ├── 03_results_analysis.ipynb       # ablation results + plots
│   └── 04_error_analysis.ipynb         # false positive / negative deep dives
│
├── outputs/
│   ├── checkpoints/
│   │   ├── best_model.pt               # best validation F1 checkpoint
│   │   └── ablation_*/                 # one checkpoint per ablation variant
│   ├── logs/                           # training logs (JSON Lines)
│   ├── results/                        # metric tables (CSV)
│   └── figures/                        # attention visualizations
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Results

> Fill in after running experiments. Template below.

### Main results (test set)

| Model | Accuracy | Precision | Recall | F1 (macro) | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|
| Text-only (PhoBERT) | — | — | — | — | — | — |
| Image-only (ViT) | — | — | — | — | — | — |
| Metadata-only (MLP) | — | — | — | — | — | — |
| Text + Image | — | — | — | — | — | — |
| Single cross-attn (baseline) | — | — | — | — | — | — |
| **Full model (proposed)** | — | — | — | — | — | — |

### Ablation summary

| Component removed | F1 drop | Conclusion |
|---|---|---|
| Dual → single cross-attention | — | Bidirectionality matters because... |
| No contrastive loss (λ=0) | — | Contrastive alignment helps because... |
| No modality dropout | — | Dropout effect on robustness... |
| No metadata | — | Metadata contribution... |
| No gating (simple average) | — | Input-conditioned gating helps because... |

