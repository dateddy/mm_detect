# Multimodal Misinformation Detection - Vietnamese Ads

A comprehensive deep learning framework for detecting misinformation in Vietnamese Facebook advertisements using multimodal learning (text, image, and metadata).

## Project Overview

This project combines:
- **Text Processing**: PhoBERT for Vietnamese text understanding
- **Vision**: Vision Transformer (ViT) for image analysis
- **Metadata**: Numerical features (impressions, spend, targeting)
- **Fusion**: Cross-attention and gated mechanisms for multimodal integration

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place the following in `data/raw/`:
- `ads_vietnam_clean.csv` - Main dataset
- `ad_images/` - Image directory with `{id}.png` files

### 3. Run Preprocessing

```bash
python scripts/preprocess.py --config configs/training/default.yaml
```

### 4. Train Model

```bash
python scripts/train.py \
    --model-config configs/model/multimodal.yaml \
    --training-config configs/training/default.yaml
```

### 5. Evaluate

```bash
python scripts/evaluate.py --checkpoint experiments/checkpoints/best.pt
```

### 6. Run Ablation Study

```bash
python scripts/run_ablation.py
```

## Project Structure

```
mm_detect/
├── configs/              # ALL experiment configurations (YAML)
│   ├── base.yaml
│   ├── model/            # Model architecture configs
│   ├── training/         # Training hyperparameters
│   └── ablation/         # Ablation study configs
├── data/                 # Dataset (raw, processed, embeddings)
├── notebooks/            # Jupyter notebooks for analysis
├── src/                  # Core Python modules
│   ├── data/             # Data loading and preprocessing
│   ├── models/           # Model architectures
│   ├── training/         # Training logic
│   ├── evaluation/       # Metrics and evaluation
│   ├── utils/            # Helper functions
│   └── ablation/         # Ablation study runners
├── scripts/              # Entry points (CLI)
├── experiments/          # Experiment outputs (auto-generated)
└── outputs/              # Final results and reports
```

## Key Components

### Models
- **Text Encoder**: PhoBERT-base (768-dim)
- **Image Encoder**: ViT-base (768-dim)
- **Metadata Encoder**: MLP with embeddings
- **Fusion**: Cross-attention + Gated combination

### Training
- **Optimizer**: Adam
- **Learning Rate**: 1e-3
- **Batch Size**: 32-64
- **Loss**: BCEWithLogits or Focal Loss
- **Epochs**: 20-30

### Metrics
- Accuracy, Precision, Recall
- F1-Score (primary metric)
- ROC-AUC

## Ablation Study

Compare model components:
- Text only
- Image only
- Metadata only
- Text + Image
- Text + Metadata
- Full model (all modalities)
- No cross-attention
- No gating fusion
- No modality dropout

## Reproducibility

All experiments are configured via YAML files in `configs/`. Random seeds are set for full reproducibility:
- `seed: 42`
- `deterministic: true`

## Performance Benchmarks

(To be updated after training)

| Configuration | F1-Score | Accuracy | ROC-AUC |
|---------------|----------|----------|---------|
| Text Only     |   -      |    -     |   -     |
| Image Only    |   -      |    -     |   -     |
| Metadata Only |   -      |    -     |   -     |
| Full Model    |   -      |    -     |   -     |

## Contributing

1. Keep configs in YAML - easy to version and compare
2. Use modular design - components are reusable
3. Log experiments with W&B or TensorBoard
4. Document findings in ablation results

## References

- PhoBERT: https://github.com/VinAIResearch/PhoBERT
- ViT: https://github.com/google-research/vision_transformer
- Multimodal Learning: https://arxiv.org/abs/2301.04612

## License

[Specify your license here]

## Contact

[Your contact information]
