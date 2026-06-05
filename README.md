# Frequency-Aware Image Captioning

Project for the Deep Learning and Applied AI course (2025/2026)  
Sapienza University of Rome — Prof. Emanuele Rodolà

## Idea

Standard image captioning models process images through a single CNN encoder,
producing a global feature vector that loses spatial and structural information.
This project proposes a dual-branch architecture that decomposes the input image
into low-frequency (structural) and high-frequency (detail) components using FFT,
encodes each branch separately with a spatial CNN encoder, and fuses them before
passing to a Transformer decoder for caption generation.

We evaluate on two datasets: **Flickr8k** (semantic-focused) and **CLEVR**
(geometry-focused), to study whether frequency decomposition benefits
structure-aware domains.

## Architecture

```
Image → FFT decomposition
          ├── Low frequency  → Spatial ResNet-18 → (B, 49, 256)
          └── High frequency → Spatial ResNet-18 → (B, 49, 256)
                                        ↓
                               Fusion (concat + linear)
                                        ↓
                              Transformer Decoder
                                        ↓
                                    Caption
```

Decoding is performed with beam search (beam size = 5).

## Datasets

| Dataset | Images | Captions | Domain |
|---------|--------|----------|--------|
| Flickr8k | 8,091 | 5 per image | Semantic (people, animals, activities) |
| CLEVR | 85,000 | 3 per image (synthetic) | Geometric (shapes, colors, spatial relations) |

CLEVR captions are synthetically generated from scene metadata,
describing object properties and spatial relationships.

## Results

### Flickr8k

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|-------|--------|--------|--------|--------|
| Baseline (spatial + beam search) | 0.620 | 0.444 | 0.319 | 0.217 |
| FreqAware (spatial + beam search) | 0.577 | 0.389 | 0.264 | 0.175 |

### CLEVR

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|-------|--------|--------|--------|--------|
| Baseline | 0.982 | 0.977 | 0.969 | 0.957 |
| FreqAware | 0.975 | 0.969 | 0.959 | 0.945 |

### Key finding

Both models achieve near-perfect scores on CLEVR, confirming that
geometric captioning is learnable with sufficient structure.
On Flickr8k, the frequency decomposition does not provide additional
benefit over a single encoder, likely because the dataset is
semantically-focused rather than geometry-focused.

FFT cutoff sensitivity analysis showed that cutoff=20 yields the
best BLEU-4, with performance degrading for both smaller and larger values.

## Repository Structure
```
├── DLAI.ipynb              # Flickr8k experiment (Cells 1-11)
├── DLAI_CLEVR.ipynb        # CLEVR experiment (Cells 1-11)
├── results/
│   ├── bleu_scores.json
│   ├── bleu_scores_clevr.json
│   ├── bleu_comparison.png
│   ├── bleu_clevr.png
│   ├── cross_dataset_bleu4.png
│   ├── training_curves.png
│   ├── qualitative.png
│   ├── qualitative_clevr.png
│   ├── cutoff_analysis.png
│   └── cutoff_analysis.json
└── README.md
```

## Requirements

- Python 3.10+
- PyTorch
- torchvision
- transformers
- nltk
- Pillow
- pandas
- matplotlib

## Usage

**Flickr8k experiment:** Run `DLAI.ipynb` cells 1–11 on Google Colab with GPU.  
**CLEVR experiment:** Run `DLAI_CLEVR.ipynb` cells 1–11 on Google Colab with GPU.  
Dataset downloads automatically. Models are saved to Google Drive automatically.
