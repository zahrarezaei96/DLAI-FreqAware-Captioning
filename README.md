# Frequency-Aware Image Captioning

Project for the Deep Learning and Applied AI course (2025/2026)  
Sapienza University of Rome — Prof. Emanuele Rodolà
GitHub: https://github.com/zahrarezaei96/DLAI-FreqAware-Captioning

## Idea

Standard image captioning models process images through a single CNN encoder,
producing a global feature vector that loses spatial and structural information.
This project proposes a dual-branch architecture that decomposes the input image
into low-frequency (structural) and high-frequency (detail) components using FFT,
encodes each branch separately with a spatial CNN encoder, and fuses them before
passing to a Transformer decoder for caption generation.

## Architecture

Image → FFT decomposition
          ├── Low frequency  → Spatial ResNet-18 → (B, 49, 256)
          └── High frequency → Spatial ResNet-18 → (B, 49, 256)
                                        ↓
                               Fusion (concat + linear)
                                        ↓
                              Transformer Decoder
                                        ↓
                                    Caption

Decoding is performed with beam search (beam size = 5).

## Dataset

Flickr8k — 8,091 images, 5 captions each  
Split: 90% train / 10% validation

## Results

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|-------|--------|--------|--------|--------|
| Baseline (spatial + beam search) | 0.620 | 0.444 | 0.319 | 0.217 |
| FreqAware (spatial + beam search) | 0.577 | 0.389 | 0.264 | 0.175 |

FFT cutoff sensitivity analysis showed that cutoff=20 yields the best BLEU-4,
with performance degrading for both smaller and larger values.

## Repository Structure

├── DLAI.ipynb          # Main notebook (Cells 1-11)
├── results/
│   ├── bleu_scores.json
│   ├── bleu_comparison.png
│   ├── training_curves.png
│   ├── qualitative.png
│   ├── cutoff_analysis.png
│   └── cutoff_analysis.json
└── README.md

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

Run cells 1–11 in order on Google Colab with GPU.  
Dataset downloads automatically from Kaggle (Flickr8k).  
Models are saved to and loaded from Google Drive automatically.
