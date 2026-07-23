# GeoJet-TS Visualization - Experiment 5.4 A1 (Adaptive Primitive Geometry)

This directory contains the visualization scripts for Experiment 5.4 A1 ("Adaptive primitive geometry") in GeoJet-TS.

## Overview

Experiment 5.4 A1 demonstrates how GeoJet-TS dynamically adapts Gaussian primitive parameters:
- Center locations ($\mu_k \in [0, 1]$ mapped to timestep indices $0 \dots 511$)
- Primitive widths ($\sigma_k > 0$)
- Primitive saliencies ($\alpha_k \in [0, 1]$)
- Active routed primitive count ($K(x) = \sum_k g_k$)

The scripts are divided into two decoupled modules:
1. `extract_geometry_a1.py`: Runs test set evaluation, extracts primitive internal variables, saves structured `.npz` raw data, and outputs quick diagnostic plots.
2. `plot_geometry_a1.py`: Reads the extracted raw data and renders publication-ready PDF and high-resolution PNG figures.

---

## Directory Structure

```text
visualization/
├── extract_geometry_a1.py    # Extraction script
├── plot_geometry_a1.py       # Plotting script
├── README.md                 # Reproduction guide
└── outputs/
    └── geometry_a1/
        ├── raw/              # Extracted .npz raw arrays
        ├── diagnostics/      # Diagnostic PNG checks
        ├── figures/          # Publication PDF & PNG figures
        └── manifest.json     # Summary manifest
```

---

## Step-by-Step Usage Guide

### 1. Data & Geometry Extraction

Run `extract_geometry_a1.py` using python (in the `demo` environment):

```bash
python visualization/extract_geometry_a1.py \
  --datasets ETTh1,ETTm1,weather,electricity \
  --pred_len 96 \
  --checkpoints ./checkpoints \
  --output_dir ./visualization/outputs/geometry_a1
```

**Parameters**:
- `--datasets`: Comma-separated dataset names (`ETTh1,ETTm1,weather,electricity`).
- `--pred_len`: Prediction horizon (default `96`).
- `--seq_len`: Input sequence length (default `512`).
- `--checkpoints`: Path to checkpoints directory containing trained model weights.
- `--output_dir`: Output root directory for raw data and figures.

---

## 2. Generate Publication Figures

Run `plot_geometry_a1.py` to generate vector PDF and 300 DPI PNG figures:

```bash
python visualization/plot_geometry_a1.py \
  --raw_dir ./visualization/outputs/geometry_a1/raw \
  --output_dir ./visualization/outputs/geometry_a1/figures \
  --datasets ETTh1,ETTm1,weather,electricity \
  --pred_len 96
```

**Output Files**:
- `figures/geometry_a1_overview_H96.pdf`: Main paper 4-dataset overview figure.
- `figures/geometry_a1_overview_H96.png`: High-res PNG version.
- `figures/{dataset}_H96_geometry_a1.pdf`: Individual dataset 3-level (Low, Mid, High complexity) figures.

---

## Figure Panel Description

Each visualization figure consists of two vertically aligned panels:
- **Upper Panel**: RevIN-normalized input sequence $x \in \mathbb{R}^{512}$ received by the primitive generator.
- **Lower Panel**: Scalar support response envelopes $r_k(t) = \widetilde{\alpha}_k \exp\left(-\frac{(t-\mu_k)^2}{2\sigma_k^2}\right)$ (or horizontal support bars $[\mu_k \pm 2\sigma_k]$) of active Gaussian primitives ($g_k = 1$).
