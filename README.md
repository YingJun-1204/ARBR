# ARBD: Adaptive Radial Basis Decomposition for Long-Term Time Series Forecasting

Official implementation of **ARBD** (Adaptive Radial Basis Decomposition Network) for long-term multivariate time series forecasting.

---

## 📖 Overview

ARBD achieves state-of-the-art forecasting performance with minimal computational overhead through three core modules:
1. **Adaptive Radial Geometry (ARG)**: Dynamically deforms temporal radial centers and multi-scale temporal spans around structured anchors.
2. **Radial Coefficient Generation & Bilinear Rendering (RCG & Renderer)**: Generates latent content representations and pairs them with query-aligned continuous radial bases.
3. **Calibrated Local Residual Correction (CLRC)**: Employs a bounded, calibrated projection to capture high-frequency local observation details.

---

## 🛠️ Environment Setup

We recommend using **Conda** to manage your Python environment. **Python 3.10** (or 3.8–3.11) is recommended.

### 1. Create and Activate Conda Environment
```bash
conda create -n arbd python=3.10 -y
conda activate arbd
```

### 2. Install PyTorch
Please install PyTorch matching your hardware and CUDA version. For example:
- **CUDA 11.8**:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```
- **CUDA 12.1**:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```
- **CPU only**:
  ```bash
  pip install torch torchvision
  ```

### 3. Install Dependencies
Install the remaining required packages via:
```bash
pip install -r requirements.txt
```

---

## 📂 Data Preparation 

Due to file size limits, benchmark datasets are not tracked in this repository. All standard benchmark datasets can be obtained from the public time-series forecasting repositories.

### 1. Download Datasets
You can download the standard benchmark CSV files from common Google Drive links widely shared by the time-series forecasting community:
- **Google Drive (Benchmark Datasets)**: [Google Drive Download Link](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) (or from the official [Autoformer](https://github.com/thuml/Autoformer) / [Informer](https://github.com/zhouhaoyi/Informer2020) repository data drives).

### 2. Place Data in `./data/`
Create a `data/` directory in the root of the project and place the downloaded CSV files inside:
```text
./data/
├── ETTh1.csv
├── ETTh2.csv
├── ETTm1.csv
├── ETTm2.csv
├── electricity.csv
├── traffic.csv
└── weather.csv
```

---

## 🚀 Reproduction & Quick Start 

### 1. Reproduce ETTh1 Benchmark
All 4 standard horizons ($H \in \{96, 192, 336, 720\}$) for `ETTh1` can be reproduced with a single command:

```bash
# On Linux / macOS / Git Bash / WSL:
bash ./scripts/etth1.sh
```

### 2. Run Other Benchmark Datasets
Scripts for all 7 standard benchmark datasets are provided in the `scripts/` folder:

```bash
bash ./scripts/etth2.sh        # ETTh2
bash ./scripts/ettm1.sh        # ETTm1
bash ./scripts/ettm2.sh        # ETTm2
bash ./scripts/weather.sh      # Weather
bash ./scripts/electricity.sh  # Electricity
bash ./scripts/traffic.sh      # Traffic
```

---

## 📁 Repository Structure 

```text
├── data/                               # Dataset directory (place downloaded CSVs here)
├── data_provider/
│   ├── data_factory.py                 # DataLoader factory
│   └── data_loader.py                  # Dataset implementations
├── exp/
│   ├── exp_basic.py                    # Experiment base class
│   └── exp_long_term_forecasting.py    # Training and evaluation engine
├── layers/
│   ├── RevIN.py                        # Reversible Instance Normalization
│   ├── adaptive_radial_geometry.py     # ARG: Adaptive center and scale deformation
│   ├── radial_coefficient_generation.py# RCG & Bilinear Renderer
│   ├── local_residual_correction.py    # CLRC: Calibrated local residual correction
│   ├── local_residual.py               # Local patch linear projection
│   ├── residual_gate.py                # Bounded residual gate
│   ├── common_blocks.py                # FlattenHead projection
│   └── arbd_encoder.py                 # Unified ARBD Encoder backbone
├── models/
│   ├── arbd.py                         # ARBD paper model assembly
│   └── model_factory.py                # Model instantiation factory
├── scripts/                            # 7 Benchmark dataset reproduction scripts
│   ├── electricity.sh
│   ├── etth1.sh
│   ├── etth2.sh
│   ├── ettm1.sh
│   ├── ettm2.sh
│   ├── traffic.sh
│   └── weather.sh
├── requirements.txt                    # Minimal environment requirements (UTF-8)
├── run.py                              # Main entry point for training and testing
└── .gitignore                          # Git ignore rules
```
