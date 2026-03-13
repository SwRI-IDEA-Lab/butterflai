# 🦋 ButterflAI

**Synthetic Solar Cycle Generation via Score-Based Diffusion Models**  
COFFIES Science Center · 11-Week Student Research Program

---

## Overview

ButterflAI is an 11-week undergraduate research program (~40 students) building a
synthetic solar cycle generator from the ground up. Starting from classical statistical
descriptions of the solar butterfly diagram (sunspot latitude-time emergence patterns),
students progressively develop reverse diffusion / score-based generative models capable
of producing physically plausible synthetic solar cycles.

**Primary environment:** Google Colab Pro  
**Stack:** Python · NumPy · Pandas · Matplotlib · SciPy · PyTorch · PyTorch Lightning

---

## Repository Structure

```
butterflai/
├── infrastructure/         # Shared, reusable code provided to students
│   ├── data/               # Data loading, preprocessing, solar cycle utilities
│   ├── models/             # Base classes and reusable model components
│   ├── utils.py              # Training helpers, metrics, Colab setup utilities
│
├── weeks/                  # Weekly Jupyter notebooks (released gradually)
│   ├── week_01/            # Orientation + Solar Data Exploration
│   ├── week_02/            # Statistical Descriptions of the Butterfly Diagram
│   ├── week_03/            # ...
│   └── ...
│
├── .github/
│   └── workflows/          # CI: notebook smoke tests, import checks
│
├── CLAUDE.md               # Context file for Claude Code (AI assistant)
├── pyproject.toml          # Project metadata and dependencies
├── requirements.txt        # Colab-compatible pinned dependencies
└── README.md               # This file
```
<!-- 
---

## Quickstart (Google Colab)

Each weekly notebook begins with a standard setup cell. Just run it:

```python
# Standard ButterflAI Colab setup — always run this first
!git clone https://github.com/YOUR_ORG/butterflai.git 2>/dev/null || \
    (cd butterflai && git pull)
import sys
sys.path.insert(0, '/content/butterflai')
from infrastructure.utils.colab_setup import setup
setup()
```

---

## Weekly Schedule

| Week | Topic |
|------|-------|
| 01 | Orientation · Solar data sources · Butterfly diagram basics |
| 02 | Statistical description of sunspot emergence |
| 03 | Probability distributions · KDE · Sampling |
| 04 | Time series · Autocorrelation · Cycle decomposition |
| 05 | Introduction to neural networks · MLPs in PyTorch |
| 06 | Variational Autoencoders · Latent space structure |
| 07 | Score functions · Denoising · Diffusion intuition |
| 08 | Score-based generative models (SMLD) |
| 09 | Reverse diffusion · DDPM implementation |
| 10 | Conditioning · Physics-informed constraints |
| 11 | Evaluation · Synthesis · Final presentations |

---

## For Students

- Notebooks are released each Monday morning via GitHub
- Weekly GitHub pushes **are** your progress reports — commit often
- Use Colab Pro for GPU-accelerated training (weeks 06+)
- Questions → open a GitHub Discussion, not email

## For the PI 

See `CLAUDE.md` for AI assistant context and `infrastructure/` for the module API. -->
