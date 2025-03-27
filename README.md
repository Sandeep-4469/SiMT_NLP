# Simultaneous Machine Translation with Adapters (SiMT_NLP)

A PyTorch implementation of fixed and adaptive simultaneous machine translation strategies using adapter modules, based on the paper ["Fixed and Adaptive Simultaneous Machine Translation Strategies Using Adapters"](https://arxiv.org/abs/XXXX.XXXX).

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Testing](#testing)
- [Acknowledgments](#acknowledgments)

## Features
- 🚀 Transformer model with lightweight adapter modules
- ⏱️ Two inference strategies:
  - Fixed wait-k policy
  - Adaptive threshold-based strategy
- 🔄 Multi-path training with parameter sharing
- 📊 Latency-quality tradeoff management
- 💾 Checkpoint averaging and management

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/SiMT_NLP.git
cd SiMT_NLP

pip install -r requirements.txt

SiMT_NLP/
│
├── adapters.py          # Adapter module implementation for Transformer
├── model.py             # Transformer model with adapter integration
├── utils.py             # Utility functions (padding, tokenization, etc.)
├── dataset.py           # Dataset loading and preprocessing
├── train.py             # Training script with checkpoint management
├── fixed_inference.py   # Fixed wait-k inference implementation (Algorithm 1 from paper)
├── adaptive_inference.py# Adaptive inference (Algorithm 2 from paper)
├── config.py            # Configuration settings (hyperparameters & enums)
├── test.py              # Interactive translation interface
├── requirements.txt     # Python dependencies
├── run.sh               # Training automation script
└── README.md            # Project documentation

./run.sh