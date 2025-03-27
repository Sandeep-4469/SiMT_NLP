# Simultaneous Machine Translation with Adapters (SiMT_NLP)

A PyTorch implementation of fixed and adaptive simultaneous machine translation strategies using adapter modules, based on the paper ["Fixed and Adaptive Simultaneous Machine Translation Strategies Using Adapters"](https://arxiv.org/pdf/2407.13469).

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
git clone https://github.com/Sandeep-4469/SiMT_NLP.git
cd SiMT_NLP

pip install -r requirements.txt

SiMT_NLP/
│
├── algo2.py          # Adapter module implementation for Transformer
├── eval.py             # Transformer model with adapter integration
├── train.py             # Utility functions (padding, tokenization, etc.)
├── dataset.py           # Dataset loading and preprocessing
├── train.py             # Training script with checkpoint management
├── requirements.txt     # Python dependencies
├── run.sh               # Training automation script
└── README.md            # Project documentation

python3 train.py