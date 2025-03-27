from enum import Enum

class TranslationMode(Enum):
    FIXED = "fixed"
    ADAPTIVE = "adaptive"

CONFIG = {
    "d_model": 512,
    "num_heads": 8,
    "d_ff": 2048,
    "num_layers": 4,
    "dropout": 0.1,
    "num_adapters": 4,
    "bottleneck_size": 64,
    "batch_size": 32,
    "lr": 5e-4,
    "num_epochs": 5
}