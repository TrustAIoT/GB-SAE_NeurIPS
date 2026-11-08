"""Composable tools for frozen-feature classification research.

The public core works with arbitrary PyTorch loaders/heads and feature callbacks.
GB-SAE and cached ImageNet adapters are optional, explicitly imported submodules.
Experiment workflows live in attribution/ and downstream_classification/.
"""
from .model import LinearProbe
from .training import EpochMetrics, TrainingConfig, fit, train_one_epoch
from .data import make_dataloader
from .pooling import mean_pool_tokens
from .io import CSVLogger, save_head

__all__ = ["LinearProbe", "EpochMetrics", "TrainingConfig", "fit", "train_one_epoch",
           "make_dataloader", "mean_pool_tokens", "CSVLogger", "save_head"]
