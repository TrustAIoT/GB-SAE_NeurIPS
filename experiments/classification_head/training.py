"""Generic frozen-feature classification loop, with explicit extension points.

Batch contract: (inputs, class_targets). feature_fn(inputs) returns [B,D].
No experiment dispatch, dataset/config construction, output paths or head types
are embedded in the loop. Feature extraction always runs without gradients;
only the supplied classification head is optimized.
"""
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import torch
from torch import nn, optim
from tqdm import tqdm


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.01  # original AdamW default
    max_batches: Optional[int] = None


@dataclass(frozen=True)
class EpochMetrics:
    """Sample-weighted loss and accuracies in percent."""
    loss: float
    acc1: float
    acc5: float
    num_samples: int


def train_one_epoch(head: nn.Module, dataloader: Iterable, feature_fn: Optional[Callable],
                    optimizer, criterion, *, device, max_batches=None, description="Train"):
    """Run a frozen-feature epoch and return sample-weighted classification metrics.
    
    If feature_fn is None, inputs must be a [B,D] tensor moved to the head's
    device. Otherwise feature_fn owns nested inputs, device transfer and pooling.
    criterion must return a scalar batch-mean loss; summation/unreduced losses
    do not match the loss aggregation below. Feature extraction uses no_grad;
    feature_fn defines the representation and normalization policy.
    """
    head.train()
    total_loss, correct1, correct5, total = 0.0, 0, 0, 0
    for step, (inputs, targets) in enumerate(tqdm(dataloader, desc=description)):
        if max_batches is not None and step >= max_batches:
            break
        targets = targets.to(device, non_blocking=True)
        with torch.no_grad():
            features = inputs.to(device) if feature_fn is None else feature_fn(inputs)
        logits = head(features)
        loss = criterion(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_n = targets.shape[0]
        total_loss += loss.item() * batch_n
        _, pred1 = logits.max(dim=1)
        correct1 += (pred1 == targets).sum().item()
        _, pred5 = logits.topk(k=min(5, logits.shape[1]), dim=1)
        correct5 += pred5.eq(targets.view(-1, 1)).any(dim=1).sum().item()
        total += batch_n
    if total == 0:
        raise RuntimeError("No samples were processed. Check max_batches or dataset size.")
    return EpochMetrics(total_loss/total, correct1/total*100.0, correct5/total*100.0, total)


def fit(head: nn.Module, dataloader: Iterable, *, feature_fn: Optional[Callable] = None,
        config: Optional[TrainingConfig] = None, optimizer=None, criterion=None,
        on_epoch_end: Optional[Callable] = None):
    """Train a caller-supplied head and return one EpochMetrics per epoch.

    Defaults retain the original AdamW/cross-entropy protocol. Pass another
    optimizer/loss explicitly for research variants. on_epoch_end(epoch,
    metrics, head) may log, save or adjust a scheduler; this function performs
    no filesystem writes itself. Place the head on the desired device first.
    """
    config = TrainingConfig() if config is None else config
    if config.epochs < 1:
        raise ValueError("epochs must be positive")
    device = next(head.parameters()).device
    optimizer = optim.AdamW(head.parameters(), lr=config.lr, weight_decay=config.weight_decay) if optimizer is None else optimizer
    criterion = nn.CrossEntropyLoss() if criterion is None else criterion
    history = []
    for epoch in range(1, config.epochs+1):
        metrics = train_one_epoch(head, dataloader, feature_fn, optimizer, criterion,
                                  device=device, max_batches=config.max_batches,
                                  description=f"Epoch {epoch}/{config.epochs}")
        history.append(metrics)
        print(f"[Epoch {epoch}] loss={metrics.loss:.4f}, acc@1={metrics.acc1:.2f}%, acc@5={metrics.acc5:.2f}%")
        if on_epoch_end is not None:
            on_epoch_end(epoch, metrics, head)
    return history
