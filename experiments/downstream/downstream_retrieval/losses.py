from typing import Dict, Iterable, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RetrievalHead(nn.Module):
    """Small trainable adapter that maps a frozen representation to a unit sphere."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        head_type: str = "linear",
        hidden_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        if head_type == "linear":
            self.projection = nn.Linear(input_dim, output_dim)
        elif head_type == "mlp":
            self.projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(features.float()), dim=-1)


def build_heads(
    view_dims: Mapping[str, int],
    head_by_view: Mapping[str, str],
    output_dim: int,
    head_type: str = "linear",
    hidden_dim: int = 512,
    dropout: float = 0.0,
) -> nn.ModuleDict:
    dims_by_head: Dict[str, int] = {}
    for view_name, head_name in head_by_view.items():
        input_dim = int(view_dims[view_name])
        old_dim = dims_by_head.setdefault(head_name, input_dim)
        if old_dim != input_dim:
            raise ValueError(
                f"Tied head {head_name!r} received incompatible dimensions: {old_dim} and {input_dim}"
            )
    return nn.ModuleDict(
        {
            head_name: RetrievalHead(
                input_dim=input_dim,
                output_dim=output_dim,
                head_type=head_type,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            for head_name, input_dim in dims_by_head.items()
        }
    )


def encode_views(
    heads: nn.ModuleDict,
    features: Mapping[str, torch.Tensor],
    head_by_view: Mapping[str, str],
) -> Dict[str, torch.Tensor]:
    return {view: heads[head_by_view[view]](features[view]) for view in head_by_view}


def _instance_infonce(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[0] != logits.shape[1]:
        raise ValueError("Instance InfoNCE requires paired query/gallery batches of equal size")
    targets = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, targets)


def _category_multi_positive_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-view multi-positive InfoNCE with the paired diagonal excluded.
    
    For each query, take negative log(sum exp same-class logits / sum exp
    all non-diagonal logits). Queries with no other same-class example are
    excluded; if all queries lack positives, return a differentiable zero.
    This pools positive probabilities rather than averaging positive log losses.
    """
    batch_size = labels.shape[0]
    if logits.shape != (batch_size, batch_size):
        raise ValueError("Category loss expects a square paired-view similarity matrix")

    eye = torch.eye(batch_size, dtype=torch.bool, device=logits.device)
    candidate_mask = ~eye
    positive_mask = labels[:, None].eq(labels[None, :]) & candidate_mask
    valid = positive_mask.any(dim=1)
    if not bool(valid.any()):
        return logits.sum() * 0.0

    neg_inf = torch.finfo(logits.dtype).min
    denominator = torch.logsumexp(logits.masked_fill(~candidate_mask, neg_inf), dim=1)
    numerator = torch.logsumexp(logits.masked_fill(~positive_mask, neg_inf), dim=1)
    return -(numerator[valid] - denominator[valid]).mean()


def cross_view_metric_loss(
    embeddings: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    ordered_pairs: Iterable[Tuple[str, str]],
    temperature: float,
    instance_weight: float,
    category_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pair_losses = []
    instance_losses = []
    category_losses = []
    for query_name, gallery_name in ordered_pairs:
        logits = embeddings[query_name] @ embeddings[gallery_name].t()
        logits = logits / temperature
        instance_loss = _instance_infonce(logits)
        category_loss = _category_multi_positive_loss(logits, labels)
        pair_loss = instance_weight * instance_loss + category_weight * category_loss
        pair_losses.append(pair_loss)
        instance_losses.append(instance_loss.detach())
        category_losses.append(category_loss.detach())

    if not pair_losses:
        raise ValueError("At least one ordered training pair is required")
    total = torch.stack(pair_losses).mean()
    stats = {
        "loss": float(total.detach()),
        "instance_loss": float(torch.stack(instance_losses).mean()),
        "category_loss": float(torch.stack(category_losses).mean()),
    }
    return total, stats
