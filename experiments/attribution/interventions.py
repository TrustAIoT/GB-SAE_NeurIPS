"""Contribution-specific interventions evaluated against the full reconstruction."""
from dataclasses import dataclass

import torch


@dataclass
class InterventionResult:
    """Per-image scores [suff/comp, image, k] and fixed-k agreement counts."""

    scores: torch.Tensor
    remove_matches: int
    keep_matches: int


@torch.no_grad()
def evaluate_concept_interventions(
    z_sh, z_pr, D_sh, D_pr, contrib, probe, targets, *, group, ks, agreement_k,
):
    """Keep only selected group contributions; remove only those contributions.

    Negative ground-truth contributions tie at zero when ranking. Shared and
    private interventions affect their respective branches. Total selects each
    index's shared-plus-private contribution. The reference is always the full
    shared-plus-private reconstruction, including the classifier bias.
    """
    ks = tuple(ks)
    if group not in ("sh", "pr", "total"):
        raise ValueError(f"Unknown contribution group: {group}")
    if not ks or min(ks) < 1 or max(ks) > z_sh.shape[1] or agreement_k not in ks:
        raise ValueError("Concept counts must fit the latent width and include agreement_k")

    rep_full = z_sh @ D_sh + z_pr @ D_pr
    full_logits = probe(rep_full)
    full = full_logits.gather(1, targets[:, None]).squeeze(1)
    denominator = full.abs() + 1e-6
    reference = full_logits.argmax(dim=1)
    indices = contrib.clamp_min(0).argsort(dim=1, descending=True)
    suff, comp = [], []
    remove_matches = keep_matches = 0

    for k in ks:
        mask = torch.zeros_like(z_sh).scatter_(1, indices[:, :k], 1.)
        if group == "sh":
            keep = (z_sh * mask) @ D_sh
        elif group == "pr":
            keep = (z_pr * mask) @ D_pr
        else:
            keep = (z_sh * mask) @ D_sh + (z_pr * mask) @ D_pr

        if group == "total":
            # Decode the complementary mask directly to preserve total arithmetic.
            remove = (z_sh * (1 - mask)) @ D_sh + (z_pr * (1 - mask)) @ D_pr
        else:
            remove = rep_full - keep

        removed_logits = probe(remove)
        kept_logits = probe(keep)
        removed = removed_logits.gather(1, targets[:, None]).squeeze(1)
        retained = kept_logits.gather(1, targets[:, None]).squeeze(1)
        comp.append((full - removed) / denominator)
        suff.append((full - retained) / denominator)
        if k == agreement_k:
            remove_matches = int((removed_logits.argmax(1) == reference).sum().item())
            keep_matches = int((kept_logits.argmax(1) == reference).sum().item())

    scores = torch.stack((torch.stack(suff, dim=1), torch.stack(comp, dim=1)))
    if not bool(torch.isfinite(scores).all()):
        raise FloatingPointError("Nonfinite concept attribution scores")
    return InterventionResult(scores, remove_matches, keep_matches)
