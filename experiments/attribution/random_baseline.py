"""Uniformly selected components of a random orthogonal decomposition."""
import torch

from .interventions import InterventionResult


def orthogonal_basis(dimension, seed):
    """Seeded Gaussian QR; columns are fixed unit orthogonal directions.

    The basis is independent of representations and labels. A private CPU RNG
    keeps its construction independent of the evaluation loader's random state.
    """
    if dimension < 1:
        raise ValueError("Representation dimension must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    gaussian = torch.randn(dimension, dimension, dtype=torch.float64, generator=generator)
    q, r = torch.linalg.qr(gaussian)
    signs = r.diagonal().sign()
    signs[signs == 0] = 1
    return (q * signs).float()


def sample_indices(batch_size, dimension, max_k, generator):
    """Uniform sampling without replacement per image, nested across k.

    Sorting independent random keys generates the subsets; no feature, label,
    activation magnitude or classifier contribution enters the selection.
    """
    if not 0 < max_k <= dimension:
        raise ValueError("Random component counts must fit the representation dimension")
    keys = torch.rand(batch_size, dimension, generator=generator, dtype=torch.float64)
    return keys.argsort(dim=1)[:, :max_k]


@torch.no_grad()
def evaluate_random_interventions(rep_full, q, indices, probe, targets, *, ks, agreement_k):
    """Project onto Q, then keep/remove the SAME sampled components.

    Coefficients alpha = R_full Q are signed projections, not SAE activations.
    Counts k are matched; contribution norms and classifier scores are not.
    Linear-probe algebra evaluates target logits for every k; explicit vectors
    evaluate all-class prediction agreement at agreement_k. Bias occurs once.
    """
    ks = tuple(ks)
    if (not ks or min(ks) < 1 or max(ks) > indices.shape[1]
            or agreement_k not in ks):
        raise ValueError("Random subsets must cover all counts and agreement_k")
    if rep_full.ndim != 2 or q.shape != (rep_full.shape[1], rep_full.shape[1]):
        raise ValueError("Expected representations [B,d] and an orthogonal basis [d,d]")
    indices = indices.to(rep_full.device)
    coefficients = rep_full @ q
    full_logits = probe(rep_full)
    full = full_logits.gather(1, targets[:, None]).squeeze(1)
    contributions = coefficients * (probe.linear.weight[targets] @ q)
    cumulative = contributions.gather(1, indices).cumsum(dim=1)
    selected = cumulative[:, [k - 1 for k in ks]]
    bias = probe.linear.bias
    bias_y = torch.zeros_like(full) if bias is None else bias[targets]
    kept = bias_y[:, None] + selected
    removed = full[:, None] - selected
    denominator = full.abs()[:, None] + 1e-6
    scores = torch.stack(((full[:, None] - kept) / denominator,
                          (full[:, None] - removed) / denominator))
    if not bool(torch.isfinite(scores).all()):
        raise FloatingPointError("Nonfinite random attribution scores")

    mask = torch.zeros_like(coefficients).scatter_(1, indices[:, :agreement_k], 1.)
    keep = (coefficients * mask) @ q.T
    reference = full_logits.argmax(dim=1)
    remove_matches = int((probe(rep_full - keep).argmax(1) == reference).sum().item())
    keep_matches = int((probe(keep).argmax(1) == reference).sum().item())
    return InterventionResult(scores, remove_matches, keep_matches)
