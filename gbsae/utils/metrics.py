"""R-squared metric, adapted from Overcomplete; see THIRD_PARTY_NOTICES.md."""
import torch

_EPSILON = 1e-6  # Stabilize the denominator for low-variance inputs.

def r2_score(x, x_hat):
    """Return scalar tensor R-squared using a global mean over all matrix entries.
    
    Inputs must have identical [N,D] shapes. SSE/TSS are computed as mean squared
    errors with epsilon 1e-6 in the denominator, matching the source utility.
    This is not a per-feature average or a Python float.
    """
    assert x.shape == x_hat.shape, "Input tensors must have the same shape"
    assert len(x.shape) == 2, "Input tensors must be 2D"

    ss_res = torch.mean((x - x_hat) ** 2)
    ss_tot = torch.mean((x - x.mean()) ** 2)

    r2 = 1 - (ss_res / (ss_tot + _EPSILON))

    return torch.mean(r2)

