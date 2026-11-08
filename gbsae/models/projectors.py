import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


# Base class
class BaseLinearProjector(nn.Module):
    """
    Base class for all linear projectors.
    Input:  (B, d_in)
    Output: (B, d_out)
    """
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# 0. Identity projector (for debugging / ablation)
class IdentityProjector(BaseLinearProjector):
    """
    Identity mapping (only allowed when d_in == d_out)
    """
    def __init__(self, d_in: int, d_out: int):
        assert d_in == d_out, "IdentityProjector requires d_in == d_out"
        super().__init__(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# 1. Frozen Random Projection
class FrozenRandomProjector(BaseLinearProjector):
    """
    Frozen random linear projection.
    - Random Gaussian or Orthogonal initialization
    - No bias
    - Weight is frozen
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        mode: str = "gaussian",  # ["gaussian", "orthogonal"]
        scale: bool = True,
    ):
        super().__init__(d_in, d_out)

        self.linear = nn.Linear(d_in, d_out, bias=False)

        if mode == "gaussian":
            nn.init.normal_(self.linear.weight, mean=0.0, std=1.0)
        elif mode == "orthogonal":
            nn.init.orthogonal_(self.linear.weight)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if scale:
            self.linear.weight.data /= math.sqrt(d_in)

        # Freeze parameters
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# Learnable linear projector
class LearnableLinearProjector(BaseLinearProjector):
    """
    Learnable linear projection:
    A'_i = W_i A_i

    Design choices:
    - No bias
    - Weight regularization is configured by the optimizer
    """

    def __init__(self, d_in: int, d_out: int, init: str = "xavier"):
        super().__init__(d_in, d_out)

        self.linear = nn.Linear(d_in, d_out, bias=False)

        if init == "xavier":
            nn.init.xavier_uniform_(self.linear.weight)
        elif init == "orthogonal":
            nn.init.orthogonal_(self.linear.weight)
        else:
            raise ValueError(f"Unknown init: {init}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# Projector factory
def build_linear_projector(
    projector_type: str,
    d_in: Dict[str, int],
    d_out: int,
    **kwargs,
) -> Dict[str, BaseLinearProjector]:
    """
    Factory method for a group of linear projectors.

    Args:
        projector_type:
            "identity", "frozen_random", "learnable"
        d_in:
            Dict[model_name -> input_dim]
        d_out:
            shared output dimension for all projectors
        kwargs:
            forwarded to specific projector constructors

    Returns:
        Dict[model_name -> BaseLinearProjector]
    """

    projector_type = projector_type.lower()
    projectors: Dict[str, BaseLinearProjector] = {}

    for name, dim in d_in.items():
        if not isinstance(dim, int):
            raise TypeError(
                f"d_in['{name}'] must be int, got {type(dim)}"
            )

        try:
            if projector_type == "identity":
                projectors[name] = IdentityProjector(dim, d_out)

            elif projector_type == "frozen_random":
                projectors[name] = FrozenRandomProjector(dim, d_out, **kwargs)

            elif projector_type == "learnable":
                projectors[name] = LearnableLinearProjector(dim, d_out, **kwargs)

            # elif projector_type == "orthogonal":
            #     projectors[name] = OrthogonalLinearProjector(dim, d_out)

            else:
                raise ValueError(f"Unknown projector_type: {projector_type}")

        except TypeError as e:
            raise TypeError(
                f"Invalid kwargs for projector '{projector_type}': {kwargs}"
            ) from e

    return projectors


