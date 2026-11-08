"""Classification heads; replace this module's head at the call site if needed."""
from torch import nn

class LinearProbe(nn.Module):
    """Bias-equipped linear head for [batch, feature_dim] inputs.

    The linear.* state-dict keys match the original paper probe checkpoints.
    Pooling and feature extraction are deliberately left to the caller.
    """
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.linear(x)

