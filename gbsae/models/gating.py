from typing import Dict
import torch
from torch import nn


# Shared/private code partition after a common Top-K selection
class SharedPrivateTopK(nn.Module):
    """Apply one Top-K budget, then split codes with a global per-component gate.
    
    k is the total budget before gating; the branches have no separate hard
    Top-K budgets. Learned gates are sigmoid logits shared across tokens and sources.
    fixed_random uses a frozen Bernoulli partition; none uses a deterministic
    binary partition of component indices. Both fixed modes still apply a gate.
    """

    def __init__(
        self,
        k: int,
        n_components: int,
        gate_init_std: float = 0.01,
        gate_mode: str = "learned",
        gate_ratio_target: float = 0.5,
        gate_random_seed: int = 0,
    ):
        super().__init__()

        self.k = k
        self.n_components = n_components

        self.gate_mode = gate_mode

        # Gate construction
        if gate_mode == "learned":
            # ---- learnable gate logits: (K,) ----
            gate = torch.randn(n_components) * gate_init_std
            self.gate_logits = nn.Parameter(gate)
            self.register_buffer("gate_buf", None)

        elif gate_mode == "fixed_random":
            # ---- random but fixed gate ----
            gen = torch.Generator()
            gen.manual_seed(gate_random_seed)

            g = torch.empty(n_components)
            g.bernoulli_(gate_ratio_target, generator=gen)

            self.register_buffer("gate_buf", g)   # (K,), values in {0,1}
            self.gate_logits = None

        elif gate_mode == "none":
            # ---- deterministic split (no gate learning) ----
            n_sh = int(round(n_components * gate_ratio_target))
            g = torch.zeros(n_components)
            g[:n_sh] = 1.0

            self.register_buffer("gate_buf", g)   # (K,), fixed
            self.gate_logits = None

        else:
            raise ValueError(
                f"Unknown gate_mode '{gate_mode}'. "
                "Expected one of ['learned', 'fixed_random', 'none']."
            )

    # Utilities
    @staticmethod
    def _topk_sparsify(h: torch.Tensor, k: int) -> torch.Tensor:
        """Keep the k largest values per row, preserving their original magnitudes.
        
        For k <= 0 or k >= the component count, return the dense input unchanged.
        """
        if k <= 0 or k >= h.shape[1]:
            return h

        vals, idx = torch.topk(h, k=k, dim=-1)
        out = torch.zeros_like(h)
        out.scatter_(-1, idx, vals)
        return out

    def _get_gate(self) -> torch.Tensor:
        """
        Returns gate g of shape (K,)
        """
        if self.gate_mode == "learned":
            return torch.sigmoid(self.gate_logits)
        else:
            return self.gate_buf

    # Forward
    def forward(self, h_dict: Dict[str, torch.Tensor], agree_lambda: float = 0.0):
        """Split source-> [N,K] encoder activations into shared/private mappings.
        
        agree_lambda is a retained compatibility argument and is not used.
        """

        z_sh = {}
        z_pr = {}

        g = self._get_gate()           # (K,)
        g_row = g.unsqueeze(0)         # (1, K)

        for name, h in h_dict.items():
            z = self._topk_sparsify(h, self.k)
            z_sh[name] = z * g_row
            z_pr[name] = z * (1.0 - g_row)

        return z_sh, z_pr

    # Public API (used by loss / logger)
    def gate(self) -> torch.Tensor:
        """
        Always return a gate tensor (K,), even for ablations.
        """
        return self._get_gate()

