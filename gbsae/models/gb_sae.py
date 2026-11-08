import torch
from torch import nn
from dataclasses import dataclass
from typing import Dict, Tuple

from gbsae.nn.dictionary import DictionaryLayer
from gbsae.nn.factory import EncoderFactory

from gbsae.models.gating import SharedPrivateTopK


@dataclass
class GBSAEOutput:
    pre_codes: Dict[str, torch.Tensor]  # before the configured output activation
    z_sh: Dict[str, torch.Tensor]
    z_pr: Dict[str, torch.Tensor]
    recon: Dict[str, torch.Tensor]
    D_sh: torch.Tensor                  # shared dictionary (n_components, input_dim)
    D_pr: Dict[str, torch.Tensor]       # per-model private dictionary (K, D)
    g: torch.Tensor                     # global sigmoid gate (n_components,)


class GatedBottleneckSAE(nn.Module):
    """
    Bottleneck SAE with:
    - shared encoder
    - Top-K sparsity followed by a global learned shared/private gate
    - ONE shared decoder (D_sh)
    - private decoder per model (D_pr[name])

    k controls the single per-token Top-K budget before gate partitioning.

    Reconstruction:
        x_hat_i = D_sh(z_sh_i) + D_pr[i](z_pr_i)
    """

    def __init__(
        self,
        input_dim: int,
        n_components: int,
        model_names,
        k: int,
        encoder_module="linear",
        dictionary_norm="l2",
        device="cpu",
    ):
        super().__init__()

        self.model_names = model_names
        self.n_components = n_components

        # shared encoder
        self.encoder = EncoderFactory.create_module(
            encoder_module,
            input_dim,
            n_components,
            device=device,
        )

        # bottleneck: TopK + global gate (from gating.py)
        self.bottleneck = SharedPrivateTopK(
            k=k,
            n_components=n_components,
            gate_init_std=0.01
        )

        # shared decoder (ONE dictionary)
        self.decoder_sh = DictionaryLayer(
            n_components, input_dim, dictionary_norm
        )

        # private decoder per model
        self.decoder_pr = nn.ModuleDict()
        for name in model_names:
            self.decoder_pr[name] = DictionaryLayer(
                n_components, input_dim, dictionary_norm
            )

    def forward(self, batch_set: Dict[str, torch.Tensor]):
        """
        batch_set: model_name -> activation tensor (N, D)
        Provide projected token tensors for all names in self.model_names.
        """

        pre_codes_dict = {}
        h = {}

        # encode
        for name, a in batch_set.items():
            pre_codes, codes = self.encoder(a)  # codes: (N, K)
            pre_codes_dict[name] = pre_codes
            h[name] = codes

        # bottleneck: TopK + gate split
        z_sh, z_pr = self.bottleneck(h)

        # reconstruction
        recon = {}
        for name in self.model_names:
            recon[name] = (
                self.decoder_sh(z_sh[name])
                + self.decoder_pr[name](z_pr[name])
            )

        # dictionaries
        D_sh = self.decoder_sh.get_dictionary()
        D_pr = {}
        for name in self.model_names:
            D_pr[name] = self.decoder_pr[name].get_dictionary()

        # gate (from bottleneck)
        g = self.bottleneck.gate()  # (K,)

        return GBSAEOutput(
            pre_codes=pre_codes_dict,
            z_sh=z_sh,
            z_pr=z_pr,
            recon=recon,
            D_sh=D_sh,
            D_pr=D_pr,
            g=g,
        )


