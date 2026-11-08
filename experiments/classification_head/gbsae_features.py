"""Optional frozen GB-SAE/projector building blocks.

This adapter returns projected tokens, not experiment-specific classifier
features. Pooling, shared reconstruction and fusion belong to experiment code.
Researchers can pass a token_transform callback for other backbone grids.
"""
import torch
from torch import nn
from einops import rearrange

from gbsae.models.config import build_sae, restore_sae_state, build_projectors

def freeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False

def build_frozen_projectors(cfg, ckpt, sources, device: str):
    projectors = {}
    for source in sources:
        projector = build_projectors(cfg, [source], device)[source]
        assert source in ckpt["linear_projectors"], f"Projector for {source} not found in checkpoint"
        projector.load_state_dict(ckpt["linear_projectors"][source])
        projector.to(device).eval()
        freeze_all(projector)
        projectors[source] = projector
    return projectors

def build_frozen_sae(cfg, ckpt, sources, device: str):
    sae = build_sae(cfg, sources, device)
    restore_sae_state(sae, ckpt["model_state_dict"])
    sae.eval()
    freeze_all(sae)
    return sae

@torch.no_grad()
def projected_tokens(batch_set, sources, projectors, device, *, token_transform=None):
    """Project each source's tokens; caller supplies any spatial alignment.

    Returns a source-> [B,T,D] mapping. No pooling/fusion or class assumptions
    are made. token_transform accepts (source, tokens), matching GB-SAE training.
    """
    batch_size = batch_set[sources[0]].shape[0]
    tokens = {}
    for source in sources:
        act = batch_set[source].to(device, non_blocking=True)
        if token_transform is not None:
            act = token_transform(source, act)
        flat = rearrange(act, "b t d -> (b t) d")
        tokens[source] = rearrange(projectors[source](flat), "(b t) d -> b t d", b=batch_size)
    return tokens
