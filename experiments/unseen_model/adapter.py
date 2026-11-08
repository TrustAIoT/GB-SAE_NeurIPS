"""Components for adapting one cached-feature source to a pretrained shared SAE."""
from typing import Callable

import torch

from gbsae.data.tokens import build_token_transform, source_names
from gbsae.data.config import build_activation_dataset
from gbsae.models.config import build_sae
from gbsae.models.config import build_projectors


def get_adapter_source(cfg: dict) -> str:
    """Select exactly one configured cache/module source for adaptation."""
    names = source_names(cfg)
    if len(names) != 1:
        raise ValueError("Adapter training requires exactly one models.sources entry")
    if float(cfg.get("loss", {}).get("lambda_align", 0.0)) != 0.0:
        raise ValueError("Single-source adaptation requires loss.lambda_align=0")
    return names[0]


def build_adapter_dataset(cfg: dict, source: str):
    """Load original-grid caches with the common configured prefix/layout policy."""
    dataset = build_activation_dataset(cfg, [source], combined_npz=cfg["data"].get("combined_npz", False))
    if not len(dataset):
        raise ValueError("The adapter dataset is empty")
    return dataset


def build_adapter_modules(cfg: dict, source: str, device: str):
    """Build a projector/private dictionary and restore the shared SAE state.

    Allow missing parameters only for the target private dictionary, and extra
    parameters only for other source private dictionaries. Shared architecture
    must match the checkpoint; never silently retain randomly initialized shared
    parameters. If the target private state already exists, load it too.
    Return (source_to_projector, sae); configure_adapter_parameters performs
    the separate freezing and trainable-parameter selection step.
    """
    projectors = build_projectors(cfg, [source], device)
    for projector in projectors.values():
        projector.to(device).train()
    sae = build_sae(cfg, [source], device)
    path = cfg["adapter"]["shared_ckpt"]
    print(f"[INFO] Loading shared SAE checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)
    message = sae.load_state_dict(ckpt["model_state_dict"], strict=False)
    invalid_missing = [k for k in message.missing_keys if not k.startswith(f"decoder_pr.{source}.")]
    invalid_extra = [k for k in message.unexpected_keys if not k.startswith("decoder_pr.")]
    if invalid_missing or invalid_extra:
        raise ValueError(
            "Shared SAE checkpoint is incompatible with the configured architecture: "
            f"missing={invalid_missing}, unexpected={invalid_extra}"
        )
    print(f"[INFO] Target private initialization keys: {message.missing_keys}")
    print(f"[INFO] Unused source private keys: {message.unexpected_keys}")
    return projectors, sae


def configure_adapter_parameters(sae, projectors, source: str) -> list[torch.nn.Parameter]:
    """Freeze the shared SAE and return trainable projector/private parameters.

    Do not change model modes or freeze BatchNorm running buffers. Projector
    requires_grad settings come from its factory; identity/frozen projections
    therefore allow private-dictionary-only ablations.
    """
    for module in (sae.encoder, sae.decoder_sh, sae.bottleneck):
        for parameter in module.parameters():
            parameter.requires_grad = False
    return [
        p for p in [*projectors[source].parameters(), *sae.decoder_pr[source].parameters()]
        if p.requires_grad
    ]


def build_adapter_token_transform(cfg: dict, source: str) -> Callable[[str, torch.Tensor], torch.Tensor]:
    """Use the same explicit data.token_alignment settings as GB-SAE training."""
    return build_token_transform(cfg, [source])
