from pathlib import Path
from typing import Dict, Mapping

import torch
from gbsae.configuration import load_model_config
from einops import rearrange

from .specs import FUSED_VIEW
from .configuration import method_sources, provider_metadata


def _torch_load(path: str, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _freeze(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False


def _pool_tokens(tokens: torch.Tensor, mode: str) -> torch.Tensor:
    if tokens.ndim == 2:
        return tokens
    if mode == "mean":
        return tokens.mean(dim=1)
    if mode == "max":
        return tokens.max(dim=1).values
    if mode == "sum":
        return tokens.sum(dim=1)
    raise ValueError(f"Unsupported token pooling: {mode}")


class FrozenFeatureProvider:
    method: str

    def __init__(self, device: str, pooling: str):
        self.device = torch.device(device)
        self.pooling = pooling
        self.view_dims: Dict[str, int] = {}
        self.checkpoint_paths: Dict[str, str] = {}
        self.cache_metadata: Dict[str, object] = {}

    @torch.inference_mode()
    def extract(self, batch_set: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class GBSAEFeatureProvider(FrozenFeatureProvider):
    """Extract checkpoint-standardized projected or shared-reconstructed features.
    
    projected uses frozen projectors without creating an SAE. gbsae reconstructs
    shared tokens before pooling and also returns their mean across sources as
    FUSED. Source order and preprocessing follow the training configuration.
    """
    def __init__(self, experiment_cfg: dict, device: str, pooling: str, method: str):
        super().__init__(device, pooling)
        from gbsae.models.config import build_sae, restore_sae_state
        from gbsae.models.config import build_projectors

        self.method = method
        checkpoint_path = experiment_cfg["checkpoints"]["gbsae"]
        config_path = experiment_cfg["gbsae_config"]
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"GB-SAE checkpoint not found: {checkpoint_path}")
        self.gb_cfg = load_model_config(config_path)
        from gbsae.data.tokens import build_token_transform
        self.token_transform = build_token_transform(self.gb_cfg)
        self.cache_metadata = provider_metadata(experiment_cfg, method)
        self.checkpoint = _torch_load(checkpoint_path, self.device)
        self.checkpoint_paths = {"gbsae": checkpoint_path}
        self.sources = method_sources(experiment_cfg, method)

        proj_cfg = self.gb_cfg["projector"]
        self.projectors = build_projectors(self.gb_cfg, self.sources, self.device)
        for source in self.sources:
            self.projectors[source].load_state_dict(self.checkpoint["linear_projectors"][source])
            self.projectors[source].to(self.device)
            _freeze(self.projectors[source])

        self.sae = None
        if method == "gbsae":
            self.sae = build_sae(self.gb_cfg, self.sources, self.device)
            restore_sae_state(self.sae, self.checkpoint["model_state_dict"])
            _freeze(self.sae)

        self.standardize = self.gb_cfg["data"].get("standardize", False)
        self.divide_norm = self.gb_cfg["data"].get("divide_norm", False)
        stats = self.checkpoint.get("standardization_stats", {})
        if self.standardize and any(source not in stats for source in self.sources):
            raise RuntimeError("GB-SAE checkpoint has no standardization_stats")
        self.standardization_stats = {
            source: {
                "mean": stats[source]["mean"].detach().to(self.device),
                "std": stats[source]["std"].detach().to(self.device),
            }
            for source in self.sources if self.standardize
        }
        output_dim = int(proj_cfg["output_dim"])
        self.view_dims = {source: output_dim for source in self.sources}
        if method == "gbsae":
            self.view_dims[FUSED_VIEW] = output_dim

    def _prepare_tokens(self, source: str, values: torch.Tensor) -> torch.Tensor:
        values = values.to(self.device, non_blocking=True).float()
        if self.standardize:
            stats = self.standardization_stats[source]
            values = (values - stats["mean"]) / (stats["std"] + 1e-5)
        elif self.divide_norm:
            values = values / values.norm(dim=-1, keepdim=True)
        return self.token_transform(source, values)

    @torch.inference_mode()
    def extract(self, batch_set: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        projected_flat = {}
        projected_tokens = {}
        batch_size = batch_set[self.sources[0]].shape[0]
        for source in self.sources:
            tokens = self._prepare_tokens(source, batch_set[source])
            flat = rearrange(tokens, "b t d -> (b t) d")
            projected = self.projectors[source](flat)
            projected_flat[source] = projected
            projected_tokens[source] = rearrange(
                projected, "(b t) d -> b t d", b=batch_size, t=tokens.shape[1]
            )

        if self.method == "projected":
            return {
                source: _pool_tokens(projected_tokens[source], self.pooling).float()
                for source in self.sources
            }

        output = self.sae(projected_flat)
        shared = {}
        for source in self.sources:
            z_shared = rearrange(
                output.z_sh[source],
                "(b t) k -> b t k",
                b=batch_size,
                t=projected_tokens[source].shape[1],
            )
            shared_tokens = z_shared @ output.D_sh
            shared[source] = _pool_tokens(shared_tokens, self.pooling).float()
        shared[FUSED_VIEW] = torch.stack([shared[source] for source in self.sources]).mean(dim=0)
        return shared


def build_feature_provider(
    method: str,
    experiment_cfg: dict,
    device: str,
    pooling: str,
) -> FrozenFeatureProvider:
    method = method.lower()
    if method in {"projected", "gbsae"}:
        return GBSAEFeatureProvider(experiment_cfg, device, pooling, method)
    raise ValueError(f"Unsupported main retrieval method: {method}")
