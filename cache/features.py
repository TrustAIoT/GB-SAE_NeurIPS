"""Backbone registry and native feature adapters; no SAE preprocessing."""
from dataclasses import dataclass
import math

import torch

from .utils import import_symbol


@dataclass
class FeatureExtractor:
    """A model with PIL->tensor preprocess and tensor->native-features extraction.

    Custom factories return this object and declare the native tensor layout,
    prefix count and provenance. Outputs are [B,T,D], [B,D,H,W] or [B,H,W,D].
    """
    model: object
    preprocess: object
    forward: object
    native_layout: str
    prefix_tokens: int
    provenance: dict


def _builtin(name, *, device="cpu", **kwargs):
    from gbsae.backbones import models
    model = getattr(models, name)(device=device, **kwargs).to(device).eval()
    weights = {
        "ViT": "vit_base_patch16_224", "SigLIP": "vit_base_patch16_siglip_224.webli",
        "DinoV2": "facebookresearch/dinov2:dinov2_vits14",
        "CLIP": "vit_base_patch16_clip_224.openai", "ConvNeXt": "convnext_small.fb_in1k",
        "ResNet": "resnetv2_50x1_bit.goog_distilled_in1k",
    }
    return FeatureExtractor(
        model, model.preprocess, model.forward_features,
        "channels_first" if name in {"ConvNeXt", "ResNet"} else "tokens",
        int(name in {"ViT", "DinoV2", "CLIP"}),
        {"weights": weights[name], "feature": ("normalized x_norm_clstoken + x_norm_patchtokens" if name == "DinoV2"
                                              else "model.forward_features, native unpooled output"),
         "preprocess": repr(model.preprocess), "torch_version": str(torch.__version__)},
    )


def _builder(name):
    return lambda **kwargs: _builtin(name, **kwargs)


EXTRACTOR_REGISTRY = {name: _builder(name) for name in ("ViT", "SigLIP", "DinoV2", "ConvNeXt", "ResNet", "CLIP")}


def register_backbone(name: str, factory):
    """Register a factory(device=..., **kwargs)->FeatureExtractor."""
    if name in EXTRACTOR_REGISTRY:
        raise ValueError(f"Backbone already registered: {name}")
    EXTRACTOR_REGISTRY[name] = factory


def build_extractor(options: dict, device: str) -> FeatureExtractor:
    kind = options["type"]
    factory = EXTRACTOR_REGISTRY.get(kind)
    if factory is None:
        factory = import_symbol(kind)
    extractor = factory(device=device, **options.get("kwargs", {}))
    if not isinstance(extractor, FeatureExtractor):
        raise TypeError("Backbone factories must return cache.features.FeatureExtractor")
    extractor.model.eval()
    if type(extractor.prefix_tokens) is not int or extractor.prefix_tokens < 0:
        raise ValueError("Extractor prefix_tokens must be a nonnegative integer")
    return extractor


def format_features(features, extractor: FeatureExtractor, options: dict):
    """Convert layout without changing values; retain prefixes and native grid."""
    if not isinstance(features, torch.Tensor):
        raise TypeError("Feature adapters must select a tensor from the backbone output")
    native = extractor.native_layout
    prefix = extractor.prefix_tokens
    if native == "tokens" and features.ndim == 3:
        tokens = features
        patch_count = tokens.shape[1] - prefix
        side = math.isqrt(max(0, patch_count))
        if patch_count <= 0 or side * side != patch_count:
            raise ValueError("Cache currently requires square patch grids; rectangular/non-spatial outputs need an adapter")
        height = width = side
    elif native in {"channels_first", "channels_last"} and features.ndim == 4:
        if prefix:
            raise ValueError("Pure spatial feature maps cannot contain CLS/prefix tokens")
        spatial = features.permute(0, 2, 3, 1) if native == "channels_first" else features
        height, width = spatial.shape[1:3]
        if height != width:
            raise ValueError(f"Rectangular feature maps are not supported: {height}x{width}")
        tokens = spatial.reshape(spatial.shape[0], height * width, spatial.shape[-1])
    else:
        raise ValueError(f"Native layout {native!r} does not match {tuple(features.shape)}")
    if "feature_dim" in options and tokens.shape[-1] != options["feature_dim"]:
        raise ValueError(f"Expected feature_dim={options['feature_dim']}, got {tokens.shape[-1]}")
    layout = options.get("storage_layout", "tokens")
    if layout == "tokens":
        saved = tokens
    else:
        if prefix:
            raise ValueError("Features with CLS/prefix tokens must use storage_layout: tokens")
        saved = tokens.reshape(tokens.shape[0], height, width, tokens.shape[-1])
        if layout == "channels_first":
            saved = saved.permute(0, 3, 1, 2)
    metadata = {"layout": layout, "prefix_tokens": prefix, "height": height, "width": width,
                "feature_dim": tokens.shape[-1], "shape": list(saved.shape[1:]), "dtype": str(saved.detach().cpu().numpy().dtype)}
    return saved.detach().cpu().contiguous(), metadata


@torch.no_grad()
def extract_features(extractor: FeatureExtractor, inputs, options: dict):
    return format_features(extractor.forward(inputs), extractor, options)
