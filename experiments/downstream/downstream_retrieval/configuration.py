"""Resolve representation sources and raw cache policies for retrieval experiments."""
import copy

from gbsae.data.tokens import source_names, token_policies
from gbsae.configuration import load_model_config


def representation_config(cfg, method):
    """Read the GB-SAE training recipe for shared or projected representations."""
    if method in {"gbsae", "projected"}:
        result = load_model_config(cfg["gbsae_config"])
    else:
        raise ValueError(f"Unsupported method: {method}")
    source_names(result)
    token_policies(result)
    return result


def method_sources(cfg, method):
    return tuple(source_names(representation_config(cfg, method)))


def raw_data_config(cfg, method, split_cfg):
    """Load raw prefix-processed tokens; providers own method-specific normalization.

    GB-SAE prefix/layout policies come exclusively from its training config.
    Downstream split settings override only paths, split, target class and cache
    storage format.
    """
    result = copy.deepcopy(representation_config(cfg, method))
    for key in ("imagenet_root", "activation_root", "split", "target_class", "combined_npz"):
        if key in split_cfg:
            result["data"][key] = split_cfg[key]
    result["data"].update(standardize=False, divide_norm=False)
    return result


def provider_metadata(cfg, method):
    """Track source/dimension/token settings so stale feature caches are rejected."""
    resolved = representation_config(cfg, method)
    result = {"models": resolved["models"], "token_preprocessing": resolved["data"]}
    if method in {"gbsae", "projected"}:
        result["model_settings"] = {"projector": resolved["projector"], "sae": resolved["sae"]}
    return result
