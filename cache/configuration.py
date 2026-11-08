"""Independent cache configuration and config-only command-line interface."""
import argparse
import copy
from pathlib import Path

import yaml


def validate_config(config: dict) -> dict:
    """Validate cache choices without reading a GB-SAE training configuration."""
    cfg = copy.deepcopy(config)
    if any(key in cfg for key in ("gbsae_config", "experiment_config")):
        raise ValueError("Cache settings are independent; specify them in this YAML")
    if cfg.get("action") not in {"extract", "combine", "inspect"}:
        raise ValueError("action must be extract, combine or inspect")
    models = cfg.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("models must be an ordered source-to-backbone mapping")
    for name, options in models.items():
        if not isinstance(name, str) or not name or any(c in name for c in '<>:"/\\|?*.') or name in {"activation", "label", "view_id"}:
            raise ValueError(f"Invalid source identifier: {name!r}")
        if not isinstance(options, dict) or not isinstance(options.get("type"), str):
            raise ValueError(f"models.{name}.type must select a registered backbone or factory")
        if options.get("storage_layout", "tokens") not in {"tokens", "channels_first", "channels_last"}:
            raise ValueError(f"Unknown storage_layout for {name}")
    if len({name.casefold() for name in models}) != len(models):
        raise ValueError("Source names must also be distinct on case-insensitive filesystems")
    output = cfg.get("output", {})
    if not output.get("root"):
        raise ValueError("Specify output.root")
    if output.get("mode", "per_source") not in {"per_source", "both", "combined"}:
        raise ValueError("output.mode must be per_source, both or combined")
    for field in ("compressed", "overwrite"):
        if field in output and type(output[field]) is not bool:
            raise ValueError(f"output.{field} must be a bool")
    if not cfg.get("data", {}).get("split"):
        raise ValueError("Specify data.split explicitly")
    split = cfg["data"]["split"]
    if not isinstance(split, str) or split in {".", ".."} or any(c in split for c in '/\\:'):
        raise ValueError("data.split must name one directory")
    augmentation = cfg.get("augmentation", {})
    if type(augmentation.get("enabled", False)) is not bool:
        raise ValueError("augmentation.enabled must be a bool")
    runtime = cfg.get("runtime", {})
    if type(runtime.get("seed", 42)) is not int or runtime.get("seed", 42) < 0:
        raise ValueError("runtime.seed must be a nonnegative integer")
    loader = cfg.get("loader", {})
    if type(loader.get("batch_size", 64)) is not int or loader.get("batch_size", 64) <= 0:
        raise ValueError("loader.batch_size must be positive")
    if type(loader.get("num_workers", 0)) is not int or loader.get("num_workers", 0) < 0:
        raise ValueError("loader.num_workers must be nonnegative")
    return cfg


def load_config(path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Cache YAML must contain a mapping")
    return validate_config(cfg)


def config_arguments(description: str, argv=None) -> dict:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Independent cache YAML; paths use the working directory.")
    return load_config(parser.parse_args(argv).config)
