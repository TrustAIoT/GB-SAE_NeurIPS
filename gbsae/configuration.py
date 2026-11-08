"""Resolve experiment inputs against an authoritative GB-SAE training recipe.

Configuration-file references are relative to the YAML containing them. Dataset,
checkpoint and output paths keep their usual working-directory interpretation.
An experiment may select other data, but cannot redefine the pretrained model's
architecture or preprocessing. Single-source adaptation explicitly defines a
new input stream while inheriting the shared SAE from its training recipe.
"""
import copy
import os
from pathlib import Path

import yaml


DATA_SELECTION_FIELDS = frozenset({
    "imagenet_root", "activation_root", "split", "target_class", "combined_npz",
})
MODEL_FIELDS = frozenset({"models", "projector", "sae"})


def read_yaml(path):
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def referenced_path(owner, reference):
    """Resolve a configuration reference beside its containing YAML."""
    path = Path(os.path.expandvars(os.path.expanduser(str(reference))))
    return str((Path(owner).resolve().parent / path).resolve())


def read_experiment_config(path, *, _parents=()):
    """Read experiment settings and resolve references without mixing namespaces."""
    path = str(Path(path).resolve())
    if path in _parents:
        raise ValueError(f"Circular experiment configuration reference: {path}")
    cfg = read_yaml(path)
    if "experiment_config" in cfg:
        reference = referenced_path(path, cfg["experiment_config"])
        base = read_experiment_config(reference, _parents=(*_parents, path))
        if MODEL_FIELDS <= base.keys() and "gbsae_config" not in base:
            raise ValueError("Use gbsae_config to reference a GB-SAE training recipe")
        cfg["experiment_config"] = reference
    else:
        base = {}
    if "gbsae_config" in cfg:
        cfg["gbsae_config"] = referenced_path(path, cfg["gbsae_config"])
    if "gbsae_configs" in cfg:
        cfg["gbsae_configs"] = {key: referenced_path(path, value) for key, value in cfg["gbsae_configs"].items()}
    for run in cfg.get("ours", []):
        if "config" in run:
            run["config"] = referenced_path(path, run["config"])
    return _merge_experiment_settings(base, cfg)


def _merge_experiment_settings(base, updates):
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_experiment_settings(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def select_data(cfg, selection=None):
    """Copy model settings and select another dataset without changing token policy."""
    selection = selection or {}
    invalid = set(selection) - DATA_SELECTION_FIELDS
    if invalid:
        raise ValueError(f"Experiment data cannot redefine GB-SAE preprocessing: {sorted(invalid)}")
    result = copy.deepcopy(cfg)
    result["data"].update(selection)
    return result


def _adapter_config(base, experiment):
    """Keep the shared architecture; configure only the new source and projector."""
    if "sae" in experiment:
        raise ValueError("Adapter SAE settings must come from gbsae_config")
    projector = experiment.get("projector", {})
    if "output_dim" in projector:
        raise ValueError("Adapter projector.output_dim must come from gbsae_config")
    cfg = copy.deepcopy(base)
    cfg["models"] = copy.deepcopy(experiment["models"])
    # New streams have their own feature layout/prefix/grid policies and inherit
    # the base normalization strategy.
    cfg["data"] = copy.deepcopy(base["data"])
    for key in ("prefix_tokens", "activation_layouts", "token_alignment"):
        cfg["data"].pop(key, None)
    cfg["data"].update(copy.deepcopy(experiment.get("data", {})))
    for key in ("standardize", "divide_norm"):
        if key in experiment.get("data", {}):
            raise ValueError(f"Adapter data.{key} must come from gbsae_config")
    cfg["projector"] = copy.deepcopy(projector)
    cfg["projector"]["output_dim"] = base["projector"]["output_dim"]
    for key in ("adapter", "run", "training", "loss", "optimizer", "schedule"):
        if key not in experiment:
            raise ValueError(f"Adapter config requires its own {key} settings")
        cfg[key] = copy.deepcopy(experiment[key])
    return cfg


def load_model_config(path, *, _parents=()):
    """Return model/data settings for a training recipe or a referring experiment.

    The full referenced recipe is read, including fields a given consumer does
    not use. Experiment optimizer/training/loader values stay in ``experiment``;
    they never replace GB-SAE training hyperparameters. Adapter optimization is
    independent and is assembled explicitly by the adapter branch.
    """
    path = str(Path(path).resolve())
    if path in _parents:
        raise ValueError(f"Circular GB-SAE configuration reference: {path}")
    experiment = read_experiment_config(path)
    if "gbsae_config" not in experiment:
        if not MODEL_FIELDS <= experiment.keys():
            raise ValueError(f"Config requires gbsae_config or a complete training model: {path}")
        return experiment
    base = load_model_config(experiment["gbsae_config"], _parents=(*_parents, path))
    if "adapter" in experiment and "models" in experiment:
        cfg = _adapter_config(base, experiment)
    else:
        invalid = MODEL_FIELDS & experiment.keys()
        if invalid:
            raise ValueError(f"Experiment must read model fields from gbsae_config: {sorted(invalid)}")
        cfg = select_data(base, experiment.get("data"))
        # Historical experiment-specific options remain available to their
        # callers without replacing any training-recipe field.
        for key, value in experiment.items():
            if key not in {"gbsae_config", "data", "training", "loss", "optimizer", "schedule", "run"}:
                cfg[key] = copy.deepcopy(value)
    cfg["experiment"] = experiment
    return cfg


def experiment_settings(cfg):
    """Return the experiment namespace from a resolved model config."""
    return cfg.get("experiment", {})
