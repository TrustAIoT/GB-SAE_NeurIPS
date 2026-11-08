"""The sole command-line parameter is --config; all run settings come from YAML."""
import argparse
import copy
from pathlib import Path

from .configuration import read_experiment_config, load_model_config, referenced_path
from .config_schema import ConfigError, ConfigSchema


_MISSING = object()


def _get(mapping, path):
    value = mapping
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return _MISSING
        value = value[key]
    return value


def load_config_arguments(schema, path):
    """Validate a job YAML into a namespace without executing its experiment."""
    path = str(Path(path).resolve())
    cfg = read_experiment_config(path)
    models = {}

    def inherited(spec):
        slot, field_path = spec
        if slot not in models:
            reference = cfg.get("gbsae_config") if slot == "base" else cfg.get("gbsae_configs", {}).get(slot)
            if reference is None:
                return _MISSING
            models[slot] = load_model_config(reference)
        return _get(models[slot], field_path)

    values = {"config": path}
    for field in schema.fields:
        value = _get(cfg, field.path)
        if value is _MISSING and field.inherit is not None:
            value = inherited(field.inherit)
        if value is _MISSING:
            raise ConfigError(f"{path}: missing configuration field {field.path}")
        if field.reference and value is not None:
            value = referenced_path(path, value)
        value = field.convert(copy.deepcopy(value))
        if field.name == "device" and value == "auto":
            import torch
            value = "cuda" if torch.cuda.is_available() else "cpu"
        values[field.name] = value
    return argparse.Namespace(**values)


def parse_config_arguments(schema, argv=None):
    """Parse --config only, then resolve and validate the declared YAML fields."""
    parser = argparse.ArgumentParser(prog=schema.prog, description=schema.description, allow_abbrev=False)
    parser.add_argument("--config", required=True, help="YAML containing all inputs and run settings.")
    args = parser.parse_args(argv)
    try:
        return load_config_arguments(schema, args.config)
    except (ConfigError, OSError, ValueError) as error:
        parser.error(str(error))
