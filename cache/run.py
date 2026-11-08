"""Extract, combine or inspect backbone caches using one independent YAML."""
import json

from .configuration import config_arguments, validate_config


def run(config):
    cfg = validate_config(config)
    if cfg["action"] == "extract":
        from .pipeline import extract_cache
        return extract_cache(cfg)
    if cfg["action"] == "combine":
        from .combine import combine_cache
        return combine_cache(cfg)
    from .validation import inspect_cache
    return inspect_cache(cfg)


def print_report(report):
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cli(argv=None):
    report = run(config_arguments(__doc__, argv))
    print_report(report)
    if report.get("issues"):
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    cli()
