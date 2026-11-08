"""Combine existing native caches or append sources to an existing combined set."""
from dataclasses import asdict
from pathlib import Path
import zipfile

from .configuration import validate_config
from .manifest import discover_records, load_manifest, save_manifest, selection_fingerprint
from .storage import activation_path, commit, file_signature, load_npz, prepare_run, read_json, reusable, sidecar_path, write_json
from .validation import array_metadata, validate_arrays


def combine_cache(config):
    """Combine only complete images; missing inputs are reported, never saved partially.

    Source roots are explicit. A manifest is recommended for legacy files; if it
    is omitted, discover the union of source identities. Legacy arrays without
    sidecars are accepted with explicit layout/prefix metadata in models, and
    their unknown extraction provenance remains marked as unknown.
    """
    cfg = validate_config(config)
    sources, inputs, output = list(cfg["models"]), cfg.get("inputs", {}), cfg["output"]
    roots = inputs.get("source_roots", {})
    combined_root = inputs.get("combined_root")
    split, root = cfg["data"]["split"], Path(output["root"])
    if combined_root and Path(combined_root).resolve() == (root / "combined").resolve():
        raise ValueError("Append sources into a different output.root to preserve the input combined cache")
    if not roots and not combined_root:
        raise ValueError("Combination requires inputs.source_roots or inputs.combined_root")
    records = load_manifest(inputs["manifest"], split) if inputs.get("manifest") else discover_records(roots, sources, split, combined_root)
    recipe = {"format_version": 1, "operation": "combine", "models": cfg["models"],
              "inputs": inputs, "selection": selection_fingerprint(records)}
    identity = prepare_run(root, recipe, records, output.get("overwrite", False))
    save_manifest(root / "manifests" / f"{split}.csv", records)
    report = {"images": len(records), "combined_written": 0, "skipped_images": 0, "issues": []}
    for record in records:
        path = activation_path(root / "combined", record)
        if reusable(path, record, sources, identity, output.get("overwrite", False), allow_extra_sources=True):
            report["skipped_images"] += 1
            continue
        try:
            arrays, metadata = {}, {}
            if combined_root:
                existing_path = activation_path(combined_root, record)
                if existing_path.exists():
                    existing = load_npz(existing_path)
                    existing_meta = _source_metadata(existing_path, existing, record, cfg)
                    arrays.update(existing)
                    metadata.update(existing_meta)
            for name in sources:
                if name in arrays:
                    if name in roots:
                        raise ValueError(f"Source {name} is supplied by both combined_root and source_roots; list only added sources")
                    continue
                if name not in roots:
                    raise ValueError(f"Missing input root for source {name}")
                source_path = activation_path(roots[name], record, name)
                value = load_npz(source_path)
                if "activation" not in value or "label" not in value or int(value["label"]) != record.label:
                    raise ValueError(f"Source array or label mismatch: {source_path}")
                source_meta = _source_metadata(source_path, value, record, cfg, name)
                arrays[name], metadata[name] = value["activation"], source_meta[name]
            # Retain extra arrays from the existing combined file and their metadata.
            validate_arrays(arrays, metadata, record)
            commit(path, arrays, record, metadata, identity, output.get("compressed", True))
            report["combined_written"] += 1
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
            report["issues"].append({"image": record.relative_path, "reason": str(error)})
    write_json(root / "combination_report.json", report)
    return report


def _source_metadata(path, arrays, record, cfg, source=None):
    if sidecar_path(path).exists():
        meta = read_json(sidecar_path(path))
        if meta["image"] != asdict(record):
            raise ValueError(f"Image identity mismatch: {path}")
        validate_arrays(arrays, meta["sources"], record)
        return {name: {**value, "origin_recipe_id": meta["recipe_id"], "origin_path": str(path.resolve()), "origin_signature": file_signature(path)}
                for name, value in meta["sources"].items()}
    names = [source] if source else list(arrays)
    metadata = {}
    for name in names:
        options = cfg["models"].get(name)
        if options is None or "prefix_tokens" not in options:
            raise ValueError(f"Legacy cache {path} needs explicit models.{name}.prefix_tokens and storage_layout")
        value = arrays["activation"] if source else arrays[name]
        descriptor = array_metadata(value, options.get("storage_layout", "tokens"), options["prefix_tokens"])
        if "feature_dim" in options and descriptor["feature_dim"] != options["feature_dim"]:
            raise ValueError(f"Feature dimension mismatch for {name}")
        metadata[name] = {**descriptor, "view_id": int(arrays["view_id"]) if source and "view_id" in arrays else None,
                          "origin_path": str(path.resolve()), "origin_signature": file_signature(path),
                          "origin_recipe_id": None, "provenance": "unknown"}
    return metadata


def cli(argv=None):
    from .configuration import config_arguments
    from .run import print_report
    cfg = config_arguments(__doc__, argv)
    if cfg["action"] != "combine":
        raise ValueError("cache.combine requires action: combine")
    report = combine_cache(cfg)
    print_report(report)
    if report["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
