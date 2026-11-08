"""Validate source-specific cache structure, without cross-source grid resizing."""
import math
from pathlib import Path
import zipfile

import numpy as np

from .manifest import discover_records, load_manifest
from .storage import activation_path, load_npz, read_json, sidecar_path, write_json


def array_metadata(array, layout, prefix_tokens=0):
    if layout == "tokens" and array.ndim == 2:
        patches = array.shape[0] - prefix_tokens
        side = math.isqrt(max(0, patches))
        if patches <= 0 or side * side != patches:
            raise ValueError(f"Expected a square patch grid plus prefixes, got {array.shape}")
        height, width, dim = side, side, array.shape[-1]
    elif layout in {"channels_first", "channels_last"} and array.ndim == 3 and prefix_tokens == 0:
        height, width, dim = (array.shape[1], array.shape[2], array.shape[0]) if layout == "channels_first" else array.shape
        if height != width or height <= 0:
            raise ValueError(f"Expected a square spatial grid, got {array.shape}")
    else:
        raise ValueError(f"Layout {layout} with {prefix_tokens} prefixes does not match {array.shape}")
    if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise ValueError("Activations must be finite floating-point features")
    return {"layout": layout, "prefix_tokens": prefix_tokens, "height": height, "width": width,
            "feature_dim": dim, "shape": list(array.shape), "dtype": str(array.dtype)}


def validate_arrays(arrays, source_metadata, record):
    per_source = "activation" in arrays
    if per_source and len(source_metadata) != 1:
        raise ValueError("A per-source NPZ must describe exactly one source")
    if not per_source and set(arrays) != set(source_metadata):
        raise ValueError(f"Combined NPZ must contain source arrays only: {set(arrays)}")
    if per_source and int(arrays["label"]) != record.label:
        raise ValueError("Cached label differs from the image manifest")
    for name, expected in source_metadata.items():
        if "activation" not in arrays and name not in arrays:
            raise ValueError(f"Missing source array: {name}")
        value = arrays["activation"] if per_source else arrays[name]
        actual = array_metadata(value, expected["layout"], expected["prefix_tokens"])
        if any(actual[key] != expected[key] for key in actual):
            raise ValueError(f"Feature shape/layout metadata mismatch for {name}: {actual}")
        if per_source and "view_id" in arrays and expected.get("view_id") != int(arrays["view_id"]):
            raise ValueError(f"View metadata mismatch for {name}")


def inspect_cache(cfg):
    """Inspect selected sources and images; save an actionable missing/invalid report."""
    sources, split = list(cfg["models"]), cfg["data"]["split"]
    root = Path(cfg["output"]["root"])
    inputs = cfg.get("inputs", {})
    combined = inputs.get("combined_root")
    roots = inputs.get("source_roots", {name: str(root / "per_source") for name in sources})
    records = load_manifest(inputs["manifest"], split) if inputs.get("manifest") else discover_records(roots, sources, split, combined)
    report = {"images": len(records), "valid": 0, "issues": []}
    for record in records:
        paths = [activation_path(combined, record)] if combined else [activation_path(roots[name], record, name) for name in sources]
        valid = True
        for path in paths:
            try:
                meta = read_json(sidecar_path(path))
                if meta["image"] != {"split": record.split, "relative_path": record.relative_path, "label": record.label, "class_id": record.class_id}:
                    raise ValueError("Image identity mismatch")
                validate_arrays(load_npz(path), meta["sources"], record)
                required = set(sources) if combined else {next(name for name in sources if path == activation_path(roots[name], record, name))}
                if not required <= set(meta["sources"]):
                    raise ValueError("Required sources missing")
            except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
                valid = False
                report["issues"].append({"image": record.relative_path, "path": str(path), "reason": str(error)})
        report["valid"] += int(valid)
    write_json(root / "inspection.json", report)
    return report
