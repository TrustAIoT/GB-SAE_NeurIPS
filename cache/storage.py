"""Compatible NPZ paths, provenance sidecars and atomic cache commits."""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np


@contextmanager
def atomic_file(path, mode):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        kwargs = {"encoding": "utf-8", "newline": ""} if "b" not in mode else {}
        with os.fdopen(descriptor, mode, **kwargs) as handle:
            yield handle
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    with atomic_file(path, "w") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def sidecar_path(path):
    return Path(str(path) + ".json")


def activation_path(root, record, source=None):
    relative = Path(record.relative_path)
    suffix = f"{source}.npz" if source else "_combined.npz"
    return Path(root) / record.split / relative.parent / f"{relative.stem}{suffix}"


def recipe_id(recipe):
    """Identify extraction settings; this is unrelated to the image RNG seed."""
    return hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prepare_run(root, recipe, records, overwrite=False):
    path = Path(root) / "cache_metadata.json"
    identity = recipe_id(recipe)
    existing = read_json(path) if path.exists() else {}
    if existing and existing["recipe_id"] != identity and not overwrite:
        raise ValueError("Existing cache uses different extraction settings/images; use another output.root or explicit output.overwrite")
    descriptors = existing.get("extractors", {}) if existing.get("recipe_id") == identity else {}
    write_json(path, {"recipe_id": identity, "recipe": recipe, "images": len(records), "extractors": descriptors})
    return identity


def commit(path, arrays, record, sources, identity, compressed=True):
    """Commit arrays then their sidecar; an incomplete pair is regenerated."""
    with atomic_file(path, "wb") as handle:
        writer = np.savez_compressed if compressed else np.savez
        writer(handle, **arrays)
    write_json(sidecar_path(path), {
        "recipe_id": identity, "image": asdict(record), "sources": sources,
    })


def file_signature(path):
    """Detect replaced combination inputs without reading every activation byte."""
    path = Path(path)
    stat = path.stat()
    meta = sidecar_path(path)
    return [stat.st_size, stat.st_mtime_ns, meta.stat().st_mtime_ns if meta.exists() else None]


def reusable(path, record, sources, identity, overwrite=False, allow_extra_sources=False):
    """Check provenance and array structure before skipping any forward pass."""
    from .validation import validate_arrays
    if overwrite or not Path(path).exists() or not sidecar_path(path).exists():
        return False
    try:
        meta = read_json(sidecar_path(path))
    except (OSError, ValueError):
        return False
    if meta.get("recipe_id") != identity:
        return False
    source_match = set(sources) <= set(meta.get("sources", {})) if allow_extra_sources else set(sources) == set(meta.get("sources", {}))
    if meta.get("image") != asdict(record) or not source_match:
        raise ValueError(f"Cache image/source metadata mismatch: {path}")
    try:
        for source in meta["sources"].values():
            if "origin_signature" in source and file_signature(source["origin_path"]) != source["origin_signature"]:
                return False
        arrays = load_npz(path)
        validate_arrays(arrays, meta["sources"], record)
        return True
    except (OSError, EOFError, ValueError, KeyError, zipfile.BadZipFile):
        return False
