"""One extraction loop for arbitrary source groups and three storage modes."""
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .augmentation import build_augmentation, generate_views
from .configuration import validate_config
from .data import ImageDataset, collate_images, export_images, select_images
from .features import build_extractor, extract_features
from .manifest import save_manifest, selection_fingerprint
from .storage import activation_path, commit, load_npz, prepare_run, read_json, reusable, sidecar_path, write_json
from .utils import assign_views, set_global_seed
from .validation import validate_arrays


def extract_cache(config: dict):
    """Extract native features; batch/source/view choices all come from cache YAML.

    Already valid artifacts are checked before loading a backbone. In both mode,
    missing combined files can be rebuilt from per-source caches without images
    or model inference. Keep the full source group when resuming so assignments
    remain unchanged, even if only one source is missing.
    """
    cfg = validate_config(config)
    sources, output = list(cfg["models"]), cfg["output"]
    root, mode = Path(output["root"]), output.get("mode", "per_source")
    overwrite, compressed = output.get("overwrite", False), output.get("compressed", True)
    seed = cfg.get("runtime", {}).get("seed", 42)
    set_global_seed(seed)
    records = select_images(cfg)
    filenames = set()
    for record in records:
        for name in sources:
            key = str(activation_path(root / "per_source", record, name)).casefold()
            if key in filenames:
                raise ValueError(f"Source/image names produce the same cache filename: {key}")
            filenames.add(key)
    recipe = {
        "format_version": 1, "models": cfg["models"], "source_order": sources,
        "augmentation": cfg.get("augmentation", {"enabled": False}), "seed": seed,
        "seed_identity": "filename", "data_root": str(Path(cfg["data"]["root"]).resolve()),
        "selection": selection_fingerprint(records, cfg["data"]["root"]),
        "versions": {name: version(name) for name in ("torch", "torchvision", "timm")},
    }
    identity = prepare_run(root, recipe, records, overwrite)
    export_images(cfg["data"], records, overwrite)
    save_manifest(root / "manifests" / f"{cfg['data']['split']}.csv", records)
    report = {"images": len(records), "per_source_written": 0, "combined_written": 0, "skipped_images": 0}
    pending, needs, combined_pending = [], {}, []
    for record in records:
        combined_path = activation_path(root / "combined", record)
        if mode == "combined":
            required = dict.fromkeys(sources, not reusable(combined_path, record, sources, identity, overwrite))
            if not any(required.values()):
                report["skipped_images"] += 1
                continue
        else:
            required = {
                name: not reusable(activation_path(root / "per_source", record, name), record, [name], identity, overwrite)
                for name in sources
            }
            combine_required = mode == "both" and (
                any(required.values()) or not reusable(combined_path, record, sources, identity, overwrite)
            )
            if combine_required:
                combined_pending.append(record)
            if not any(required.values()):
                if not combine_required:
                    report["skipped_images"] += 1
                continue
        pending.append(record)
        needs[record] = required
    if pending:
        _extract_pending(cfg, root, sources, pending, needs, identity, report)
    # Both mode completes the per-source stage before creating combined outputs.
    for record in combined_pending:
        arrays, metadata = _read_per_source(root, record, sources)
        validate_arrays(arrays, metadata, record)
        commit(activation_path(root / "combined", record), arrays, record, metadata, identity, compressed)
        report["combined_written"] += 1
    write_json(root / "extraction_report.json", report)
    return report


def _read_per_source(root, record, sources):
    arrays, metadata = {}, {}
    for name in sources:
        path = activation_path(root / "per_source", record, name)
        arrays[name] = load_npz(path)["activation"]
        metadata[name] = read_json(sidecar_path(path))["sources"][name]
    return arrays, metadata


def _extract_pending(cfg, root, sources, pending, needs, identity, report):
    runtime, output, loader = cfg.get("runtime", {}), cfg["output"], cfg.get("loader", {})
    device = runtime.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    seed, mode = runtime.get("seed", 42), output.get("mode", "per_source")
    enabled = cfg.get("augmentation", {}).get("enabled", False)
    augmenter = build_augmentation(cfg.get("augmentation", {}))
    batches = DataLoader(
        ImageDataset(cfg["data"]["root"], pending), batch_size=loader.get("batch_size", 64),
        num_workers=loader.get("num_workers", 0), shuffle=False, collate_fn=collate_images,
        pin_memory=False,
    )
    extractors = {}
    for images, records in tqdm(batches, desc="Cache features"):
        views = [generate_views(image, Path(record.relative_path).name, len(sources), seed, augmenter)
                 for image, record in zip(images, records)]
        assignments = [assign_views(Path(record.relative_path).name, sources, seed, enabled) for record in records]
        arrays = {record: {} for record in records}
        metadata = {record: {} for record in records}
        for name in sources:
            indices = [i for i, record in enumerate(records) if needs[record][name]]
            if indices:
                if name not in extractors:
                    extractors[name] = build_extractor(cfg["models"][name], device)
                    run_metadata = read_json(root / "cache_metadata.json")
                    run_metadata.setdefault("extractors", {})[name] = {
                        **extractors[name].provenance, "native_layout": extractors[name].native_layout,
                        "prefix_tokens": extractors[name].prefix_tokens,
                    }
                    write_json(root / "cache_metadata.json", run_metadata)
                extractor = extractors[name]
                inputs = torch.stack([extractor.preprocess(views[i][assignments[i][name]]) for i in indices]).to(device)
                features, descriptor = extract_features(extractor, inputs, cfg["models"][name])
                if features.shape[0] != len(indices):
                    raise ValueError(f"{name}: backbone changed the input batch size")
                for j, i in enumerate(indices):
                    record, view_id = records[i], assignments[i][name]
                    arrays[record][name] = features[j].numpy()
                    metadata[record][name] = {**descriptor, "view_id": view_id}
            for i, record in enumerate(records):
                if not needs[record][name]:
                    continue
                value = arrays[record][name]
                source_meta = {name: metadata[record][name]}
                artifact = {"activation": value, "label": np.array(record.label),
                            "view_id": np.array(assignments[i][name])}
                validate_arrays(artifact, source_meta, record)
                if mode in {"per_source", "both"}:
                    commit(activation_path(root / "per_source", record, name), artifact, record, source_meta,
                           identity, output.get("compressed", True))
                    report["per_source_written"] += 1
        if mode == "combined":
            for record in records:
                validate_arrays(arrays[record], metadata[record], record)
                commit(activation_path(root / "combined", record), arrays[record], record, metadata[record],
                       identity, output.get("compressed", True))
                report["combined_written"] += 1
