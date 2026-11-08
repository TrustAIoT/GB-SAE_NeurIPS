#!/usr/bin/env python3
from gbsae.config_cli import ConfigSchema, parse_config_arguments
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm


from .data_utils import (
    CachedFeatureDataset,
    PKBatchSampler,
    canonicalize_train_val_datasets,
    sanity_check_image_activation_alignment,
    save_dataset_audit,
)
from .feature_providers import build_feature_provider
from .losses import build_heads, cross_view_metric_loss, encode_views
from .metrics import evaluate_method, write_evaluation_results
from .specs import get_method_spec
from .configuration import method_sources, raw_data_config, provider_metadata


def load_config(path: str) -> dict:
    from gbsae.configuration import read_experiment_config
    return read_experiment_config(path)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch/CUDA RNGs without enabling deterministic kernels.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU")
        return "cpu"
    return requested


def build_raw_dataset(cfg, method, split_cfg):
    from gbsae.data.config import build_activation_dataset
    return build_activation_dataset(raw_data_config(cfg, method, split_cfg))


def load_torch_file(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def atomic_torch_save(value, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(value, temporary_path)
    temporary_path.replace(output_path)


@torch.inference_mode()
def extract_feature_cache(
    dataset,
    provider,
    method: str,
    split_name: str,
    output_path: Path,
    batch_size: int,
    num_workers: int,
    cache_dtype: str,
) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=provider.device.type == "cuda",
    )
    buffers: Dict[str, list] = {}
    labels = []
    indices = []
    for batch_set, batch_labels, batch_indices in tqdm(
        loader, desc=f"Cache {method}/{split_name}", dynamic_ncols=True
    ):
        batch_features = provider.extract(batch_set)
        for view_name, values in batch_features.items():
            buffers.setdefault(view_name, []).append(values.detach().cpu())
        labels.append(batch_labels.long().cpu())
        indices.append(batch_indices.long().cpu())

    features = {name: torch.cat(parts, dim=0) for name, parts in buffers.items()}
    if cache_dtype == "float16":
        features = {name: values.half() for name, values in features.items()}
    elif cache_dtype != "float32":
        raise ValueError(f"Unsupported cache dtype: {cache_dtype}")
    labels_tensor = torch.cat(labels)
    indices_tensor = torch.cat(indices)
    expected_indices = torch.arange(len(dataset))
    if not torch.equal(indices_tensor, expected_indices):
        raise RuntimeError("Feature extraction changed dataset order")

    cache = {
        "version": 1,
        "method": method,
        "split": split_name,
        "features": features,
        "labels": labels_tensor,
        "sample_ids": dataset.sample_ids,
        "wnids": dataset.wnids,
        "view_dims": {name: int(values.shape[-1]) for name, values in features.items()},
        "checkpoint_paths": provider.checkpoint_paths,
        "provider_metadata": provider.cache_metadata,
        "pooling": provider.pooling,
        "cache_dtype": cache_dtype,
    }
    atomic_torch_save(cache, output_path)
    print(f"[OK] Saved {method}/{split_name} feature cache: {output_path}")
    return cache


def expected_checkpoint_paths(cfg: dict, method: str) -> Dict[str, str]:
    if method in {"projected", "gbsae"}:
        return {"gbsae": str(cfg["checkpoints"]["gbsae"])}
    raise ValueError(method)


def expected_provider_metadata(cfg: dict, method: str) -> Dict[str, object]:
    return provider_metadata(cfg, method)


def validate_cache(
    cache: Mapping[str, object],
    dataset,
    method: str,
    split_name: str,
    cfg: dict,
) -> None:
    """Check cached method/split, sample order, labels, checkpoint paths and protocol.
    
    Checkpoint paths are compared as strings; file contents/weight hashes are
    not verified. Replacing weights in place requires rebuilding the cache.
    """
    if cache.get("method") != method or cache.get("split") != split_name:
        raise RuntimeError(
            f"Cache metadata mismatch: expected {method}/{split_name}, "
            f"found {cache.get('method')}/{cache.get('split')}"
        )
    if cache.get("sample_ids") != dataset.sample_ids:
        raise RuntimeError(
            f"{method}/{split_name} cache sample order differs from the configured dataset; "
            "set execution.rebuild_cache: true and rerun"
        )
    cached_labels = cache.get("labels")
    expected_labels = torch.tensor(dataset.labels, dtype=torch.long)
    if not isinstance(cached_labels, torch.Tensor) or not torch.equal(
        cached_labels.long().cpu(), expected_labels
    ):
        raise RuntimeError(
            f"{method}/{split_name} cache labels differ from the canonical WNID mapping; "
            "set execution.rebuild_cache: true and rerun"
        )
    if cache.get("checkpoint_paths") != expected_checkpoint_paths(cfg, method):
        raise RuntimeError(
            f"{method}/{split_name} cache was built from different representation checkpoints; "
            "set execution.rebuild_cache: true and rerun"
        )
    if cache.get("provider_metadata", {}) != expected_provider_metadata(cfg, method):
        raise RuntimeError(
            f"{method}/{split_name} cache uses a different feature-extraction protocol; "
            "set execution.rebuild_cache: true and rerun"
        )
    if cache.get("pooling") != cfg["preprocessing"]["token_pooling"]:
        raise RuntimeError(
            f"{method}/{split_name} cache uses a different pooling mode; "
            "set execution.rebuild_cache: true and rerun"
        )


def get_or_build_cache(
    dataset,
    provider,
    method: str,
    split_name: str,
    cache_path: Path,
    cfg: dict,
    rebuild: bool,
) -> dict:
    if cache_path.is_file() and not rebuild:
        cache = load_torch_file(cache_path)
        validate_cache(cache, dataset, method, split_name, cfg)
        print(f"[OK] Loaded feature cache: {cache_path}")
        return cache
    if provider is None:
        raise RuntimeError(f"Cannot build missing cache without a {method} feature provider")
    return extract_feature_cache(
        dataset=dataset,
        provider=provider,
        method=method,
        split_name=split_name,
        output_path=cache_path,
        batch_size=int(cfg["cache"]["batch_size"]),
        num_workers=int(cfg["cache"]["num_workers"]),
        cache_dtype=cfg["cache"].get("dtype", "float32"),
    )


def cache_view_dims(cache: Mapping[str, object]) -> Dict[str, int]:
    return {name: int(values.shape[-1]) for name, values in cache["features"].items()}


@torch.inference_mode()
def encode_validation_cache(
    heads: torch.nn.ModuleDict,
    dataset: CachedFeatureDataset,
    head_by_view: Mapping[str, str],
    device: str,
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    heads.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    buffers: Dict[str, list] = {view: [] for view in head_by_view}
    for features, _, _ in loader:
        features = {view: values.to(device, non_blocking=True) for view, values in features.items()}
        embeddings = encode_views(heads, features, head_by_view)
        for view_name, values in embeddings.items():
            buffers[view_name].append(values.cpu())
    return {view: torch.cat(parts, dim=0) for view, parts in buffers.items()}


def save_head_checkpoint(
    output_path: Path,
    heads: torch.nn.ModuleDict,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    score: float,
    method: str,
    method_spec,
    view_dims: Mapping[str, int],
    experiment_cfg: dict,
    source_cache: Mapping[str, object],
) -> None:
    checkpoint = {
        "version": 1,
        "method": method,
        "epoch": int(epoch),
        "validation_category_mAP_macro": float(score),
        "head_state_dict": heads.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "method_spec": asdict(method_spec),
        "view_dims": dict(view_dims),
        "head_config": experiment_cfg["head"],
        "training_config": experiment_cfg["training"],
        "representation_checkpoint_paths": source_cache.get("checkpoint_paths", {}),
        "representation_provider_metadata": source_cache.get("provider_metadata", {}),
    }
    atomic_torch_save(checkpoint, output_path)


def write_history(path: Path, rows: list) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Validation fields first appear at eval_every. Build the CSV schema from
    # every row instead of freezing it to the non-validation fields in epoch 1.
    fields = []
    seen = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_and_save(
    heads,
    val_dataset,
    spec,
    cfg,
    device,
    output_dir,
    stem,
) -> dict:
    embeddings = encode_validation_cache(
        heads=heads,
        dataset=val_dataset,
        head_by_view=spec.head_by_view,
        device=device,
        batch_size=int(cfg["evaluation"]["embedding_batch_size"]),
    )
    metric_device = resolve_device(cfg["evaluation"].get("device", device))
    results = evaluate_method(
        embeddings=embeddings,
        labels=val_dataset.labels,
        spec=spec,
        ks=cfg["evaluation"]["ks"],
        device=metric_device,
        query_chunk_size=int(cfg["evaluation"]["query_chunk_size"]),
    )
    write_evaluation_results(output_dir, results, stem)
    return results


def train_heads(
    method: str,
    train_cache: Mapping[str, object],
    val_cache: Mapping[str, object],
    cfg: dict,
    output_dir: Path,
    device: str,
    eval_only: bool,
    head_checkpoint: Optional[str],
) -> None:
    """Train one method's tied/per-view retrieval heads on cached frozen features.
    
    Use PK batches, AdamW and an epoch cosine scheduler. Select best.pth by
    validation macro category mAP; last.pth records every epoch. eval_only
    loads a selected head checkpoint and writes metrics without optimizer steps.
    """
    spec = get_method_spec(method, method_sources(cfg, method))
    train_dataset = CachedFeatureDataset(train_cache)
    val_dataset = CachedFeatureDataset(val_cache)
    if train_dataset.features.keys() != val_dataset.features.keys():
        raise RuntimeError("Train and validation caches expose different views")
    if train_dataset.sample_ids == val_dataset.sample_ids:
        raise RuntimeError("Train and validation sample IDs are unexpectedly identical")

    view_dims = cache_view_dims(train_cache)
    val_view_dims = cache_view_dims(val_cache)
    if view_dims != val_view_dims:
        raise RuntimeError(f"Train/val feature dimensions differ: {view_dims} vs {val_view_dims}")
    missing_views = set(spec.views) - set(view_dims)
    if missing_views:
        raise RuntimeError(f"Feature cache is missing required views: {sorted(missing_views)}")

    head_cfg = cfg["head"]
    heads = build_heads(
        view_dims=view_dims,
        head_by_view=spec.head_by_view,
        output_dim=int(head_cfg["output_dim"]),
        head_type=head_cfg.get("type", "linear"),
        hidden_dim=int(head_cfg.get("hidden_dim", 512)),
        dropout=float(head_cfg.get("dropout", 0.0)),
    ).to(device)
    if len(heads) != spec.expected_head_count:
        raise RuntimeError(
            f"{method} should create {spec.expected_head_count} heads, created {len(heads)}"
        )

    if eval_only:
        checkpoint_path = Path(head_checkpoint) if head_checkpoint else output_dir / "best.pth"
        checkpoint = load_torch_file(checkpoint_path, map_location=device)
        if checkpoint.get("method_spec") != asdict(spec) or checkpoint.get("view_dims") != view_dims:
            raise ValueError("Retrieval head checkpoint sources or feature dimensions differ from the configured ensemble")
        if checkpoint.get("representation_provider_metadata") != train_cache.get("provider_metadata"):
            raise ValueError("Retrieval head checkpoint uses a different representation preprocessing configuration")
        heads.load_state_dict(checkpoint["head_state_dict"])
        results = evaluate_and_save(
            heads, val_dataset, spec, cfg, device, output_dir / "evaluation", "eval_only"
        )
        print(json.dumps(results["macro"], indent=2))
        return

    train_cfg = cfg["training"]
    sampler = PKBatchSampler(
        labels=train_dataset.labels.tolist(),
        classes_per_batch=int(train_cfg["classes_per_batch"]),
        samples_per_class=int(train_cfg["samples_per_class"]),
        batches_per_epoch=train_cfg.get("batches_per_epoch"),
        seed=int(cfg["seed"]),
    )
    loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
    )
    optimizer = torch.optim.AdamW(
        heads.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(train_cfg["epochs"])
    )

    history = []
    best_score = float("-inf")
    eval_every = int(train_cfg.get("eval_every", 1))
    epochs = int(train_cfg["epochs"])
    for epoch_index in range(epochs):
        sampler.set_epoch(epoch_index)
        heads.train()
        running = {"loss": 0.0, "instance_loss": 0.0, "category_loss": 0.0, "steps": 0}
        progress = tqdm(loader, desc=f"[{method}] epoch {epoch_index + 1}/{epochs}", dynamic_ncols=True)
        for features, labels, _ in progress:
            labels = labels.to(device, non_blocking=True)
            features = {
                view: values.to(device, non_blocking=True)
                for view, values in features.items()
                if view in spec.head_by_view
            }
            embeddings = encode_views(heads, features, spec.head_by_view)
            loss, loss_stats = cross_view_metric_loss(
                embeddings=embeddings,
                labels=labels,
                ordered_pairs=spec.train_pairs,
                temperature=float(train_cfg["temperature"]),
                instance_weight=float(train_cfg["instance_loss_weight"]),
                category_weight=float(train_cfg["category_loss_weight"]),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if train_cfg.get("gradient_clip_norm") is not None:
                torch.nn.utils.clip_grad_norm_(
                    heads.parameters(), float(train_cfg["gradient_clip_norm"])
                )
            optimizer.step()
            for key in ("loss", "instance_loss", "category_loss"):
                running[key] += loss_stats[key]
            running["steps"] += 1
            progress.set_postfix(loss=f"{loss_stats['loss']:.4f}")
        scheduler.step()

        row = {
            "epoch": epoch_index + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": running["loss"] / running["steps"],
            "train_instance_loss": running["instance_loss"] / running["steps"],
            "train_category_loss": running["category_loss"] / running["steps"],
        }
        should_evaluate = (epoch_index + 1) % eval_every == 0 or epoch_index + 1 == epochs
        score = float("nan")
        if should_evaluate:
            results = evaluate_and_save(
                heads=heads,
                val_dataset=val_dataset,
                spec=spec,
                cfg=cfg,
                device=device,
                output_dir=output_dir / "evaluation",
                stem=f"epoch_{epoch_index + 1:03d}",
            )
            score = float(results["macro"]["category_mAP"])
            row.update({f"val_{key}": value for key, value in results["macro"].items()})
            if score > best_score:
                best_score = score
                save_head_checkpoint(
                    output_dir / "best.pth",
                    heads,
                    optimizer,
                    epoch_index + 1,
                    score,
                    method,
                    spec,
                    view_dims,
                    cfg,
                    train_cache,
                )
        history.append(row)
        write_history(output_dir / "training_history.csv", history)
        save_head_checkpoint(
            output_dir / "last.pth",
            heads,
            optimizer,
            epoch_index + 1,
            score,
            method,
            spec,
            view_dims,
            cfg,
            train_cache,
        )
        print(f"[{method}] epoch={epoch_index + 1} train_loss={row['train_loss']:.5f} val_mAP={score}")

    best = load_torch_file(output_dir / "best.pth", map_location=device)
    heads.load_state_dict(best["head_state_dict"])
    final_results = evaluate_and_save(
        heads, val_dataset, spec, cfg, device, output_dir / "evaluation", "best"
    )
    print("[DONE] Best validation summary:")
    print(json.dumps(final_results["macro"], indent=2))
    if "all_hit" in final_results:
        print(json.dumps(final_results["all_hit"], indent=2))


def parse_args():
    parser = ConfigSchema(description='Train and evaluate retrieval heads for a configured source ensemble.')
    
    parser.add_field('method', path='method', choices=['projected', 'gbsae'], type=str)
    parser.add_field('rebuild_cache', path='execution.rebuild_cache', action='store_true')
    parser.add_field('cache_only', path='execution.cache_only', action='store_true')
    parser.add_field('eval_only', path='execution.eval_only', action='store_true')
    parser.add_field('head_checkpoint', path='execution.head_checkpoint', nullable=True, type=str)
    return parse_config_arguments(parser)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    method = args.method.lower()
    spec = get_method_spec(method, method_sources(cfg, method))
    device = resolve_device(cfg.get("device", "cuda"))

    output_root = Path(cfg["output_root"])
    run_name = str(cfg.get("run_name_by_method", {}).get(method, method))
    if Path(run_name).name != run_name:
        raise ValueError(f"run_name_by_method must contain simple directory names, got {run_name!r}")
    method_output = output_root / run_name
    cache_root = output_root / "feature_cache"
    print(f"[RUN] method={method} run_name={run_name} output={method_output}")
    method_output.mkdir(parents=True, exist_ok=True)
    with open(method_output / "resolved_config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    with open(method_output / "method_spec.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(spec), handle, indent=2)

    print("[INFO] Building train/validation activation datasets")
    raw_train = build_raw_dataset(cfg, method, cfg["data"]["train"])
    raw_val = build_raw_dataset(cfg, method, cfg["data"]["val"])
    train_dataset, val_dataset, audit = canonicalize_train_val_datasets(
        raw_train,
        raw_val,
        expected_train_classes=cfg["data"].get("expected_train_classes", 1000),
    )
    save_dataset_audit(audit, method_output / "dataset_audit.json")
    print(
        f"[DATA] train={audit.train_samples} images/{audit.train_classes} classes; "
        f"val={audit.val_samples} images/{audit.val_classes} classes; "
        f"missing-val-classes={len(audit.missing_train_wnids_in_val)}"
    )
    sanity_check_image_activation_alignment(
        train_dataset, int(cfg["data"]["alignment_checks"]), int(cfg["seed"])
    )
    sanity_check_image_activation_alignment(
        val_dataset, int(cfg["data"]["alignment_checks"]), int(cfg["seed"]) + 1
    )

    train_cache_path = cache_root / f"{run_name}_train.pt"
    val_cache_path = cache_root / f"{run_name}_val.pt"
    caches_exist = train_cache_path.is_file() and val_cache_path.is_file()

    provider = None
    if args.rebuild_cache or not caches_exist:
        provider = build_feature_provider(
            method=method,
            experiment_cfg=cfg,
            device=device,
            pooling=cfg["preprocessing"]["token_pooling"],
        )

    train_cache = get_or_build_cache(
        train_dataset,
        provider,
        method,
        "train",
        train_cache_path,
        cfg,
        args.rebuild_cache,
    )
    val_cache = get_or_build_cache(
        val_dataset,
        provider,
        method,
        "val",
        val_cache_path,
        cfg,
        args.rebuild_cache,
    )
    del provider
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if args.cache_only:
        print("[DONE] Feature caches built; execution.cache_only is enabled")
        return

    train_heads(
        method=method,
        train_cache=train_cache,
        val_cache=val_cache,
        cfg=cfg,
        output_dir=method_output,
        device=device,
        eval_only=args.eval_only,
        head_checkpoint=args.head_checkpoint,
    )


if __name__ == "__main__":
    main()
