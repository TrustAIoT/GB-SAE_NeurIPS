"""Configuration-driven token processing and batch-evaluator utilities."""
import copy
import csv
import glob
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from gbsae.data.tokens import build_token_transform, source_names

def load_config(path: str):
    from gbsae.configuration import load_model_config
    return load_model_config(path)


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def linear_to_raw(name, x_hat_proj, linear_projectors, pinv_cache):
    """Back-project with a cached/recomputed weight pseudoinverse.
    
    Identity projectors pass through. No inverse standardization or affine-bias
    correction is performed; the result is in the pre-projector feature space.
    """
    proj = linear_projectors[name]
    if not hasattr(proj, "linear"):
        return x_hat_proj
    if name in pinv_cache:
        return x_hat_proj @ pinv_cache[name].T
    with torch.no_grad():
        W_pinv = torch.linalg.pinv(proj.linear.weight)
    return x_hat_proj @ W_pinv.T


def resolve_checkpoints(ckpt_paths: Sequence[str], ckpt_globs: Sequence[str], ckpt_dir: Optional[str]):
    paths = []
    for p in ckpt_paths:
        if Path(p).is_file():
            paths.append(str(Path(p).resolve()))
    for pattern in ckpt_globs:
        paths.extend(str(Path(p).resolve()) for p in glob.glob(pattern, recursive=True))
    if ckpt_dir:
        ckpt_root = Path(ckpt_dir)
        for suffix in ("*.pth", "*.pt", "*.ckpt"):
            paths.extend(str(p.resolve()) for p in ckpt_root.rglob(suffix))
    unique_sorted = sorted(set(paths))
    if not unique_sorted:
        raise FileNotFoundError("No checkpoint files were found.")
    return unique_sorted


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def model_names(cfg):
    """Use the configured source order without backbone-specific dispatch."""
    return source_names(cfg)


def align_activations(cfg, act, name):
    """Apply the configured post-standardization patch interpolation."""
    return build_token_transform(cfg, [name])(name, act)


def load_checkpoint(path, device):
    # Original training checkpoints can include optimizer/Python metadata.
    return torch.load(path, map_location=device, weights_only=False)


def with_data_overrides(cfg, imagenet_root=None, activation_root=None, split=None):
    cfg = copy.deepcopy(cfg)
    for key, value in [("imagenet_root", imagenet_root),
                       ("activation_root", activation_root), ("split", split)]:
        if value is not None:
            cfg["data"][key] = value
    return cfg
