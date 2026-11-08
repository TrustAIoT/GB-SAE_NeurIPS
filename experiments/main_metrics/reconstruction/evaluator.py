"""Projected/raw R-squared evaluation from batch_eval_sae_checkpoints.py.

Uses shuffled per-source loaders, frozen parameters and BatchNorm buffers,
current-batch encoder statistics, pseudoinverse back-projection, and the
arithmetic mean of per-batch R-squared values.
"""
from pathlib import Path
import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from gbsae.models.config import build_sae, restore_sae_state
from gbsae.models.config import build_projectors
from gbsae.training.logger import compute_r2
from gbsae.data.config import build_activation_dataset
from ..common import (align_activations, freeze_all,
                      linear_to_raw, model_names)
from .normalization import use_current_batch_statistics

def build_single_source_dataset(cfg, eval_source):
    return build_activation_dataset(cfg, [eval_source], standardize=cfg["data"].get("standardize", False))


def build_single_source_loader(cfg, dataset):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 8),
        pin_memory=True,
    )


def build_single_source_projectors(cfg, ckpt, eval_source, projector_from, device):
    linear_projectors = build_projectors(cfg, [eval_source], device)
    trained_projectors = ckpt["linear_projectors"].keys()
    for name, proj in linear_projectors.items():
        if name in trained_projectors:
            proj.load_state_dict(ckpt["linear_projectors"][name])
        elif projector_from is not None:
            src = projector_from
            if src not in trained_projectors:
                raise RuntimeError(
                    f"projector_from='{src}' not found in checkpoint, "
                    f"available: {list(trained_projectors)}"
                )
            proj.load_state_dict(ckpt["linear_projectors"][src])
        else:
            raise RuntimeError(
                f"No projector found for '{name}' in checkpoint, "
                f"and inputs.projector_from not specified. Available: {list(trained_projectors)}"
            )
        proj.to(device).eval()
    return linear_projectors


def build_reconstruction_sae(cfg, eval_source, ckpt, device):
    sae = build_sae(cfg, [eval_source], device)
    missing_keys, unexpected_keys = restore_sae_state(sae, ckpt["model_state_dict"])
    return sae, missing_keys, unexpected_keys


@torch.no_grad()
def evaluate_projected_reconstruction(cfg, ckpt, eval_source, projector_from, device):
    """Measure per-batch R-squared in standardized/projected feature space.
    
    Return total, shared, private and private-versus-(input - shared) scores.
    Encoder normalization uses current-batch statistics with fixed parameters
    and running buffers. Batch scores are averaged equally, using a shuffled,
    newly constructed single-source dataset.
    """
    dataset = build_single_source_dataset(cfg, eval_source)
    dataloader = build_single_source_loader(cfg, dataset)
    linear_projectors = build_single_source_projectors(
        cfg, ckpt, eval_source, projector_from, device
    )
    sae, _, _ = build_reconstruction_sae(cfg, eval_source, ckpt, device)
    freeze_all(sae)
    sae.eval()
    use_current_batch_statistics(sae.encoder)

    r2_total_list = []
    r2_shared_list = []
    r2_private_list = []
    r2_private_residual_list = []

    for _, batch in tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Linear eval [{eval_source}]",
        dynamic_ncols=True,
    ):
        batch_set, _ = batch
        proj_acts = {}

        for name in [eval_source]:
            act = batch_set[name].to(device, non_blocking=True)
            act = align_activations(cfg, act, name)
            act = rearrange(act, "b t d -> (b t) d")
            proj_acts[name] = linear_projectors[name](act)

        out = sae(proj_acts)
        x = proj_acts[eval_source]
        x_hat = out.recon[eval_source]
        x_hat_sh = sae.decoder_sh(out.z_sh[eval_source])
        x_hat_pr = sae.decoder_pr[eval_source](out.z_pr[eval_source])
        residual = x - x_hat_sh

        r2_total_list.append(compute_r2(x, x_hat))
        r2_shared_list.append(compute_r2(x, x_hat_sh))
        r2_private_list.append(compute_r2(x, x_hat_pr))
        r2_private_residual_list.append(compute_r2(residual, x_hat_pr))

    return {
        "r2_total": float(np.mean(r2_total_list)),
        "r2_shared": float(np.mean(r2_shared_list)),
        "r2_private": float(np.mean(r2_private_list)),
        "r2_private_residual": float(np.mean(r2_private_residual_list)),
    }


@torch.no_grad()
def evaluate_raw_reconstruction(cfg, ckpt, eval_source, projector_from, device):
    """Measure per-batch R-squared after pseudoinverse back-projection.
    
    Raw means the loader-transformed features before the projector, not restored
    pixel/raw-cache scale: this function does not invert standardization. Dataset
    statistics may be resampled independently from the projected evaluation.
    Encoder normalization uses current-batch statistics with fixed parameters
    and running buffers; batch scores are averaged equally.
    """
    dataset = build_single_source_dataset(cfg, eval_source)
    dataloader = build_single_source_loader(cfg, dataset)
    linear_projectors = build_single_source_projectors(
        cfg, ckpt, eval_source, projector_from, device
    )
    sae, _, _ = build_reconstruction_sae(cfg, eval_source, ckpt, device)
    freeze_all(sae)
    sae.eval()
    use_current_batch_statistics(sae.encoder)

    pinv_cache = {}
    for name, proj in linear_projectors.items():
        if hasattr(proj, "linear"):
            pinv_cache[name] = torch.linalg.pinv(proj.linear.weight)

    r2_raw_total_list = []
    r2_raw_shared_list = []
    r2_raw_private_list = []
    r2_raw_private_residual_list = []

    for _, batch in tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Raw eval [{eval_source}]",
        dynamic_ncols=True,
    ):
        batch_set, _ = batch
        proj_acts = {}
        raw_acts = {}

        for name in [eval_source]:
            act = batch_set[name].to(device, non_blocking=True)
            act = align_activations(cfg, act, name)
            act = rearrange(act, "b t d -> (b t) d")
            raw_acts[name] = act
            proj_acts[name] = linear_projectors[name](act)

        out = sae(proj_acts)

        x_raw = raw_acts[eval_source]
        x_hat_proj = out.recon[eval_source]
        x_hat_sh_proj = sae.decoder_sh(out.z_sh[eval_source])
        x_hat_pr_proj = sae.decoder_pr[eval_source](out.z_pr[eval_source])

        x_hat_raw = linear_to_raw(eval_source, x_hat_proj, linear_projectors, pinv_cache)
        x_hat_sh_raw = linear_to_raw(eval_source, x_hat_sh_proj, linear_projectors, pinv_cache)
        x_hat_pr_raw = linear_to_raw(eval_source, x_hat_pr_proj, linear_projectors, pinv_cache)
        residual_raw = x_raw - x_hat_sh_raw

        r2_raw_total_list.append(compute_r2(x_raw, x_hat_raw))
        r2_raw_shared_list.append(compute_r2(x_raw, x_hat_sh_raw))
        r2_raw_private_list.append(compute_r2(x_raw, x_hat_pr_raw))
        r2_raw_private_residual_list.append(compute_r2(residual_raw, x_hat_pr_raw))

    return {
        "r2_total": float(np.mean(r2_raw_total_list)),
        "r2_shared": float(np.mean(r2_raw_shared_list)),
        "r2_private": float(np.mean(r2_raw_private_list)),
        "r2_private_residual": float(np.mean(r2_raw_private_residual_list)),
    }


def build_reconstruction_rows(ckpt_path, linear_metrics, raw_metrics):
    rows = []
    for model_name in linear_metrics:
        if model_name not in linear_metrics or model_name not in raw_metrics:
            continue
        row = {
            "checkpoint_path": ckpt_path,
            "checkpoint_name": Path(ckpt_path).name,
            "model": model_name,
            "r2_linear_total": linear_metrics[model_name]["r2_total"],
            "r2_linear_shared": linear_metrics[model_name]["r2_shared"],
            "r2_linear_private": linear_metrics[model_name]["r2_private"],
            "r2_linear_private_residual": linear_metrics[model_name]["r2_private_residual"],
            "r2_raw_total": raw_metrics[model_name]["r2_total"],
            "r2_raw_shared": raw_metrics[model_name]["r2_shared"],
            "r2_raw_private": raw_metrics[model_name]["r2_private"],
            "r2_raw_private_residual": raw_metrics[model_name]["r2_private_residual"],
        }
        rows.append(row)
    return rows


def evaluate_reconstruction(cfg, ckpt, device="cpu", sources=None, projector_from=None):
    names = model_names(cfg)
    if sources is not None:
        missing = set(sources) - set(names)
        if missing:
            raise ValueError(f"Sources not present in training config: {sorted(missing)}")
        names = list(sources)
    linear_metrics, raw_metrics = {}, {}
    for name in names:
        linear_metrics[name] = evaluate_projected_reconstruction(cfg, ckpt, name, projector_from, device)
        raw_metrics[name] = evaluate_raw_reconstruction(cfg, ckpt, name, projector_from, device)
    return linear_metrics, raw_metrics
