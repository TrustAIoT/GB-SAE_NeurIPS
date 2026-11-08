"""Module for training Bottleneck SAE with Shared / Private TopK."""

from __future__ import annotations

import os
import random
import math
from typing import Callable, Dict, Optional
from collections import defaultdict

import torch
from tqdm import tqdm
from einops import rearrange
from gbsae.data.tokens import validate_token_correspondence

from gbsae.training.schedule import (
    TrainingScheduleConfig,
    TrainingPhaseController,
)

from gbsae.models.losses import (
    bottleneck_sae_total_loss,
    shared_latent_alignment_loss,
)


def save_checkpoint(
    model,
    linear_projectors: Dict[str, torch.nn.Module],
    optimizer,
    epoch: int,
    lr: float,
    dataset=None,
    path: str = "checkpoint.pth",
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "learning_rate": lr,
        "linear_projectors": {
            name: proj.state_dict()
            for name, proj in linear_projectors.items()
        },
    }

    if dataset is not None and hasattr(dataset, "standardization_stats"):
        checkpoint["standardization_stats"] = {
            name: {
                "mean": stats["mean"].cpu(),
                "std": stats["std"].cpu(),
            }
            for name, stats in dataset.standardization_stats.items()
        }

    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)

    torch.save(checkpoint, path)
    print(f"[Checkpoint saved] Epoch {epoch} -> {path}")


def train_bottleneck_sae(
    sae,
    linear_projectors: Dict[str, torch.nn.Module],
    dataloader,
    dataset,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[object] = None,
    nb_epochs: int = 20,
    enc_clip: float = 1.0,
    device: str = "cpu",
    model_name: str = "",
    lambda_sh_only: float = 0.0,
    lambda_shpr: float = 0.0,
    recon_loss: str = "l1",
    save_every: int = 10,
    save_dir: str = "./weights",
    seeded: bool = True,
    warmup_pct: float = 0.3,
    steady_pct: float = 0.7,
    lambda_gate_bin: float = 0.0,
    lambda_gate_ratio: float = 0.0,
    gate_ratio_target: float = 0.5,
    lambda_align: float = 0.0,
    logger=None,
    token_transform: Optional[Callable[[str, torch.Tensor], torch.Tensor]] = None,
):
    """Train the gated SAE on projected token features.
    
    The caller must supply a compatible logger; the default None is not handled.
    When seeded=True, Python and torch are seeded to 10 here, after model/data
    construction; NumPy is not seeded. A supplied token_transform(name, [B,T,D])
    aligns tokens before flattening/projection. None leaves token grids unchanged;
    there are no model-name-specific resize rules. Standardization is in the dataset.
    Per-source losses are summed, then pairwise shared-code alignment is added.
    Gate penalties use the phase controller's scale; the auxiliary loss stays on.
    Checkpoint epochs/names are zero-based, saved at epoch % save_every == 0
    and at the final epoch. Returns None.
    """
    if seeded:
        random.seed(10)
        torch.manual_seed(10)

    model_names = list(linear_projectors.keys())
    check_correspondence = lambda_align > 0.0 and len(model_names) > 1
    total_steps = nb_epochs * len(dataloader)
    schedule_cfg = TrainingScheduleConfig(
        warmup_pct=warmup_pct,
        steady_pct=steady_pct,
    )
    schedule = TrainingPhaseController(total_steps=total_steps, cfg=schedule_cfg)

    global_step = 0

    for epoch in range(nb_epochs):
        sae.train()

        for _, batch in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch+1}/{nb_epochs}",
            dynamic_ncols=True,
        ):
            batch_set, _ = batch
            proj_acts: Dict[str, torch.Tensor] = {}
            raw_acts: Dict[str, torch.Tensor] = {}
            token_acts: Dict[str, torch.Tensor] = {}

            for name in model_names:
                act = batch_set[name].to(device, non_blocking=True)
                if token_transform is not None:
                    act = token_transform(name, act)

                token_acts[name] = act
                act = rearrange(act, "b t d -> (b t) d")
                raw_acts[name] = act
                proj_acts[name] = linear_projectors[name](act)

            if check_correspondence:
                validate_token_correspondence(
                    token_acts, policies=getattr(token_transform, "policies", None),
                    context="Shared-code alignment",
                )

            state = schedule.get(global_step)
            gate_scale = state["gate_scale"]

            logger.start_step(
                aux_prob=1.0,
                lr=float(optimizer.param_groups[0]["lr"]),
                gate_scale=gate_scale,
            )

            out = sae(proj_acts)
            g = out.g

            step_losses = defaultdict(float)
            loss = 0.0
            x_hat_sh_dict: Dict[str, torch.Tensor] = {}
            orthogonality_dict: Dict[str, float] = {}

            for name in model_names:
                x = proj_acts[name]
                x_hat = out.recon[name]

                loss_i, extras = bottleneck_sae_total_loss(
                    x=x,
                    x_hat=x_hat,
                    pre_codes=out.pre_codes[name],
                    z_sh=out.z_sh[name],
                    z_pr=out.z_pr[name],
                    D_sh=out.D_sh,
                    D_pr=out.D_pr[name],
                    g=out.g,
                    penalty=0.1,
                    enable_aux=True,
                    lambda_sh_only=lambda_sh_only,
                    lambda_shsh=0.0,
                    lambda_shpr=lambda_shpr,
                    recon_loss=recon_loss,
                    lambda_gate_bin=lambda_gate_bin * gate_scale,
                    lambda_gate_ratio=lambda_gate_ratio * gate_scale,
                    gate_ratio_target=gate_ratio_target,
                )

                loss = loss + loss_i
                for k, v in extras.items():
                    if k.startswith("loss/"):
                        step_losses[k] += float(v)

                if "x_hat_sh" in extras and extras["x_hat_sh"] is not None:
                    x_hat_sh_dict[name] = extras["x_hat_sh"]
                if "orthogonality" in extras and extras["orthogonality"] is not None:
                    orthogonality_dict[name] = float(extras["orthogonality"].item())

            if (lambda_align > 0.0) and (len(out.z_sh) > 1):
                align, align_breakdown = shared_latent_alignment_loss(
                    out.z_sh,
                    return_breakdown=True,
                )
                loss = loss + lambda_align * align
                logger.log_shared_align(
                    float(align.item()),
                    support=float(align_breakdown["support"].item()),
                    mag=float(align_breakdown["mag"].item()),
                )
                step_losses["loss/align"] += float((lambda_align * align).detach())
            else:
                logger.log_shared_align(0.0, support=0.0, mag=0.0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if enc_clip is not None:
                M = len(model_names)
                clip_enc = enc_clip
                clip_sh = enc_clip
                clip_pr = enc_clip / math.sqrt(M)

                torch.nn.utils.clip_grad_norm_(sae.encoder.parameters(), clip_enc)
                torch.nn.utils.clip_grad_norm_(sae.decoder_sh.parameters(), clip_sh)

                for name in model_names:
                    torch.nn.utils.clip_grad_norm_(sae.decoder_pr[name].parameters(), clip_pr)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            logger.log_gate(g)
            for name in model_names:
                logger.log_step_model(
                    name=name,
                    x=proj_acts[name],
                    x_hat=out.recon[name],
                    x_hat_sh=x_hat_sh_dict.get(name, None),
                    act=raw_acts[name],
                    z_sh=out.z_sh[name],
                    z_pr=out.z_pr[name],
                    orthogonality=orthogonality_dict.get(name, None),
                )
            logger.log_step_losses(step_losses)
            logger.end_step(total_loss=float(loss.item()))

            global_step += 1

        _epoch_metrics = logger.end_epoch()

        if (epoch % save_every == 0) or (epoch + 1 == nb_epochs):
            ckpt_path = os.path.join(save_dir, f"{model_name}_epoch_{epoch}.pth")
            save_checkpoint(
                model=sae,
                linear_projectors=linear_projectors,
                optimizer=optimizer,
                epoch=epoch,
                lr=float(optimizer.param_groups[0]["lr"]),
                dataset=dataset,
                path=ckpt_path,
            )
