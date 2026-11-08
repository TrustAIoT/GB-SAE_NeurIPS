"""Model, optimizer, scheduler and logger construction for configured GB-SAE runs."""
import torch
from gbsae.data.tokens import source_names
from gbsae.models.config import build_sae, build_projectors
from gbsae.nn.optimizer import CosineScheduler


def build_training_models(cfg: dict, device: str):
    """Construct source-ordered projections and the shared/private SAE."""
    names = source_names(cfg)
    projectors = build_projectors(cfg, names, device)
    sae = build_sae(cfg, names, device)
    return projectors, sae


def build_training_optimizer(cfg: dict, sae, total_steps: int):
    """Build AdamW/cosine schedule with the original SAE-only parameter group.

    Custom optimization can use the model builder and supply its own optimizer.
    SAE gradient clipping and scheduler timing remain in the existing training loop.
    """
    options = cfg["optimizer"]
    parameters = list(sae.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=options["lr"], weight_decay=options.get("weight_decay", 0.0))
    scheduler = CosineScheduler(
        optimizer, base_value=options["lr"], final_value=options.get("final_lr", 1e-6),
        total_iters=total_steps, warmup_iters=int(total_steps * options.get("warmup_pct", 0.0)),
        start_warmup_value=options.get("warmup_lr", 1e-6),
    )
    for group in optimizer.param_groups:
        group["lr"] = scheduler[0].item()
    return optimizer, scheduler


def build_training_logger(cfg: dict, projectors):
    """Build the complete trainer logger; callers can supply compatible custom loggers."""
    if cfg["run"].get("logger_type", "full") != "full":
        raise ValueError("Use logger_type: full or supply a compatible custom logger")
    from gbsae.training.logger import BottleneckSAELogger
    return BottleneckSAELogger(
        model_names=list(projectors), linear_projectors=projectors,
        granularity=cfg["run"].get("logger_granularity", 1), enable_wandb=cfg["run"].get("use_wandb", False),
    )
