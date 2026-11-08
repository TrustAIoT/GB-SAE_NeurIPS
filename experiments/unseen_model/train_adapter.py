"""Train one configured source's projector and private dictionary against a shared SAE."""
from gbsae.config_cli import ConfigSchema, parse_config_arguments
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from experiments.unseen_model.adapter import (
    build_adapter_dataset,
    build_adapter_modules,
    build_adapter_token_transform,
    configure_adapter_parameters,
    get_adapter_source,
)
from gbsae.nn.optimizer import CosineScheduler
from experiments.unseen_model.training import train_weighted_adapter


def load_config(path: str) -> dict:
    from gbsae.configuration import load_model_config
    return load_model_config(path)


def train_adapter(cfg: dict, device: str | None = None, logger=None) -> None:
    """Assemble a single-source adaptation run using the existing SAE training loop.

    Override logger with a compatible custom logger to inspect training. Model,
    dataset and token construction helpers are separately callable in adapter.py.
    Initialization and trainer seeding retain the original order. Checkpoints
    keep their existing fields; save the supplied YAML beside them for reuse.
    """
    source = get_adapter_source(cfg)
    token_transform = build_adapter_token_transform(cfg, source)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run = cfg["run"]
    if logger is None and run.get("logger_type", "full") != "full":
        raise ValueError("Use logger_type: full or pass a compatible custom logger")

    if run.get("use_wandb", False):
        import wandb

        wandb_dir = run.get("wandb_save_dir")
        if wandb_dir is not None:
            os.makedirs(wandb_dir, exist_ok=True)
        wandb.init(
            project=run["wandb_project"], name=run["name"], config=cfg,
            mode="online" if run.get("wandb_online", True) else "offline",
            dir=wandb_dir,
        )

    dataset = build_adapter_dataset(cfg, source)
    training = cfg["training"]
    dataloader = DataLoader(
        dataset, batch_size=training["batch_size"], shuffle=True,
        num_workers=training.get("num_workers", 8), pin_memory=True,
    )
    projectors, sae = build_adapter_modules(cfg, source, device)
    # Inspect preprocessing before optimization; invalid grids fail before training.
    token_transform(source, dataset[0][0][source].unsqueeze(0).to(device))
    parameters = configure_adapter_parameters(sae, projectors, source)
    print(f"[INFO] Source={source}; trainable parameters={sum(p.numel() for p in parameters):,}")
    optim_cfg = cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        parameters, lr=optim_cfg["lr"], weight_decay=optim_cfg.get("weight_decay", 0.0),
    )
    total_steps = training["epochs"] * len(dataloader)
    scheduler = CosineScheduler(
        optimizer, base_value=optim_cfg["lr"],
        final_value=optim_cfg.get("final_lr", 1e-6), total_iters=total_steps,
        warmup_iters=int(total_steps * optim_cfg.get("warmup_pct", 0.0)),
        start_warmup_value=optim_cfg.get("warmup_lr", 1e-6),
    )
    if logger is None:
        from gbsae.training.logger import BottleneckSAELogger
        from experiments.unseen_model.logger import CLIPTrainingLogger

        logger_class = CLIPTrainingLogger if source == "CLIP" else BottleneckSAELogger
        logger = logger_class(
            model_names=[source], linear_projectors=projectors,
            granularity=run.get("logger_granularity", 1),
            enable_wandb=run.get("use_wandb", False),
        )

    save_dir = Path(run.get("save_dir", "./weights"))
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "adapter_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.get("experiment", cfg), handle, sort_keys=False)
    loss, schedule = cfg["loss"], cfg["schedule"]
    train_weighted_adapter(
        lambda_total=loss.get("lambda_total", 1.0),
        sae=sae, linear_projectors=projectors, dataloader=dataloader, dataset=dataset,
        optimizer=optimizer, scheduler=scheduler, nb_epochs=training["epochs"],
        enc_clip=training.get("enc_clip", 1.0), device=device, model_name=run["name"],
        lambda_sh_only=loss.get("lambda_sh_only", 0.0),
        lambda_shpr=loss.get("lambda_shpr", 0.0), recon_loss=loss.get("recon_loss", "l1"),
        save_every=training.get("save_every", 10), save_dir=str(save_dir),
        seeded=training.get("seeded", True), warmup_pct=schedule.get("warmup_pct", 0.3),
        steady_pct=schedule.get("steady_pct", 0.7),
        lambda_gate_bin=loss.get("lambda_gate_bin", 0.0),
        lambda_gate_ratio=loss.get("lambda_gate_ratio", 0.0),
        gate_ratio_target=loss.get("gate_ratio_target", 0.5), lambda_align=0.0,
        logger=logger, token_transform=token_transform,
    )


def main():
    parser = ConfigSchema(description=__doc__)
    
    parser.add_field('device', path='runtime.device', help='Optional torch device, for example cpu or cuda:0.', nullable=True, inherit=('base', 'runtime.device'), type=str)
    args = parse_config_arguments(parser)
    train_adapter(load_config(args.config), device=args.device)


if __name__ == "__main__":
    main()
