"""Train GB-SAE with explicit source/prefix/grid settings from one configuration."""
from gbsae.config_cli import ConfigSchema, parse_config_arguments
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import yaml
from gbsae.data.config import build_activation_dataset
from gbsae.data.tokens import build_token_transform, source_names, validate_token_correspondence
from gbsae.training.components import build_training_logger, build_training_models, build_training_optimizer
from gbsae.training.train import train_bottleneck_sae


def load_config(path: str) -> dict:
    from gbsae.configuration import load_model_config
    return load_model_config(path)


def train(cfg: dict, device: str | None = None, logger=None) -> None:
    """Assemble the shared training pipeline; keep scientific choices in cfg.

    Cached features are prefix-processed and normalized in their original grid.
    Interpolation happens only before projection in the training loop. Custom
    loops can call construction helpers and train_bottleneck_sae separately.
    """
    names = source_names(cfg)
    transform = build_token_transform(cfg)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run, training, loss, schedule = cfg["run"], cfg["training"], cfg["loss"], cfg["schedule"]
    if run.get("use_wandb", False):
        import wandb
        directory = run.get("wandb_save_dir")
        if directory is not None:
            os.makedirs(directory, exist_ok=True)
        wandb.init(project=run["wandb_project"], name=run["name"], config=cfg,
                   mode="online" if run.get("wandb_online", True) else "offline", dir=directory)
    dataset = build_activation_dataset(cfg)
    if not len(dataset):
        raise ValueError("The training dataset is empty")
    sample, _ = dataset[0]
    aligned_sample = {name: transform(name, sample[name].unsqueeze(0)) for name in names}
    if loss.get("lambda_align", 0.0) > 0 and len(names) > 1:
        validate_token_correspondence(
            aligned_sample, policies=transform.policies, context="Shared-code alignment",
        )
    dataloader = DataLoader(dataset, batch_size=training["batch_size"], shuffle=True,
                            num_workers=training.get("num_workers", 8), pin_memory=True)
    projectors, sae = build_training_models(cfg, device)
    optimizer, scheduler = build_training_optimizer(cfg, sae, training["epochs"] * len(dataloader))
    if logger is None:
        logger = build_training_logger(cfg, projectors)
    save_dir = Path(run.get("save_dir", "./weights"))
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "training_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    train_bottleneck_sae(
        sae=sae, linear_projectors=projectors, dataloader=dataloader, dataset=dataset,
        optimizer=optimizer, scheduler=scheduler, nb_epochs=training["epochs"],
        enc_clip=training.get("enc_clip", 1.0), device=device, model_name=run["name"],
        lambda_sh_only=loss.get("lambda_sh_only", 0.0), lambda_shpr=loss.get("lambda_shpr", 0.0),
        recon_loss=loss.get("recon_loss", "l1"), save_every=training.get("save_every", 10),
        save_dir=str(save_dir), seeded=training.get("seeded", True),
        warmup_pct=schedule.get("warmup_pct", 0.3), steady_pct=schedule.get("steady_pct", 0.7),
        lambda_gate_bin=loss.get("lambda_gate_bin", 0.0), lambda_gate_ratio=loss.get("lambda_gate_ratio", 0.0),
        gate_ratio_target=loss.get("gate_ratio_target", 0.5), lambda_align=loss.get("lambda_align", 0.0),
        logger=logger, token_transform=transform,
    )


def main(config_path: str, device: str | None = None) -> None:
    train(load_config(config_path), device=device)


def cli():
    parser = ConfigSchema(description=__doc__)
    
    parser.add_field('device', path='runtime.device', help='Optional torch device such as cpu or cuda:0.', nullable=True, inherit=('base', 'runtime.device'), type=str)
    args = parse_config_arguments(parser)
    main(args.config, device=args.device)


if __name__ == "__main__":
    cli()
