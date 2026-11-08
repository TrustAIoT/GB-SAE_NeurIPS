"""Train one linear classifier on fused shared representations.

This experiment owns shared reconstruction, token/source means and artifact
metadata. Generic training is provided by experiments.classification_head.
"""
from gbsae.config_cli import ConfigSchema, parse_config_arguments
from pathlib import Path

import torch
from einops import rearrange

from experiments.classification_head import CSVLogger, LinearProbe, TrainingConfig, fit, mean_pool_tokens, save_head
from experiments.classification_head.activation_data import load_inputs, make_activation_loader
from experiments.classification_head.cli import add_activation_fields, add_training_fields
from experiments.classification_head.gbsae_features import (
    build_frozen_projectors, build_frozen_sae, projected_tokens,
)
from gbsae.data.tokens import build_token_transform


@torch.no_grad()
def extract_features(batch_set, sources, projectors, sae, device, *, token_transform):
    """Reconstruct shared tokens, mean-pool tokens, then mean across models."""
    tokens = projected_tokens(batch_set, sources, projectors, device,
                              token_transform=token_transform)
    batch_size = tokens[sources[0]].shape[0]
    out = sae({s: rearrange(x, "b t d -> (b t) d") for s, x in tokens.items()})
    source_features = []
    for source in sources:
        z_sh = rearrange(out.z_sh[source], "(b t) k -> b t k", b=batch_size)
        source_features.append(mean_pool_tokens(z_sh @ out.D_sh))
    return torch.stack(source_features, dim=0).mean(dim=0)


def fused_checkpoint_path(save_dir, std_mode, epoch):
    return Path(save_dir) / "shared_mean" / f"probe_shared_mean_pool-mean_std-{std_mode}_epoch-{epoch}.pth"


def train_fused_head(cfg, ckpt, sources, *, options, save_dir, device,
                      batch_size=64, num_workers=16, std_mode="ckpt",
                      num_classes=None, save_every=10):
    """Assemble this experiment's feature pipeline and train exactly one head."""
    if save_every < 1:
        raise ValueError("save_every must be positive")
    loader = make_activation_loader(cfg, ckpt, sources, batch_size=batch_size,
                                    num_workers=num_workers, std_mode=std_mode, shuffle=True)
    projectors = build_frozen_projectors(cfg, ckpt, sources, device)
    sae = build_frozen_sae(cfg, ckpt, sources, device)
    transform = build_token_transform(cfg, sources)
    dim = cfg["projector"]["output_dim"]
    classes = num_classes or cfg["data"].get("num_classes", ckpt.get("num_classes", 1000))
    head = LinearProbe(dim, classes).to(device)
    logger = CSVLogger(fused_checkpoint_path(save_dir, std_mode, options.epochs).parent / "linear_probe_metrics.csv")
    metadata = dict(probe_type="shared_mean", pooling="mean", std_mode=std_mode,
                    projected_dim=dim, num_classes=classes, sources=list(sources), fusion="mean_across_sources")

    def on_epoch_end(epoch, metrics, trained_head):
        logger(epoch, metrics)
        if epoch % save_every == 0 or epoch == options.epochs:
            save_head(fused_checkpoint_path(save_dir, std_mode, epoch), trained_head,
                      epoch=epoch, metadata=metadata)

    return fit(head, loader, feature_fn=lambda batch: extract_features(batch, sources, projectors, sae, device, token_transform=transform),
                config=options, on_epoch_end=on_epoch_end)


def build_config_schema():
    parser = ConfigSchema(description=__doc__)
    add_activation_fields(parser)
    add_training_fields(parser)
    parser.add_field('save_dir', path='output.save_dir', type=str)
    return parser


def main(argv=None):
    args = parse_config_arguments(build_config_schema(),argv)
    cfg, ckpt, sources = load_inputs(args)
    options = TrainingConfig(epochs=args.epochs, lr=args.lr,
                             weight_decay=args.weight_decay, max_batches=args.max_batches)
    return train_fused_head(cfg, ckpt, sources, options=options, save_dir=args.save_dir,
                            device=args.device, batch_size=args.batch_size,
                            num_workers=args.num_workers, std_mode=args.std_mode,
                            num_classes=args.num_classes, save_every=args.save_every)


if __name__ == "__main__":
    main()
