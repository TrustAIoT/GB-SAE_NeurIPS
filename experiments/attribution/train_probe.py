"""Train an independent linear classifier on each source's projected features.

Cached token activations pass through frozen projectors and mean pooling.
Each source has its own classifier, training metrics and checkpoints. The
classifiers can be reused for concept attribution or classification evaluation.

Modify extract_features to use another representation, or train_projector_heads
to change how data, classifiers and output callbacks are assembled. The shared
classification_head package supplies the reusable training components.
"""
from gbsae.config_cli import ConfigSchema, parse_config_arguments
from pathlib import Path

from experiments.classification_head import CSVLogger, LinearProbe, TrainingConfig, fit, mean_pool_tokens, save_head
from experiments.classification_head.activation_data import load_inputs, make_activation_loader
from experiments.classification_head.cli import add_activation_fields, add_training_fields
from experiments.classification_head.gbsae_features import build_frozen_projectors, projected_tokens
from gbsae.data.tokens import build_token_transform


def extract_features(batch_set, source, projectors, device, *, token_transform):
    """Return mean-pooled projected features for one backbone source."""
    tokens = projected_tokens(batch_set, [source], projectors, device,
                              token_transform=token_transform)
    return mean_pool_tokens(tokens[source])


def projector_checkpoint_path(save_dir, source, std_mode, epoch):
    """Locate a source's classifier checkpoint using its training metadata."""
    return Path(save_dir) / source / "projector" / f"probe_{source}_projector_pool-mean_std-{std_mode}_epoch-{epoch}.pth"


def train_projector_heads(cfg, ckpt, sources, *, options, save_dir, device,
                          batch_size=64, num_workers=16, std_mode="ckpt",
                          num_classes=None, save_every=10):
    """Train one linear head per source and return their epoch histories.

    Each source uses its own loader, frozen projector and classifier. Feature
    selection and checkpoint metadata are defined here; the reusable fit loop
    receives the assembled loader, head, feature function and output callback.
    """
    if save_every < 1:
        raise ValueError("save_every must be positive")
    histories = {}
    for source in sources:
        loader = make_activation_loader(cfg, ckpt, [source], batch_size=batch_size,
                                        num_workers=num_workers, std_mode=std_mode, shuffle=True)
        projectors = build_frozen_projectors(cfg, ckpt, [source], device)
        transform = build_token_transform(cfg, [source])
        dim = cfg["projector"]["output_dim"]
        classes = num_classes or cfg["data"].get("num_classes", ckpt.get("num_classes", 1000))
        head = LinearProbe(dim, classes).to(device)
        logger = CSVLogger(projector_checkpoint_path(save_dir, source, std_mode, options.epochs).parent / "linear_probe_metrics.csv")
        metadata = dict(source=source, probe_type="projector", pooling="mean",
                        std_mode=std_mode, projected_dim=dim, num_classes=classes)

        def on_epoch_end(epoch, metrics, trained_head):
            logger(epoch, metrics)
            if epoch % save_every == 0 or epoch == options.epochs:
                save_head(projector_checkpoint_path(save_dir, source, std_mode, epoch),
                          trained_head, epoch=epoch, metadata=metadata)

        histories[source] = fit(head, loader,
                                feature_fn=lambda batch: extract_features(batch, source, projectors, device, token_transform=transform),
                                config=options, on_epoch_end=on_epoch_end)
    return histories


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
    return train_projector_heads(cfg, ckpt, sources, options=options, save_dir=args.save_dir,
                                  device=args.device, batch_size=args.batch_size,
                                  num_workers=args.num_workers, std_mode=args.std_mode,
                                  num_classes=args.num_classes, save_every=args.save_every)


if __name__ == "__main__":
    main()
