"""YAML field schema and entry orchestration for reconstruction/retrieval."""
from gbsae.config_cli import ConfigSchema, parse_config_arguments
import torch

from .common import load_config, model_names, resolve_checkpoints, with_data_overrides


def parse_args(argv=None):
    parser = ConfigSchema(description='Configuration-driven projected/raw reconstruction and PROJECTED/SHARED/PRIVATE retrieval.')
    
    parser.add_field('ckpt', path='inputs.ckpt', action='append', type=str)
    parser.add_field('ckpt_glob', path='inputs.ckpt_glob', action='append', type=str)
    parser.add_field('ckpt_dir', path='inputs.ckpt_dir', nullable=True, type=str)
    parser.add_field('output_dir', path='output.output_dir', type=str)
    parser.add_field('device', path='runtime.device', inherit=('base', 'runtime.device'), type=str)
    parser.add_field('metrics', path='evaluation.metrics', nargs='+', choices=['reconstruction', 'retrieval'], type=str)
    parser.add_field('sources', path='selection.sources', nargs='+', help='Optional source subset for reconstruction only.', nullable=True, type=str)
    parser.add_field('projector_from', path='inputs.projector_from', nullable=True, type=str)
    parser.add_field('imagenet_root', path='data.imagenet_root', help='ImageNet root override for both evaluations.', nullable=True, inherit=('base', 'data.imagenet_root'), type=str)
    parser.add_field('activation_root', path='data.activation_root', help='Combined-cache root override for both evaluations.', nullable=True, inherit=('base', 'data.activation_root'), type=str)
    parser.add_field('split', path='data.split', help="Split override; otherwise retain the config's data.split.", nullable=True, inherit=('base', 'data.split'), type=str)
    parser.add_field('batch_size', path='loader.batch_size', type=int, help='Batch size override for both evaluations.', nullable=True)
    parser.add_field('num_workers', path='loader.num_workers', type=int, help='Worker count override for both evaluations.', nullable=True)
    parser.add_field('retrieval_imagenet_root', path='retrieval_data.imagenet_root', nullable=True, type=str)
    parser.add_field('retrieval_activation_root', path='retrieval_data.activation_root', nullable=True, type=str)
    parser.add_field('retrieval_split', path='retrieval_data.split', nullable=True, type=str)
    parser.add_field('retrieval_batch_size', path='retrieval_loader.batch_size', type=int, nullable=True)
    parser.add_field('retrieval_num_workers', path='retrieval_loader.num_workers', type=int, nullable=True)
    parser.add_field('retrieval_max_images', path='evaluation.max_images', type=int, help='Optional diagnostic subset; by default evaluate the entire selected dataset.', nullable=True)
    parser.add_field('pool', path='evaluation.pool', choices=['mean', 'max', 'sum'], type=str)
    parser.add_field('shared_repr', path='evaluation.shared_repr', choices=['xhat_sh', 'z_sh'], type=str)
    parser.add_field('private_repr', path='evaluation.private_repr', choices=['xhat_pr', 'z_pr'], type=str)
    args = parse_config_arguments(parser,argv)
    for name in ["batch_size", "retrieval_batch_size", "retrieval_max_images"]:
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"{name} must be positive in the YAML")
    for name in ["num_workers", "retrieval_num_workers"]:
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"{name} must be nonnegative in the YAML")
    return args


def run_cli(argv=None):
    args = parse_args(argv)
    cfg = with_data_overrides(load_config(args.config), args.imagenet_root, args.activation_root, args.split)
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["training"]["num_workers"] = args.num_workers
    print(f"Sources: {model_names(cfg)}; token alignment: {cfg['data']['token_alignment']}")
    checkpoint_dir = args.ckpt_dir
    if not args.ckpt and not args.ckpt_glob and checkpoint_dir is None:
        checkpoint_dir = cfg.get("run", {}).get("save_dir")
        if checkpoint_dir is None:
            raise ValueError("Set inputs.ckpt, inputs.ckpt_glob or inputs.ckpt_dir, or supply the training run.save_dir")
    checkpoint_paths = resolve_checkpoints(args.ckpt, args.ckpt_glob, checkpoint_dir)
    device = args.device if torch.cuda.is_available() else "cpu"
    batch_size = args.retrieval_batch_size if args.retrieval_batch_size is not None else args.batch_size
    workers = args.retrieval_num_workers if args.retrieval_num_workers is not None else args.num_workers
    from .batch_checkpoint_eval import evaluate_checkpoints
    return evaluate_checkpoints(
        cfg, checkpoint_paths, args.output_dir, device=device, metrics=args.metrics,
        sources=args.sources, projector_from=args.projector_from,
        retrieval_imagenet_root=args.retrieval_imagenet_root,
        retrieval_activation_root=args.retrieval_activation_root,
        retrieval_split=args.retrieval_split,
        retrieval_batch_size=cfg["training"]["batch_size"] if batch_size is None else batch_size,
        retrieval_num_workers=cfg["training"]["num_workers"] if workers is None else workers,
        retrieval_max_images=args.retrieval_max_images,
        pool=args.pool, shared_repr=args.shared_repr, private_repr=args.private_repr,
    )
