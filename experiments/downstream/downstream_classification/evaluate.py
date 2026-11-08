"""Evaluate a fused shared classifier or independent projected-feature classifiers.

Retains original WNID label mapping, exclusion of non-overlapping classes,
sample-weighted cross entropy and Acc@1/Acc@5. Evaluation accuracy is a fraction;
training CSVs use percentages. USAE/SPARC retain their native baseline evaluators.
"""
from gbsae.config_cli import ConfigSchema, parse_config_arguments
from experiments.configuration import add_data_fields, apply_data_arguments
import csv
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from gbsae.data.tokens import build_token_transform
from experiments.classification_head import LinearProbe
from experiments.classification_head.activation_data import build_eval_to_train_label_lut, load_config, make_activation_loader
from experiments.classification_head.gbsae_features import build_frozen_projectors, build_frozen_sae, freeze_all
from experiments.attribution.train_probe import extract_features as projected_features, projector_checkpoint_path
from .train_head import extract_features as fused_shared_features, fused_checkpoint_path


@torch.no_grad()
def evaluate_heads(args, cfg, ckpt):
    sources = args.sources
    device = args.device
    lut = build_eval_to_train_label_lut(
        cfg, sources[0], cfg["data"]["imagenet_root"], cfg["data"]["activation_root"],
        args.train_imagenet_root, args.train_activation_root,
        cfg["data"].get("split", "train"), args.train_split,
    )
    loader = make_activation_loader(cfg, ckpt, sources, batch_size=args.batch_size or cfg["training"]["batch_size"],
                                    num_workers=args.num_workers, std_mode=args.std_mode, shuffle=False)
    projectors = build_frozen_projectors(cfg, ckpt, sources, device)
    sae = build_frozen_sae(cfg, ckpt, sources, device) if args.method == "gbsae" else None
    transform = build_token_transform(cfg, sources)
    names = ["shared_mean"] if args.method == "gbsae" else sources
    heads = {}
    for name in names:
        probe_type = "shared_mean" if args.method == "gbsae" else "projector"
        path = (fused_checkpoint_path(args.probe_root, args.std_mode, args.probe_epoch) if args.method == "gbsae"
                else projector_checkpoint_path(args.probe_root, name, args.std_mode, args.probe_epoch))
        saved = torch.load(path, map_location=device, weights_only=False)
        if saved.get("pooling", "mean") != "mean":
            raise ValueError(f"Head requires incompatible pooling: {path}")
        if saved.get("std_mode", args.std_mode) != args.std_mode:
            raise ValueError(f"Head standardization mode differs: {path}")
        if args.method == "gbsae" and saved.get("sources", sources) != sources:
            raise ValueError("Fused head sources differ from the selected sources")
        classes = saved.get("num_classes", saved["probe_state_dict"]["linear.weight"].shape[0])
        head = LinearProbe(cfg["projector"]["output_dim"], classes).to(device)
        head.load_state_dict(saved["probe_state_dict"])
        head.eval()
        freeze_all(head)
        heads[name] = head
    stats = {name: dict(loss=0.0, correct1=0, correct5=0, total=0) for name in names}
    criterion = nn.CrossEntropyLoss()
    skipped = 0
    for step, (batch_set, targets) in enumerate(tqdm(loader, desc="Eval classification")):
        if args.max_batches is not None and step >= args.max_batches:
            break
        mapped = lut[targets]
        valid = mapped >= 0
        skipped += (~valid).sum().item()
        if not valid.any():
            continue
        targets = mapped[valid].to(device)
        batch_set = {s: x[valid] for s, x in batch_set.items()}
        if args.method == "gbsae":
            features = {"shared_mean": fused_shared_features(batch_set, sources, projectors, sae, device, token_transform=transform)}
        else:
            features = {source: projected_features(batch_set, source, projectors, device, token_transform=transform) for source in sources}
        for name in names:
            logits = heads[name](features[name])
            stat = stats[name]
            stat["loss"] += criterion(logits, targets).item() * targets.size(0)
            stat["correct1"] += (logits.argmax(1) == targets).sum().item()
            pred5 = logits.topk(k=min(5, logits.shape[1]), dim=1).indices
            stat["correct5"] += pred5.eq(targets[:, None]).any(1).sum().item()
            stat["total"] += targets.size(0)
    rows = []
    for name in names:
        stat = stats[name]
        if stat["total"] == 0:
            raise RuntimeError("No overlapping evaluation samples were found")
        rows.append(dict(method=args.method, source="fused" if args.method == "gbsae" else name,
                         probe_type="shared_mean" if args.method == "gbsae" else "projector",
                         epoch=args.probe_epoch, loss=stat["loss"]/stat["total"],
                         **{"acc@1": stat["correct1"]/stat["total"], "acc@5": stat["correct5"]/stat["total"]}))
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Skipped non-overlap samples: {skipped}")
    print(f"[DONE] Evaluation metrics saved to {output}")
    return rows


def main(argv=None):
    parser = ConfigSchema(description=__doc__)
    parser.add_field('method', path='method', choices=['gbsae', 'projected'], type=str)
    
    add_data_fields(parser)
    parser.add_field('ckpt', path='inputs.ckpt', type=str)
    parser.add_field('probe_root', path='inputs.probe_root', type=str)
    parser.add_field('probe_epoch', path='evaluation.probe_epoch', type=int)
    parser.add_field('sources', path='selection.sources', nargs='+', nullable=True, type=str)
    parser.add_field('std_mode', path='evaluation.std_mode', choices=['ckpt', 'recompute'], type=str)
    parser.add_field('batch_size', path='loader.batch_size', type=int, nullable=True)
    parser.add_field('num_workers', path='loader.num_workers', type=int)
    parser.add_field('max_batches', path='evaluation.max_batches', type=int, nullable=True)
    parser.add_field('output_csv', path='output.output_csv', type=str)
    parser.add_field('train_imagenet_root', path='reference_data.imagenet_root', inherit=('base', 'data.imagenet_root'), type=str)
    parser.add_field('train_activation_root', path='reference_data.activation_root', inherit=('base', 'data.activation_root'), type=str)
    parser.add_field('train_split', path='reference_data.split', inherit=('base', 'data.split'), type=str)
    parser.add_field('device', path='runtime.device', inherit=('base', 'runtime.device'), type=str)
    args = parse_config_arguments(parser,argv)
    cfg = apply_data_arguments(load_config(args.config), args)
    args.sources = list(cfg["models"]["sources"]) if args.sources is None else args.sources
    if not args.sources or len(set(args.sources)) != len(args.sources):
        raise ValueError("sources must be nonempty and unique")
    if any(s not in cfg["models"]["sources"] for s in args.sources):
        raise ValueError("Selected source is absent from the SAE configuration")
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    return evaluate_heads(args, cfg, ckpt)


if __name__ == "__main__":
    main()
