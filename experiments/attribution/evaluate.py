"""Frozen branch attribution with a uniform random orthogonal baseline."""
import os
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from einops import rearrange

from gbsae.config_cli import ConfigSchema, parse_config_arguments
from experiments.configuration import add_data_fields, apply_data_arguments
from gbsae.data.config import build_activation_dataset
from gbsae.data.tokens import build_token_transform
from gbsae.models.config import build_sae, restore_sae_state, configured_sources
from gbsae.models.config import build_projectors
from experiments.classification_head import LinearProbe, mean_pool_tokens
from experiments.classification_head.activation_data import build_eval_to_train_label_lut
from experiments.main_metrics.reconstruction.normalization import use_current_batch_statistics
from .interventions import evaluate_concept_interventions
from .random_baseline import orthogonal_basis, sample_indices, evaluate_random_interventions


GROUPS = ("sh", "pr", "total")


def seed_evaluation(seed):
    """Reset each source's loader/model RNGs to the same evaluation seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_config(path: str):
    from gbsae.configuration import load_model_config
    return load_model_config(path)


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def find_probe_path(probe_root: str, source: str, probe_match=None):
    candidates = []
    probe_match = probe_match or []

    source_root = os.path.join(probe_root, source)
    search_roots = []
    if os.path.isdir(source_root):
        search_roots.append(source_root)
    if os.path.isdir(probe_root):
        search_roots.append(probe_root)

    for root in search_roots:
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.endswith(".pth") and filename.startswith(f"probe_{source}"):
                    path = os.path.join(dirpath, filename)
                    normalized_path = path.replace("\\", "/")
                    if all(token in normalized_path for token in probe_match):
                        candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No probe checkpoint for {source} found under probe_root: {probe_root} "
            f"with probe_match={probe_match}"
        )

    candidates = sorted(set(candidates))
    if len(candidates) > 1:
        raise RuntimeError(
            "Multiple probe checkpoints matched. Set a more specific inputs.probe_root, "
            "or add inputs.probe_match filters such as projector and epoch-50.\n"
            + "\n".join(candidates)
        )

    print(f"[INFO] Using {source} probe: {candidates[0]}")
    return candidates[0]


def configure_dataset_standardization(dataset, ckpt, source: str, std_mode: str):
    """Select checkpoint statistics or retain statistics computed by the dataset."""
    if not dataset.standardize:
        return
    if std_mode != "ckpt":
        print(f"[INFO] Using dataset-computed standardization stats for {source}")
        return

    ckpt_std = ckpt.get("standardization_stats", None)
    assert ckpt_std is not None, "Checkpoint has no standardization_stats"
    assert source in ckpt_std, (
        f"std stats for {source} not found in checkpoint, "
        f"available: {list(ckpt_std.keys())}"
    )
    dataset.standardization_stats = {
        source: {
            "mean": ckpt_std[source]["mean"].detach().cpu(),
            "std": ckpt_std[source]["std"].detach().cpu(),
        }
    }
    print(f"[INFO] Using checkpoint standardization stats for {source}")


def build_config_schema():
    parser = ConfigSchema(description=__doc__)

    add_data_fields(parser)
    parser.add_field('batch_size', path='loader.batch_size', type=int, nullable=True)
    parser.add_field('num_workers', path='loader.num_workers', type=int, nullable=True)
    parser.add_field('ckpt', path='inputs.ckpt', type=str)
    parser.add_field('sources', path='selection.sources', type=str, nargs='+', nullable=True)
    parser.add_field('probe_root', path='inputs.probe_root', type=str)
    parser.add_field('probe_match', path='evaluation.probe_match', type=str, nargs='*', help='Optional substrings that must appear in the probe checkpoint path. This only filters probe loading; mean pooling is fixed.', nullable=True)
    parser.add_field('probe_std_mode', path='evaluation.probe_std_mode', type=str, choices=['ckpt', 'recompute'])
    parser.add_field('max_batches', path='evaluation.max_batches', type=int, nullable=True)
    parser.add_field('agreement_k', path='evaluation.prediction_agreement_k', type=int)
    parser.add_field('seed', path='runtime.seed', type=int)
    parser.add_field('random_seeds', path='random_baseline.seeds', type=int, nargs='+')
    parser.add_field('mask_seed_offset', path='random_baseline.mask_seed_offset', type=int)
    parser.add_field('output_csv', path='output.output_csv', type=str)

    # Probe-training dataset used to map evaluation labels to training labels.
    parser.add_field('check_imagenet_root', path='reference_data.imagenet_root', type=str, inherit=('base', 'data.imagenet_root'))
    parser.add_field('check_activation_root', path='reference_data.activation_root', type=str, inherit=('base', 'data.activation_root'))
    parser.add_field('check_split', path='reference_data.split', type=str, help='Split used by the check/reference dataset for label mapping.', inherit=('base', 'data.split'))

    parser.add_field('device', path='runtime.device', nullable=True, inherit=('base', 'runtime.device'), type=str)
    return parser


@torch.no_grad()
def main(argv=None):
    args = parse_config_arguments(build_config_schema(), argv)
    if len(args.random_seeds) < 2 or len(set(args.random_seeds)) != len(args.random_seeds):
        raise ValueError("random_baseline.seeds must contain at least two distinct seeds")
    if not 0 <= args.seed < 2**32 or min(args.random_seeds) < 0 or args.mask_seed_offset < 0:
        raise ValueError("Seeds/offset must be nonnegative; runtime.seed must be less than 2**32")
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("evaluation.max_batches must be positive or null")

    cfg = apply_data_arguments(load_config(args.config), args)
    args.sources = configured_sources(cfg, args.sources)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    all_results, random_rows, agreement_rows, agreement_repeat_rows = [], [], [], []
    source_metadata = {}
    # Build one WNID-based label mapping shared by all backbone sources.
    print("[INFO] Building eval train label mapping")

    label_lut = build_eval_to_train_label_lut(
        cfg=cfg,
        source=args.sources[0],
        eval_root=cfg["data"]["imagenet_root"],
        eval_act_root=cfg["data"]["activation_root"],
        train_root=args.check_imagenet_root,
        train_act_root=args.check_activation_root,
        eval_split=cfg["data"].get("split", "train"),
        train_split=args.check_split,
    )

    if torch.all(label_lut[label_lut >= 0] == torch.arange(len(label_lut))[label_lut >= 0]):
        print("[LABEL CHECK] Eval labels already aligned with train labels (identity mapping).")
    else:
        print("[LABEL CHECK] Eval labels NOT aligned; applying eval train label mapping.")
    for source in args.sources:
        print(f"\n================ Evaluating {source} ================\n")
        seed_evaluation(args.seed)

        dataset = build_activation_dataset(cfg, sources=[source])
        transform = build_token_transform(cfg, [source])
        configure_dataset_standardization(dataset, ckpt, source, args.probe_std_mode)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size if args.batch_size is not None else cfg["training"]["batch_size"],
            shuffle=True,
            num_workers=args.num_workers if args.num_workers is not None else cfg["training"].get("num_workers", 8),
            pin_memory=True,
            drop_last=True,
        )

        proj_cfg = cfg["projector"]
        proj = build_projectors(cfg, [source], device)[source]

        proj.load_state_dict(ckpt["linear_projectors"][source])
        proj.to(device).eval()
        freeze_all(proj)

        sae_cfg = cfg["sae"]
        sae = build_sae(cfg, [source], device)
        eval_ks = list(range(1, sae_cfg["k"] + 1))
        if args.agreement_k not in eval_ks:
            raise ValueError("evaluation.prediction_agreement_k must lie in 1..sae.k")
        if max(eval_ks) > proj_cfg["output_dim"]:
            raise ValueError("sae.k must not exceed the random orthogonal basis dimension")


        restore_sae_state(sae, ckpt["model_state_dict"])
        freeze_all(sae)
        # Normalize this source with current-batch encoder statistics.
        sae.eval()
        use_current_batch_statistics(sae.encoder)

        probe_ckpt = torch.load(
            find_probe_path(args.probe_root, source, args.probe_match),
            map_location=device, weights_only=False,
        )
        if probe_ckpt.get("pooling", "mean") != "mean":
            raise ValueError("Attribution requires a mean-pooled projector probe")
        classes = probe_ckpt.get("num_classes", probe_ckpt["probe_state_dict"]["linear.weight"].shape[0])
        probe = LinearProbe(proj_cfg["output_dim"], classes).to(device)
        probe.load_state_dict(probe_ckpt["probe_state_dict"])
        probe.eval()
        freeze_all(probe)

        banks = [orthogonal_basis(proj_cfg["output_dim"], seed).to(device)
                 for seed in args.random_seeds]
        generators = [torch.Generator(device="cpu").manual_seed(seed + args.mask_seed_offset)
                      for seed in args.random_seeds]
        metrics = {g: np.zeros((2, len(eval_ks)), dtype=np.float64) for g in GROUPS}
        learned_counts = {g: np.zeros(2, dtype=np.int64) for g in GROUPS}
        random_sums = np.zeros((len(banks), 2, len(eval_ks)), dtype=np.float64)
        random_counts = np.zeros((len(banks), 2), dtype=np.int64)
        batches = images = 0
        for step, (batch_set, targets) in enumerate(tqdm(dataloader, desc=f"Eval {source}")):

            if args.max_batches and step >= args.max_batches:
                break

            targets = label_lut[targets].to(device)

            # Verify label mapping before scoring the first batch.
            if step == 0:
                assert (targets >= 0).all(), "[FATAL] Unmapped eval labels detected"
                print("[CHECK] mapped targets:",
                      targets.min().item(),
                      targets.max().item(),
                      targets.unique().numel())

            B = targets.size(0)

            act = batch_set[source].to(device)
            act = transform(source, act)

            act = rearrange(act, "b t d -> (b t) d")
            proj_act = proj(act)

            out = sae({source: proj_act})

            z_sh_tokens = rearrange(out.z_sh[source], "(b t) k -> b t k", b=B)
            z_pr_tokens = rearrange(out.z_pr[source], "(b t) k -> b t k", b=B)
            z_sh = mean_pool_tokens(z_sh_tokens)
            z_pr = mean_pool_tokens(z_pr_tokens)

            D_sh = out.D_sh
            D_pr = out.D_pr[source]

            w_y = probe.linear.weight[targets]

            contrib_sh = z_sh * (D_sh @ w_y.T).T
            contrib_pr = z_pr * (D_pr @ w_y.T).T
            contrib_total = contrib_sh + contrib_pr

            # Each ranking signal selects contributions from its own group.
            for name, contrib in [
                ("sh", contrib_sh),
                ("pr", contrib_pr),
                ("total", contrib_total),
            ]:
                res = evaluate_concept_interventions(
                    z_sh, z_pr, D_sh, D_pr, contrib, probe, targets,
                    group=name, ks=eval_ks, agreement_k=args.agreement_k,
                )
                metrics[name] += res.scores.mean(dim=1).cpu().numpy()
                learned_counts[name] += (res.remove_matches, res.keep_matches)

            rep_full = z_sh @ D_sh + z_pr @ D_pr
            for repeat, (bank, generator) in enumerate(zip(banks, generators)):
                indices = sample_indices(B, rep_full.shape[1], max(eval_ks), generator)
                res = evaluate_random_interventions(
                    rep_full, bank, indices, probe, targets,
                    ks=eval_ks, agreement_k=args.agreement_k,
                )
                random_sums[repeat] += res.scores.mean(dim=1).cpu().numpy()
                random_counts[repeat] += (res.remove_matches, res.keep_matches)
            batches += 1
            images += B

        if not batches:
            raise ValueError(f"No complete evaluation batches for {source}; check data and batch size")
        # Curves average batch means; agreement pools matching image counts.
        for group in GROUPS:
            values = metrics[group] / batches
            for col, k in enumerate(eval_ks):
                all_results.append(dict(model=source, group=group, k=k,
                    suff=float(values[0, col]), comp=float(values[1, col]),
                    suff_std=0., comp_std=0., repeats=1, num_batches=batches, num_samples=images))
            counts = learned_counts[group]
            agreement_rows.append(dict(model=source, group=group, k=args.agreement_k,
                remove_agreement=float(counts[0] / images), keep_agreement=float(counts[1] / images),
                remove_std=0., keep_std=0., num_samples=images, repeats=1))

        values = random_sums / batches
        means, stds = values.mean(axis=0), values.std(axis=0, ddof=1)
        for col, k in enumerate(eval_ks):
            all_results.append(dict(model=source, group="random", k=k,
                suff=float(means[0, col]), comp=float(means[1, col]),
                suff_std=float(stds[0, col]), comp_std=float(stds[1, col]),
                repeats=len(banks), num_batches=batches, num_samples=images))
            for repeat, seed in enumerate(args.random_seeds):
                random_rows.append(dict(model=source, group="random", seed=seed, k=k,
                    suff=float(values[repeat, 0, col]), comp=float(values[repeat, 1, col])))
        rates = random_counts / images
        agreement_rows.append(dict(model=source, group="random", k=args.agreement_k,
            remove_agreement=float(rates[:, 0].mean()), keep_agreement=float(rates[:, 1].mean()),
            remove_std=float(rates[:, 0].std(ddof=1)), keep_std=float(rates[:, 1].std(ddof=1)),
            num_samples=images, repeats=len(banks)))
        for repeat, seed in enumerate(args.random_seeds):
            agreement_repeat_rows.append(dict(model=source, group="random", seed=seed, k=args.agreement_k,
                remove_agreement=float(rates[repeat, 0]), keep_agreement=float(rates[repeat, 1]),
                remove_matches=int(random_counts[repeat, 0]), keep_matches=int(random_counts[repeat, 1]),
                num_samples=images))
        source_metadata[source] = dict(num_batches=batches, num_samples=images,
            representation_dim=proj_cfg["output_dim"], ks=eval_ks)

    output = Path(args.output_csv)
    write_csv(output, all_results)
    write_csv(output.parent / "random_repeats.csv", random_rows)
    write_csv(output.parent / "prediction_agreement.csv", agreement_rows)
    write_csv(output.parent / "prediction_agreement_random_repeats.csv", agreement_repeat_rows)
    metadata = dict(seed=args.seed, random_seeds=args.random_seeds,
        mask_seed_offset=args.mask_seed_offset, prediction_agreement_k=args.agreement_k,
        intervention="branch_contribution", random_selection="uniform_without_replacement_per_image",
        random_coefficients="signed projections of the full reconstruction onto a fixed random orthogonal basis",
        norm_matching=False, nested_k_subsets=True,
        target="ground_truth", sample_filter="none", shuffle=True, drop_last=True,
        batch_size=dataloader.batch_size, curve_aggregation="equal batch means; random mean/std across repeats",
        agreement_aggregation="pooled image counts; random mean/std across repeats",
        bn="eval with current-batch encoder statistics; frozen affine parameters and running buffers",
        sources=source_metadata)
    (output.parent / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] Results saved to {args.output_csv}")
    return all_results


if __name__ == "__main__":
    main()
