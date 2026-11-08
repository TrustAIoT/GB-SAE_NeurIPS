"""ImageNet cached-activation adapter for the supplied GB-SAE experiments.

Generic datasets/loaders do not need this module. Standardization and label
mapping retain the original probe protocol. Experiment callers select sources,
roots/splits, batch sizes and shuffling explicitly.
"""
import copy
import os
import random

import torch

from gbsae.data.config import build_activation_dataset
from gbsae.data.validation import validate_sample_alignment
from .data import make_dataloader

def load_config(path: str):
    from gbsae.configuration import load_model_config
    return load_model_config(path)

def sanity_check_alignment(dataset, num_checks=10):
    if len(dataset.samples) == 0:
        raise RuntimeError("Dataset is empty")

    num_checks = min(num_checks, len(dataset.samples))
    print(f"[CHECK] Running image / activation alignment check ({num_checks} samples)")
    indices = random.sample(range(len(dataset.samples)), num_checks)

    validate_sample_alignment(dataset, indices)

    print("[CHECK] Alignment sanity check PASSED")

def build_dataset(cfg, sources):
    return build_activation_dataset(cfg, sources=sources)

def configure_dataset_standardization(dataset, ckpt, sources, std_mode: str):
    """Select checkpoint statistics or retain statistics computed by the dataset."""
    if not dataset.standardize:
        return
    if std_mode == "ckpt":
        ckpt_std = ckpt.get("standardization_stats", None)
        assert ckpt_std is not None, "Checkpoint has no standardization_stats"

        dataset.standardization_stats = {}
        for source in sources:
            assert source in ckpt_std, (
                f"std stats for {source} not found in checkpoint, "
                f"available: {list(ckpt_std.keys())}"
            )
            dataset.standardization_stats[source] = {
                "mean": ckpt_std[source]["mean"].detach().cpu(),
                "std": ckpt_std[source]["std"].detach().cpu(),
            }
        print(f"[INFO] Using checkpoint standardization stats for {sources}")
    else:
        print(f"[INFO] Using dataset-computed standardization stats for {sources}")

def make_activation_loader(cfg, ckpt, sources, *, batch_size, num_workers=0,
                           std_mode="ckpt", shuffle=True, generator=None):
    dataset = build_dataset(cfg, sources)
    configure_dataset_standardization(dataset, ckpt, sources, std_mode)
    sanity_check_alignment(dataset)
    return make_dataloader(dataset, batch_size=batch_size, shuffle=shuffle,
                           num_workers=num_workers, generator=generator)


def load_inputs(args):
    """Read the SAE run and explicit data overrides, without selecting a workflow."""
    cfg = load_config(args.config)
    sources = list(cfg["models"]["sources"]) if args.sources is None else list(args.sources)
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("sources must be nonempty and unique")
    for source in sources:
        if source not in cfg["models"]["sources"]:
            raise ValueError(f"Source {source} is absent from the training configuration")
    cfg = copy.deepcopy(cfg)
    for key in ("imagenet_root", "activation_root", "split"):
        value = getattr(args, key, None)
        if value is not None:
            cfg["data"][key] = value
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    return cfg, ckpt, sources


def build_eval_to_train_label_lut(
    cfg,
    source,
    eval_root,
    eval_act_root,
    train_root,
    train_act_root,
    eval_split,
    train_split,
):
    """Map evaluation label indices to probe-training indices using class-directory WNIDs.
    
    Return a CPU LongTensor lookup table with -1 for evaluation classes absent
    from training. This utility does not drop samples; callers must handle -1.
    cfg supplies source prefix/layout policies. Both datasets are enumerated
    without activation standardization, and eval/train splits are independent.
    """
    print(f"[INFO] Label LUT eval split : {eval_split}")
    print(f"[INFO] Label LUT check split: {train_split}")

    # -------- load datasets --------
    ds_eval = build_activation_dataset(
        cfg, sources=[source], imagenet_root=eval_root,
        activation_root=eval_act_root, split=eval_split, standardize=False,
    )

    ds_train = build_activation_dataset(
        cfg, sources=[source], imagenet_root=train_root,
        activation_root=train_act_root, split=train_split, standardize=False,
    )

    # -------- build wnid -> label maps from samples --------
    def build_wnid_to_label(ds):
        mapping = {}
        for img_path, label in ds.samples:
            # img_path: .../train/n01440764/xxx.JPEG
            wnid = os.path.basename(os.path.dirname(img_path))
            mapping[wnid] = label
        return mapping

    eval_wnid_to_label = build_wnid_to_label(ds_eval)
    train_wnid_to_label = build_wnid_to_label(ds_train)

    eval_wnids = set(eval_wnid_to_label.keys())
    train_wnids = set(train_wnid_to_label.keys())

    overlap = sorted(eval_wnids & train_wnids)
    missing = sorted(eval_wnids - train_wnids)

    print(f"[CHECK] eval wnids on disk      : {len(eval_wnids)}")
    print(f"[CHECK] train wnids on disk     : {len(train_wnids)}")
    print(f"[CHECK] overlap (usable classes): {len(overlap)}")

    if missing:
        print("[WARN] eval wnids missing in train (mapped to -1):", missing[:10])

    # -------- build eval_label -> train_label --------
    eval_label_to_train = {}
    for wnid in overlap:
        eval_label = eval_wnid_to_label[wnid]
        train_label = train_wnid_to_label[wnid]
        eval_label_to_train[eval_label] = train_label

    # -------- build LUT --------
    max_eval_label = max(eval_wnid_to_label.values())
    lut = torch.full((max_eval_label + 1,), -1, dtype=torch.long)

    for k, v in eval_label_to_train.items():
        lut[k] = v

    print(f"[OK] Built eval train LUT ({len(eval_label_to_train)} classes)")

    return lut

