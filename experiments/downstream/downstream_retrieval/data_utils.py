import json
import math
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm


def wnid_from_image_path(image_path: str) -> str:
    return os.path.basename(os.path.dirname(image_path))


def image_id_from_path(image_path: str) -> str:
    return Path(image_path).stem


def build_wnid_to_label(samples: Sequence[tuple]) -> Dict[str, int]:
    wnid_to_label: Dict[str, int] = {}
    label_to_wnid: Dict[int, str] = {}
    for image_path, raw_label in samples:
        wnid = wnid_from_image_path(image_path)
        raw_label = int(raw_label)
        if wnid in wnid_to_label and wnid_to_label[wnid] != raw_label:
            raise RuntimeError(f"WNID {wnid} maps to multiple labels")
        if raw_label in label_to_wnid and label_to_wnid[raw_label] != wnid:
            raise RuntimeError(
                f"Raw label {raw_label} maps to both {label_to_wnid[raw_label]} and {wnid}"
            )
        wnid_to_label[wnid] = raw_label
        label_to_wnid[raw_label] = wnid
    return wnid_to_label


@dataclass
class DatasetAudit:
    train_samples: int
    val_samples: int
    train_classes: int
    val_classes: int
    missing_train_wnids_in_val: List[str]
    unknown_val_wnids: List[str]
    val_class_counts: Dict[str, int]
    val_raw_to_train_label: Dict[int, int]


class CanonicalActivationDataset(Dataset):
    """Wrap an activation dataset and replace split-local labels by train labels."""

    def __init__(self, base_dataset, raw_to_canonical: Mapping[int, int]):
        self.base_dataset = base_dataset
        self.raw_to_canonical = {int(k): int(v) for k, v in raw_to_canonical.items()}
        self.samples = base_dataset.samples
        self.samples_used = base_dataset.samples_used
        self.labels: List[int] = []
        self.sample_ids: List[str] = []
        self.wnids: List[str] = []
        for image_path, raw_label in self.samples:
            raw_label = int(raw_label)
            if raw_label not in self.raw_to_canonical:
                raise RuntimeError(f"Sample label {raw_label} has no canonical train mapping")
            self.labels.append(self.raw_to_canonical[raw_label])
            self.sample_ids.append(image_id_from_path(image_path))
            self.wnids.append(wnid_from_image_path(image_path))

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        activations, raw_label = self.base_dataset[index]
        canonical_label = self.raw_to_canonical[int(raw_label)]
        return activations, canonical_label, index


def canonicalize_train_val_datasets(
    train_dataset,
    val_dataset,
    expected_train_classes: Optional[int],
) -> tuple:
    train_wnid_to_label = build_wnid_to_label(train_dataset.samples)
    val_wnid_to_label = build_wnid_to_label(val_dataset.samples)

    if expected_train_classes is not None and len(train_wnid_to_label) != expected_train_classes:
        raise RuntimeError(
            f"Expected {expected_train_classes} train classes, found {len(train_wnid_to_label)}"
        )

    train_labels = sorted(train_wnid_to_label.values())
    if train_labels != list(range(len(train_labels))):
        raise RuntimeError(
            "Train labels are not a contiguous canonical range; refusing to silently re-index them"
        )

    unknown_val = sorted(set(val_wnid_to_label) - set(train_wnid_to_label))
    if unknown_val:
        raise RuntimeError(
            "Validation contains WNIDs absent from train: " + ", ".join(unknown_val[:20])
        )

    train_raw_to_canonical = {label: label for label in train_labels}
    val_raw_to_train = {
        val_wnid_to_label[wnid]: train_wnid_to_label[wnid] for wnid in val_wnid_to_label
    }

    val_counts = defaultdict(int)
    for image_path, _ in val_dataset.samples:
        val_counts[wnid_from_image_path(image_path)] += 1

    missing_train_in_val = sorted(set(train_wnid_to_label) - set(val_wnid_to_label))
    audit = DatasetAudit(
        train_samples=len(train_dataset),
        val_samples=len(val_dataset),
        train_classes=len(train_wnid_to_label),
        val_classes=len(val_wnid_to_label),
        missing_train_wnids_in_val=missing_train_in_val,
        unknown_val_wnids=unknown_val,
        val_class_counts=dict(sorted(val_counts.items())),
        val_raw_to_train_label=dict(sorted(val_raw_to_train.items())),
    )
    return (
        CanonicalActivationDataset(train_dataset, train_raw_to_canonical),
        CanonicalActivationDataset(val_dataset, val_raw_to_train),
        audit,
    )


def save_dataset_audit(audit: DatasetAudit, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(asdict(audit), handle, indent=2, ensure_ascii=False)


def sanity_check_image_activation_alignment(
    dataset: CanonicalActivationDataset,
    num_checks: int,
    seed: int,
) -> None:
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty")
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(num_checks, len(dataset)))
    from gbsae.data.validation import validate_sample_alignment
    validate_sample_alignment(dataset.base_dataset, indices)


class PKBatchSampler(Sampler[List[int]]):
    """Yield P-class by K-sample batches using a seed+epoch RNG.
    
    Classes cycle through reshuffled permutations. Draw within a class without
    replacement when enough examples exist, otherwise with replacement. This
    is deterministic for a fixed seed/epoch, not an exact-once dataset traversal.
    """

    def __init__(
        self,
        labels: Sequence[int],
        classes_per_batch: int,
        samples_per_class: int,
        batches_per_epoch: Optional[int],
        seed: int,
    ):
        self.class_to_indices: Dict[int, List[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            self.class_to_indices[int(label)].append(index)
        self.classes = sorted(self.class_to_indices)
        self.classes_per_batch = int(classes_per_batch)
        self.samples_per_class = int(samples_per_class)
        self.batch_size = self.classes_per_batch * self.samples_per_class
        if self.classes_per_batch > len(self.classes):
            raise ValueError("classes_per_batch exceeds the number of train classes")
        self.batches_per_epoch = (
            int(batches_per_epoch)
            if batches_per_epoch is not None
            else math.ceil(len(labels) / self.batch_size)
        )
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        class_order = self.classes.copy()
        rng.shuffle(class_order)
        class_cursor = 0
        for _ in range(self.batches_per_epoch):
            if class_cursor + self.classes_per_batch > len(class_order):
                rng.shuffle(class_order)
                class_cursor = 0
            selected_classes = class_order[class_cursor : class_cursor + self.classes_per_batch]
            class_cursor += self.classes_per_batch

            batch: List[int] = []
            for label in selected_classes:
                candidates = self.class_to_indices[label]
                if len(candidates) >= self.samples_per_class:
                    batch.extend(rng.sample(candidates, self.samples_per_class))
                else:
                    batch.extend(rng.choices(candidates, k=self.samples_per_class))
            rng.shuffle(batch)
            yield batch


class CachedFeatureDataset(Dataset):
    def __init__(self, cache: Mapping[str, object]):
        self.features: Mapping[str, torch.Tensor] = cache["features"]
        self.labels: torch.Tensor = cache["labels"].long()
        self.sample_ids: List[str] = list(cache["sample_ids"])
        lengths = {tensor.shape[0] for tensor in self.features.values()}
        lengths.add(self.labels.shape[0])
        lengths.add(len(self.sample_ids))
        if len(lengths) != 1:
            raise RuntimeError(f"Inconsistent feature-cache lengths: {sorted(lengths)}")

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int):
        return {name: values[index] for name, values in self.features.items()}, self.labels[index], index
