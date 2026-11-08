"""Dataset selection and PIL-image loading; all sources share one manifest."""
from collections import defaultdict
from pathlib import Path
import random
import shutil

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder, ImageNet
from PIL import Image

from .manifest import ImageRecord, load_manifest, validate_records


def balanced_indices(samples, fraction, seed):
    """Retain the original near-class-balanced subset allocation and ordering.

    Scarce classes are capped without redistributing the shortfall. Use mode all
    for the complete split; balanced sampling is an explicit separate policy.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("sampling.fraction must be in (0,1]")
    by_class = defaultdict(list)
    for index, (_, target) in enumerate(samples):
        by_class[int(target)].append(index)
    classes = len(by_class)
    if not classes:
        raise ValueError("The source dataset is empty")
    target = max(classes, round(len(samples) * fraction))
    base, remainder = divmod(target, classes)
    rng = random.Random(seed)
    extra = set(rng.sample(sorted(by_class), remainder))
    selected = []
    for label in sorted(by_class):
        indices = by_class[label].copy()
        rng.shuffle(indices)
        selected.extend(indices[:min(len(indices), base + int(label in extra))])
    return sorted(selected)


def select_images(cfg):
    data, sampling = cfg["data"], cfg.get("sampling", {})
    root, split = Path(data["root"]), data["split"]
    kind = data.get("type", "imagenet")
    if kind == "imagenet":
        dataset = ImageNet(str(root), split=split, transform=None)
    elif kind == "image_folder":
        dataset = ImageFolder(str(root / split), transform=None)
    else:
        raise ValueError("data.type must be imagenet or image_folder")
    available = {
        Path(path).relative_to(root / split).as_posix(): ImageRecord(
            split, Path(path).relative_to(root / split).as_posix(), int(label), Path(path).relative_to(root / split).parts[0],
        ) for path, label in dataset.samples
    }
    mode = sampling.get("mode", "all")
    if mode == "manifest":
        records = load_manifest(sampling["manifest"], split)
        for record in records:
            if available.get(record.relative_path) != record:
                raise ValueError(f"Manifest image/label does not match dataset: {record}")
    elif mode in {"all", "balanced"}:
        indices = range(len(dataset)) if mode == "all" else balanced_indices(
            dataset.samples, float(sampling.get("fraction", 0.1)), cfg.get("runtime", {}).get("seed", 42),
        )
        records = [available[Path(dataset.samples[i][0]).relative_to(root / split).as_posix()] for i in indices]
    else:
        raise ValueError("sampling.mode must be all, balanced or manifest")
    if sampling.get("exclude_manifest"):
        excluded = {record.relative_path for record in load_manifest(sampling["exclude_manifest"], split)}
        records = [record for record in records if record.relative_path not in excluded]
    validate_records(records, split)
    return records


def export_images(data, records, overwrite=False):
    """Optionally copy selected originals and ImageNet metadata for training."""
    if not data.get("export_root"):
        return
    root, destination_root = Path(data["root"]), Path(data["export_root"])
    metadata, target_metadata = root / "meta.bin", destination_root / "meta.bin"
    if data.get("type", "imagenet") == "imagenet" and metadata.exists():
        if target_metadata.exists() and target_metadata.read_bytes() != metadata.read_bytes() and not overwrite:
            raise ValueError("Exported ImageNet metadata differs; choose another export_root or explicit output.overwrite")
        target_metadata.parent.mkdir(parents=True, exist_ok=True)
        if not target_metadata.exists() or overwrite:
            shutil.copy2(metadata, target_metadata)
    for record in records:
        original = root / record.split / record.relative_path
        target = destination_root / record.split / record.relative_path
        if original.resolve() == target.resolve():
            continue
        if target.exists():
            same = (original.stat().st_size, original.stat().st_mtime_ns) == (target.stat().st_size, target.stat().st_mtime_ns)
            if same:
                continue
            if not overwrite:
                raise ValueError(f"Exported original differs: {target}; choose another export_root or explicit output.overwrite")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)


class ImageDataset(Dataset):
    def __init__(self, root, records):
        self.root, self.records = Path(root), records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.root / record.split / record.relative_path) as image:
            return image.convert("RGB"), record


def collate_images(batch):
    images, records = zip(*batch)
    return list(images), list(records)
