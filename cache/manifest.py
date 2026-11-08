"""Stable image identities shared by extraction, combination and inspection."""
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class ImageRecord:
    split: str
    relative_path: str
    label: int
    class_id: str


def selection_fingerprint(records, image_root=None):
    """Stream image identities into bounded-size metadata; retain the CSV list.

    Extraction additionally checks original file sizes/timestamps. The selection
    includes order and labels, independently of decoding workers or batch size.
    """
    fingerprint = hashlib.sha256()
    for record in records:
        value = asdict(record)
        if image_root is not None:
            stat = (Path(image_root) / record.split / record.relative_path).stat()
            value["input_file"] = [stat.st_size, stat.st_mtime_ns]
        fingerprint.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
        fingerprint.update(b"\n")
    return {"count": len(records), "fingerprint": fingerprint.hexdigest(), "checks_input_files": image_root is not None}


def validate_records(records, split):
    seen, outputs = set(), set()
    for record in records:
        path = PurePosixPath(record.relative_path)
        if path.is_absolute() or ".." in path.parts or "\\" in record.relative_path or ":" in record.relative_path:
            raise ValueError(f"Manifest path must be a safe relative path: {record.relative_path}")
        if record.split != split or len(path.parts) < 2 or path.parts[0] != record.class_id:
            raise ValueError(f"Manifest split/class mismatch: {record}")
        output_key = str(path.with_suffix(""))
        if record.relative_path in seen or output_key in outputs:
            raise ValueError(f"Duplicate image identity or cache filename: {record.relative_path}")
        if record.label < 0:
            raise ValueError(f"Negative label: {record}")
        seen.add(record.relative_path)
        outputs.add(output_key)
    if not records:
        raise ValueError("The selected image manifest is empty")
    return records


def load_manifest(path, split):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    records = [ImageRecord(
        row.get("split") or split, row["relative_path"], int(row.get("label", row.get("target"))),
        row.get("class_id") or PurePosixPath(row["relative_path"]).parts[0],
    ) for row in rows]
    return validate_records(records, split)


def save_manifest(path, records):
    from .storage import atomic_file
    with atomic_file(path, "w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "relative_path", "label", "class_id"])
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def discover_records(roots, sources, split, combined_root=None):
    """Build a union of existing cache identities so missing sources are visible."""
    from .storage import load_npz, sidecar_path, read_json
    candidates = {}
    scans = [(Path(root) / split, source) for source, root in roots.items() if source in sources]
    if combined_root:
        scans.append((Path(combined_root) / split, None))
    for root, source in scans:
        suffix = f"{source}.npz" if source else "_combined.npz"
        for path in sorted(root.rglob("*" + suffix)):
            meta = read_json(sidecar_path(path)) if sidecar_path(path).exists() else None
            if meta:
                row = meta["image"]
                record = ImageRecord(**row)
            else:
                if source is None:
                    raise ValueError("Directory discovery of legacy combined caches needs image sidecars or inputs.manifest")
                values = load_npz(path)
                if "label" not in values:
                    raise ValueError(f"Missing label for directory discovery: {path}; provide inputs.manifest")
                relative = path.relative_to(root).as_posix()[:-len(suffix)] + ".JPEG"
                if len(PurePosixPath(relative).parts) < 2:
                    raise ValueError(f"Cache must be organized by split/class: {path}")
                record = ImageRecord(split, relative, int(values["label"]), PurePosixPath(relative).parts[0])
            previous = candidates.get(record.relative_path)
            if previous and previous != record:
                raise ValueError(f"Image identity/label mismatch: {record.relative_path}")
            candidates[record.relative_path] = record
    return validate_records(sorted(candidates.values(), key=lambda r: r.relative_path), split)
