"""Optional CSV/checkpoint helpers; output protocol belongs to the experiment."""
import csv
from pathlib import Path

import torch


class CSVLogger:
    """Write sample-weighted epoch metrics, with accuracy expressed in percent."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="") as handle:
            csv.writer(handle).writerow(["epoch", "loss", "acc@1", "acc@5"])

    def __call__(self, epoch, metrics):
        with self.path.open("a", newline="") as handle:
            csv.writer(handle).writerow([epoch, metrics.loss, metrics.acc1, metrics.acc5])


def save_head(path, head, *, epoch, metadata=None):
    """Save state_dict plus caller-defined metadata; do not infer head semantics.

    The probe_state_dict key retains compatibility with the supplied artifacts.
    Paths, source lists, fusion names and save intervals are chosen by callers.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata or {})
    payload.update(probe_state_dict=head.state_dict(), epoch=epoch)
    torch.save(payload, path)
    print(f"[DONE] Saved checkpoint to {path}")
