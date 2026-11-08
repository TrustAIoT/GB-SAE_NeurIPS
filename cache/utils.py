"""Cache-stage randomness, view assignment and external component imports."""
from contextlib import contextmanager
import hashlib
import importlib
import random

import numpy as np
import torch


def set_global_seed(seed: int):
    """Seed cache execution; image/view seeds are derived separately below."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def deterministic_seed(image_name: str, view_id: int, base_seed: int = 42) -> int:
    """Retain the original filename/view/base-seed mapping exactly."""
    key = f"{image_name}_view{view_id}_seed{base_seed}"
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**31)


def assign_views(image_name: str, sources, base_seed: int, enabled: bool) -> dict[str, int]:
    """Assign original+N-1 views; a single augmented source uses view 1."""
    if not enabled:
        return dict.fromkeys(sources, 0)
    if len(sources) == 1:
        return {sources[0]: 1}
    permutation = list(range(len(sources)))
    stable_hash = int(hashlib.md5(image_name.encode("utf-8")).hexdigest(), 16)
    random.Random(stable_hash + base_seed).shuffle(permutation)
    return dict(zip(sources, permutation))


@contextmanager
def image_random_state(seed: int):
    """Isolate Python/NumPy/torch CPU RNGs used by PIL-image augmentations."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.random.default_generator.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def import_symbol(reference: str):
    """Import an explicitly configured module:attribute factory."""
    module, separator, name = reference.partition(":")
    if not separator or not module or not name:
        raise ValueError(f"Expected a module:attribute reference, got {reference!r}")
    return getattr(importlib.import_module(module), name)
