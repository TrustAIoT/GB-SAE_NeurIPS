"""Generic DataLoader construction, independent of datasets and SAE configs."""
import torch


def make_dataloader(dataset, *, batch_size, shuffle=True, num_workers=0,
                    pin_memory=True, generator=None, drop_last=False, **kwargs):
    """Accept any PyTorch Dataset; custom collate functions pass through kwargs.

    Researchers may also construct their own DataLoader and pass it to fit().
    No image/cache paths, standardization or source names are inferred here.
    """
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=pin_memory, generator=generator, drop_last=drop_last, **kwargs,
    )
