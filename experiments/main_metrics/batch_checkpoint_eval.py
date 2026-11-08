"""Batch checkpoint evaluation through the common reconstruction/retrieval modules."""
from pathlib import Path

from .common import load_checkpoint, write_csv
from .reconstruction.evaluator import evaluate_reconstruction, build_reconstruction_rows
from .retrieval.evaluator import build_retrieval_rows, evaluate_retrieval


RECON_FIELDS = [
    "checkpoint_path", "checkpoint_name", "model",
    "r2_linear_total", "r2_linear_shared", "r2_linear_private", "r2_linear_private_residual",
    "r2_raw_total", "r2_raw_shared", "r2_raw_private", "r2_raw_private_residual",
]
RETRIEVAL_FIELDS = [
    "checkpoint_path", "checkpoint_name", "task", "branch", "recall_at_k",
    "accuracy", "random_baseline", "shared_vs_random_ratio",
]


def evaluate_checkpoints(
    cfg, checkpoint_paths, output_dir, device="cpu",
    metrics=("reconstruction", "retrieval"), sources=None, projector_from=None,
    retrieval_imagenet_root=None, retrieval_activation_root=None, retrieval_split=None,
    retrieval_batch_size=32, retrieval_num_workers=16, retrieval_max_images=None,
    pool="mean", shared_repr="xhat_sh", private_repr="xhat_pr",
):
    """Write cumulative CSVs after each checkpoint, as in the original batch script.

    All architecture and preprocessing choices come from the training config.
    Data overrides should be applied before calling this function. Separate
    retrieval roots/split remain supported for the original batch workflow.
    """
    if not metrics or set(metrics) - {"reconstruction", "retrieval"}:
        raise ValueError("metrics must select reconstruction and/or retrieval")
    if not checkpoint_paths:
        raise ValueError("No checkpoint paths supplied")
    recon_rows, retrieval_rows = [], []
    output_dir = Path(output_dir)
    for checkpoint_path in checkpoint_paths:
        checkpoint_path = str(checkpoint_path)
        print(f"\n========== Evaluating checkpoint ==========\n{checkpoint_path}\n")
        ckpt = load_checkpoint(checkpoint_path, device)
        if "reconstruction" in metrics:
            linear, raw = evaluate_reconstruction(cfg, ckpt, device, sources, projector_from)
            recon_rows.extend(build_reconstruction_rows(checkpoint_path, linear, raw))
            write_csv(output_dir / "reconstruction_metrics.csv", recon_rows, RECON_FIELDS)
        if "retrieval" in metrics:
            retrieval = evaluate_retrieval(
                cfg=cfg, ckpt=ckpt, device=device,
                imagenet_root=retrieval_imagenet_root or cfg["data"]["imagenet_root"],
                activation_root=retrieval_activation_root or cfg["data"]["activation_root"],
                split=retrieval_split or cfg["data"].get("split", "train"),
                batch_size=retrieval_batch_size, num_workers=retrieval_num_workers,
                max_images=retrieval_max_images, pool=pool,
                shared_repr=shared_repr, private_repr=private_repr,
            )
            retrieval_rows.extend(build_retrieval_rows(checkpoint_path, retrieval))
            write_csv(output_dir / "retrieval_metrics.csv", retrieval_rows, RETRIEVAL_FIELDS)
    return {"reconstruction": recon_rows, "retrieval": retrieval_rows}


if __name__ == "__main__":
    from .cli import run_cli
    run_cli()
