"""Projected-space training diagnostics for the CLIP adapter."""
import torch

from gbsae.training.logger import BottleneckSAELogger, compute_r2


class CLIPTrainingLogger(BottleneckSAELogger):
    """Record projected R-squared, losses, gates and dictionary usage."""

    def __init__(self, model_names, linear_projectors, **kwargs):
        super().__init__(model_names, {}, **kwargs)
        self.linear_projectors = linear_projectors

    def log_step_model(
        self, *, name, x, x_hat, x_hat_sh, act, z_sh, z_pr, orthogonality,
    ):
        r2 = {"linear/total": compute_r2(x, x_hat)}
        if x_hat_sh is not None:
            r2["linear/shared"] = compute_r2(x, x_hat_sh)
        for key, value in r2.items():
            self._epoch_r2_sum[name][key] += value
            self._epoch_r2_cnt[name][key] += 1
        self._step_r2_linear_total[name] = r2["linear/total"]
        if "linear/shared" in r2:
            self._step_r2_linear_shared[name] = r2["linear/shared"]

        if orthogonality is not None:
            self._epoch_ortho[name].append(float(orthogonality))
        with torch.no_grad():
            sh_active = (z_sh != 0).sum(dim=0).cpu()
            pr_active = (z_pr != 0).sum(dim=0).cpu()
        if self._epoch_sh_counts[name] is None:
            self._epoch_sh_counts[name] = sh_active
            self._epoch_pr_counts[name] = pr_active
        else:
            self._epoch_sh_counts[name] += sh_active
            self._epoch_pr_counts[name] += pr_active
