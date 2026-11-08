# logger.py
"""
BottleneckSAELogger
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Any, Optional

import numpy as np
import torch

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


def compute_r2(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Return Python-float R-squared after flattening all features into one vector.
    
    The denominator uses one global batch/feature mean and epsilon 1e-12.
    """
    x = x.detach().reshape(-1)
    x_hat = x_hat.detach().reshape(-1)
    ss_res = torch.sum((x - x_hat) ** 2)
    ss_tot = torch.sum((x - x.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def _nanmean(vals):
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.nanmean(arr))


Granularity = int


class BottleneckSAELogger:
    """Accumulate batch-level reconstruction, gate and dictionary-usage diagnostics.
    
    Projected R-squared is measured each step; pseudoinverse raw R-squared is
    measured every tenth global step. Epoch reconstruction values average those
    batch scores rather than recomputing a global dataset score. granularity=1
    logs per-source metrics; 2 logs means over sources. Optional wandb logging
    requires an initialized run. end_epoch returns the per-source metrics.
    """
    def __init__(
        self,
        model_names,
        linear_projectors,
        granularity: Granularity = 1,
        *,
        enable_wandb: bool = True,
        terminal_print: bool = True,
    ):
        self.model_names = list(model_names)
        self.linear_projectors = linear_projectors
        self.granularity = self._normalize_granularity(granularity)

        self.enable_wandb = enable_wandb
        self.terminal_print = terminal_print

        self._projector_mode: Dict[str, str] = {}
        self._pinv_cache: Dict[str, torch.Tensor] = {}

        for name, proj in linear_projectors.items():
            if not hasattr(proj, "linear"):
                self._projector_mode[name] = "identity"
            else:
                W = proj.linear.weight
                if W.requires_grad:
                    self._projector_mode[name] = "learnable"
                else:
                    self._projector_mode[name] = "frozen"
                    self._pinv_cache[name] = torch.linalg.pinv(W)

        self._global_step: int = -1
        self._epoch_idx: int = -1
        self._step_global: Dict[str, float] = {}
        self._step_r2_linear_total: Dict[str, float] = {}
        self._step_r2_linear_shared: Dict[str, float] = {}

        self._wandb_defined = False
        self._define_wandb_metrics()
        self.reset_epoch()

    def _wandb_available(self) -> bool:
        return (
            self.enable_wandb
            and (wandb is not None)
            and getattr(wandb, "run", None) is not None
        )

    def _define_wandb_metrics(self):
        if not self._wandb_available() or self._wandb_defined:
            return
        try:
            wandb.define_metric("trainer/step")
            wandb.define_metric("trainer/epoch")
            wandb.define_metric("step/*", step_metric="trainer/step")
            wandb.define_metric("epoch/*", step_metric="trainer/epoch")
        except Exception:
            pass
        self._wandb_defined = True

    def _wandb_log(self, data: Dict[str, Any]):
        if self._wandb_available():
            wandb.log(data)

    @staticmethod
    def _normalize_granularity(g: int) -> int:
        if g not in (1, 2):
            raise ValueError(
                f"Invalid granularity={g}. "
                "Expected: 1 (per-model) or 2 (model-mean)."
            )
        return g

    def _linear_to_raw(self, name: str, x_hat: torch.Tensor) -> torch.Tensor:
        mode = self._projector_mode[name]
        if mode == "identity":
            return x_hat
        if mode == "frozen":
            W_pinv = self._pinv_cache[name]
            return x_hat @ W_pinv.T
        if mode == "learnable":
            with torch.no_grad():
                W = self.linear_projectors[name].linear.weight
                W_pinv = torch.linalg.pinv(W)
            return x_hat @ W_pinv.T
        raise RuntimeError(f"Unknown projector mode: {mode}")

    def reset_epoch(self):
        self._epoch_r2_sum = defaultdict(lambda: defaultdict(float))
        self._epoch_r2_cnt = defaultdict(lambda: defaultdict(int))
        self._epoch_ortho = defaultdict(list)
        self._epoch_sh_counts = defaultdict(lambda: None)
        self._epoch_pr_counts = defaultdict(lambda: None)
        self._epoch_global = defaultdict(list)

        self._step_global = {}
        self._step_r2_linear_total = {}
        self._step_r2_linear_shared = {}

        self._epoch_gate_mean = []
        self._epoch_gate_std = []
        self._epoch_gate_sharpness = []
        self._epoch_shared_align = []
        self._epoch_shared_align_support = []
        self._epoch_shared_align_mag = []

        self._epoch_hard_shared = []
        self._epoch_hard_private = []
        self._epoch_mid = []

    def start_step(self, *, aux_prob: float, lr: float, gate_scale: float):
        self._global_step += 1
        self._step_global = {
            "aux_prob": aux_prob,
            "lr": lr,
            "gate_scale": gate_scale,
        }
        for k, v in self._step_global.items():
            self._epoch_global[k].append(v)

    def log_step_losses(self, losses: Dict[str, float]):
        for k, v in losses.items():
            self._epoch_global[k].append(float(v))

    def log_step_model(
        self,
        *,
        name: str,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        x_hat_sh: Optional[torch.Tensor],
        act: torch.Tensor,
        z_sh: torch.Tensor,
        z_pr: torch.Tensor,
        orthogonality: Optional[torch.Tensor],
    ):
        act_hat = self._linear_to_raw(name, x_hat)
        act_hat_sh = self._linear_to_raw(name, x_hat_sh) if x_hat_sh is not None else None

        r2 = {"linear/total": compute_r2(x, x_hat)}
        if self._global_step % 10 == 0:
            r2["raw/total"] = compute_r2(act, act_hat)

        if x_hat_sh is not None:
            r2["linear/shared"] = compute_r2(x, x_hat_sh)
            if self._global_step % 10 == 0:
                r2["raw/shared"] = compute_r2(act, act_hat_sh)

        for k_name, v in r2.items():
            self._epoch_r2_sum[name][k_name] += float(v)
            self._epoch_r2_cnt[name][k_name] += 1

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

    def log_gate(self, g: torch.Tensor):
        g = g.detach()
        self._epoch_gate_mean.append(g.mean().item())
        self._epoch_gate_std.append(g.std().item())
        self._epoch_gate_sharpness.append(torch.abs(g - 0.5).mean().item())
        self._epoch_hard_shared.append((g > 0.9).float().mean().item())
        self._epoch_hard_private.append((g < 0.1).float().mean().item())
        self._epoch_mid.append(((g >= 0.3) & (g <= 0.7)).float().mean().item())

    def log_shared_align(
        self,
        val: float,
        *,
        support: Optional[float] = None,
        mag: Optional[float] = None,
    ):
        self._epoch_shared_align.append(val)
        if support is not None:
            self._epoch_shared_align_support.append(float(support))
        if mag is not None:
            self._epoch_shared_align_mag.append(float(mag))

    def end_step(self, *, total_loss: float):
        total_loss = float(total_loss)
        self._epoch_global["total_loss"].append(total_loss)

        data: Dict[str, Any] = {
            "trainer/step": self._global_step,
            "step/total_loss": total_loss,
            "step/lr": float(self._step_global.get("lr", np.nan)),
            "step/gate_scale": float(self._step_global.get("gate_scale", np.nan)),
        }

        for name in self.model_names:
            if name in self._step_r2_linear_total:
                data[f"step/{name}/r2/linear/total"] = float(self._step_r2_linear_total[name])
            if name in self._step_r2_linear_shared:
                data[f"step/{name}/r2/linear/shared"] = float(self._step_r2_linear_shared[name])

        self._wandb_log(data)

    def end_epoch(self) -> Dict[str, Dict[str, float]]:
        self._epoch_idx += 1
        results: Dict[str, Dict[str, float]] = {}

        for name in self.model_names:
            metrics: Dict[str, float] = {}
            for k_name in self._epoch_r2_sum[name]:
                metrics[f"r2/{k_name}"] = (
                    self._epoch_r2_sum[name][k_name]
                    / self._epoch_r2_cnt[name][k_name]
                )

            if len(self._epoch_ortho[name]) > 0:
                metrics["orthogonality"] = float(np.mean(self._epoch_ortho[name]))

            sh_counts = self._epoch_sh_counts[name]
            pr_counts = self._epoch_pr_counts[name]

            if sh_counts is not None:
                total = sh_counts.sum()
                metrics["sh_dict_active_fraction"] = float((sh_counts > 0).float().mean())
                if total > 0:
                    p = sh_counts / total
                    H = -(p * (p + 1e-12).log()).sum()
                    metrics["sh_dict_usage_entropy"] = float(H / math.log(len(p) + 1e-12))
                else:
                    metrics["sh_dict_usage_entropy"] = 0.0

            if pr_counts is not None:
                total = pr_counts.sum()
                metrics["pr_dict_active_fraction"] = float((pr_counts > 0).float().mean())
                if total > 0:
                    p = pr_counts / total
                    H = -(p * (p + 1e-12).log()).sum()
                    metrics["pr_dict_usage_entropy"] = float(H / math.log(len(p) + 1e-12))
                else:
                    metrics["pr_dict_usage_entropy"] = 0.0

            results[name] = metrics

        mean_metrics: Dict[str, float] = {}
        if len(results) > 0:
            keys = set().union(*[set(m.keys()) for m in results.values()])
            for k in keys:
                mean_metrics[k] = float(np.mean([results[n].get(k, np.nan) for n in self.model_names]))

        data: Dict[str, Any] = {
            "trainer/epoch": self._epoch_idx,
            "epoch/total_loss": _nanmean(self._epoch_global.get("total_loss", [np.nan])),
        }

        if self.granularity == 1:
            for name in self.model_names:
                for k, v in results[name].items():
                    data[f"epoch/{name}/{k}"] = float(v)
            for k, vals in self._epoch_global.items():
                if k.startswith("loss/"):
                    data[f"epoch/{k}"] = _nanmean(vals)
        else:
            for k, v in mean_metrics.items():
                data[f"epoch/{k}"] = float(v)

        if len(self._epoch_gate_mean) > 0:
            data["epoch/gate_mean"] = float(np.mean(self._epoch_gate_mean))
            data["epoch/gate_std"] = float(np.mean(self._epoch_gate_std))
            data["epoch/gate_sharpness"] = float(np.mean(self._epoch_gate_sharpness))
            data["epoch/gate_hard_shared_frac"] = float(np.mean(self._epoch_hard_shared))
            data["epoch/gate_hard_private_frac"] = float(np.mean(self._epoch_hard_private))
            data["epoch/gate_mid_frac"] = float(np.mean(self._epoch_mid))

        if len(self._epoch_shared_align) > 0:
            data["epoch/shared_align"] = float(np.mean(self._epoch_shared_align))
        if len(self._epoch_shared_align_support) > 0:
            data["epoch/shared_align_support"] = float(np.mean(self._epoch_shared_align_support))
        if len(self._epoch_shared_align_mag) > 0:
            data["epoch/shared_align_mag"] = float(np.mean(self._epoch_shared_align_mag))

        self._wandb_log(data)

        if self.terminal_print:
            r2_lin_total_mean = mean_metrics.get("r2/linear/total", np.nan)
            r2_lin_shared_mean = mean_metrics.get("r2/linear/shared", np.nan)
            print(
                f"[Epoch {self._epoch_idx}] "
                f"loss={data['epoch/total_loss']:.6f} "
                f"r2_linear_total_mean={r2_lin_total_mean:.6f} "
                f"r2_linear_shared_mean={r2_lin_shared_mean:.6f}"
            )

        self.reset_epoch()
        return results
