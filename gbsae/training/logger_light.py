# logger_light_mean.py
import torch
import numpy as np
import random

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ---------------- utils ----------------

def compute_r2(x, x_hat):
    """Return scalar-tensor R-squared centered separately per feature over the batch.
    """
    ss_res = ((x - x_hat) ** 2).sum()
    ss_tot = ((x - x.mean(dim=0)) ** 2).sum()
    return 1.0 - ss_res / (ss_tot + 1e-8)


def compute_dict_active_fraction(z):
    return (z.abs().sum(dim=0) > 0).float().mean().item()


def compute_dict_usage_entropy(z):
    counts = (z != 0).float().sum(dim=0)
    p = counts / (counts.sum() + 1e-8)
    entropy = -(p * (p + 1e-8).log()).sum()
    return (entropy / np.log(len(p))).item()


# ---------------- logger ----------------

class BottleneckSAELogger:
    """Legacy aggregate logger with sampled raw reconstruction diagnostics.
    
    R-squared centers each feature over the current batch. Epoch values average
    source/batch scores; raw scores are sampled every raw_r2_every steps.
    Shared-code usage uses a Python-random reservoir of at most max_z_samples
    rows. It has no gate/alignment/loss-breakdown methods and is not interface-
    compatible with the current main trainer. Kept for historical experiments.
    """

    def __init__(
        self,
        model_names,
        linear_projectors,
        use_wandb=True,
        raw_r2_every=10,
        max_z_samples=20000,
    ):
        self.model_names = model_names
        self.projectors = linear_projectors
        self.use_wandb = use_wandb and _WANDB_AVAILABLE

        self.raw_r2_every = raw_r2_every
        self.max_z_samples = max_z_samples

        self._global_step = 0
        self._epoch_idx = 0

        
        self._pinv = {}
        for name, proj in linear_projectors.items():
            if hasattr(proj, "linear") and not proj.linear.weight.requires_grad:
                W = proj.linear.weight.detach()
                self._pinv[name] = torch.linalg.pinv(W)

        self._reset_epoch()

        if self.use_wandb:
            wandb.define_metric("trainer/step")
            wandb.define_metric("trainer/epoch")
            wandb.define_metric("step/*", step_metric="trainer/step")
            wandb.define_metric("epoch/*", step_metric="trainer/epoch")

    def _reset_epoch(self):
        self._epoch_loss = []

        self._lin_r2_total = []
        self._lin_r2_shared = []

        self._raw_r2_total = []
        self._raw_r2_shared = []

        self._orthogonality = []

        self._z_reservoir = []
        self._z_seen = 0

    # ---------------- step ----------------

    def start_step(self, agree_lambda=None, aux_prob=None, lr=None):
        self._global_step += 1
        self._step_lr = lr

    def _to_raw(self, name, x_hat):
        proj = self.projectors[name]
        if not hasattr(proj, "linear"):
            return x_hat
        W = proj.linear.weight
        if W.requires_grad:
            with torch.no_grad():
                pinv = torch.linalg.pinv(W)
        else:
            pinv = self._pinv[name]
        return x_hat @ pinv.T

    def log_step_model(
        self,
        name,
        x,
        x_hat,
        x_hat_sh,
        act,
        z_sh,
        z_pr,
        orthogonality,
    ):
        
        self._lin_r2_total.append(compute_r2(x, x_hat).item())
        if x_hat_sh is not None:
            self._lin_r2_shared.append(compute_r2(x, x_hat_sh).item())

        
        if self._global_step % self.raw_r2_every == 0:
            with torch.no_grad():
                act_hat = self._to_raw(name, x_hat)
                self._raw_r2_total.append(compute_r2(act, act_hat).item())
                if x_hat_sh is not None:
                    act_hat_sh = self._to_raw(name, x_hat_sh)
                    self._raw_r2_shared.append(compute_r2(act, act_hat_sh).item())

        
        if orthogonality is not None:
            self._orthogonality.append(float(orthogonality))

        # reservoir sampling on z_sh
        if z_sh is not None:
            z_flat = z_sh.detach().reshape(-1, z_sh.shape[-1])
            for zi in z_flat:
                self._z_seen += 1
                if len(self._z_reservoir) < self.max_z_samples:
                    self._z_reservoir.append(zi.cpu())
                else:
                    j = random.randint(0, self._z_seen - 1)
                    if j < self.max_z_samples:
                        self._z_reservoir[j] = zi.cpu()

    def end_step(self, total_loss):
        # total_loss is expected to be a float here
        loss_val = float(total_loss)
        self._epoch_loss.append(loss_val)

        if self.use_wandb:
            log_data = {
                "step/total_loss": loss_val,
                "step/lr": self._step_lr,
                "step/r2/linear/total_mean": np.mean(
                    self._lin_r2_total[-len(self.model_names):]
                ),
                "trainer/step": self._global_step,
            }

            
            if self._lin_r2_shared:
                log_data["step/r2/linear/shared_mean"] = np.mean(
                    self._lin_r2_shared[-len(self.model_names):]
                )

            wandb.log(log_data)

    # ---------------- epoch ----------------

    def end_epoch(self):
        self._epoch_idx += 1

        epoch_log = {
            "epoch/loss_mean": float(np.mean(self._epoch_loss)),
            "epoch/r2/linear/total_mean": float(np.mean(self._lin_r2_total)),
            "trainer/epoch": self._epoch_idx,
        }
        
        if self._lin_r2_shared:
            epoch_log["epoch/r2/linear/shared_mean"] = float(
                np.mean(self._lin_r2_shared)
            )

        if self._raw_r2_total:
            epoch_log["epoch/r2/raw/total_mean"] = float(
                np.mean(self._raw_r2_total)
            )
        if self._raw_r2_shared:
            epoch_log["epoch/r2/raw/shared_mean"] = float(
                np.mean(self._raw_r2_shared)
            )

        if self._orthogonality:
            epoch_log["epoch/orthogonality_mean"] = float(np.mean(self._orthogonality))

        if self._z_reservoir:
            z = torch.stack(self._z_reservoir)
            epoch_log["epoch/dict_active_fraction_sh"] = compute_dict_active_fraction(z)
            epoch_log["epoch/dict_usage_entropy_sh"] = compute_dict_usage_entropy(z)

        if self.use_wandb:
            wandb.log(epoch_log)

        print(
            f"Epoch {self._epoch_idx:04d} | "
            f"loss {epoch_log['epoch/loss_mean']:.4f} | "
            f"r2_lin_tot {epoch_log['epoch/r2/linear/total_mean']:.4f} | "
            f"r2_raw_tot {epoch_log.get('epoch/r2/raw/total_mean', 0):.4f}"
        )

        self._reset_epoch()
