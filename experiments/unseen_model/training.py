"""Reconstruction weighting local to new-source adapter training."""
import math
from types import FunctionType

from gbsae.models.losses import bottleneck_sae_total_loss
from gbsae.training.train import train_bottleneck_sae


def make_weighted_loss(original, lambda_total):
    """Weight Total MSE and auxiliary residual reconstruction together."""
    alpha = float(lambda_total)
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("loss.lambda_total must be finite and positive")

    def weighted(**kwargs):
        if alpha == 1.0:
            return original(**kwargs)
        if kwargs.get("recon_loss", "l2") != "l2":
            raise ValueError("Weighted adapter reconstruction requires loss.recon_loss=l2")
        arguments = dict(kwargs)
        arguments["penalty"] = alpha * arguments.get("penalty", 0.1)
        loss, extras = original(**arguments)
        mse = (arguments["x"] - arguments["x_hat"]).square().mean()
        loss = loss + (alpha - 1.0) * mse
        extras = dict(extras)
        auxiliary_weighted = extras["loss/recon"] - mse.detach()
        extras["loss/recon"] = alpha * mse.detach() + auxiliary_weighted
        extras["loss/total_mse_weighted"] = alpha * mse.detach()
        extras["loss/auxiliary_weighted"] = auxiliary_weighted
        extras["loss/recon_unweighted"] = mse.detach() + auxiliary_weighted / alpha
        return loss, extras

    return weighted


def train_weighted_adapter(*, lambda_total=1.0, **kwargs):
    """Reuse the shared loop with an adapter-local loss binding."""
    namespace = dict(train_bottleneck_sae.__globals__)
    namespace["bottleneck_sae_total_loss"] = make_weighted_loss(
        bottleneck_sae_total_loss, lambda_total,
    )
    loop = FunctionType(
        train_bottleneck_sae.__code__, namespace, train_bottleneck_sae.__name__,
        train_bottleneck_sae.__defaults__, train_bottleneck_sae.__closure__,
    )
    loop.__kwdefaults__ = train_bottleneck_sae.__kwdefaults__
    return loop(**kwargs)
