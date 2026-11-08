"""
Loss functions for Bottleneck / Top-K Sparse Autoencoder.

This file ONLY contains loss definitions.
It assumes sparsity is enforced structurally (Top-K),
not via regularization penalties.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict


def gated_auxiliary_reconstruction_l2_loss(
        x,
        x_hat,
        pre_codes,
        z_sh,
        z_pr,
        D_sh,
        D_pr,
        g,
        penalty=0.1,
        enable_aux: bool = True,
):
    """Reconstruction error plus a gated auxiliary residual reconstruction term.
    
    Uses mean squared errors. The auxiliary codes select half of the component
    count from ReLU(pre_codes) - (z_sh + z_pr), then apply g and 1-g
    before shared/private decoding. penalty scales the auxiliary error;
    enable_aux=False or g=None returns only the main reconstruction error.
    """
    residual = x - x_hat
    mse = residual.square().mean()

    if (not enable_aux) or (g is None):
        return mse

    pre = torch.relu(pre_codes)
    codes_total = z_sh + z_pr
    unused = pre - codes_total

    if unused.max() == 0:
        return mse

    aux_topk = torch.topk(unused, k=unused.shape[1] // 2, dim=1)
    unused = torch.zeros_like(unused).scatter(-1, aux_topk.indices, aux_topk.values)

    g_row = g.unsqueeze(0).to(unused.dtype)
    unused_sh = unused * g_row
    unused_pr = unused * (1.0 - g_row)

    residual_hat = (unused_sh @ D_sh) + (unused_pr @ D_pr)
    aux_mse = (residual - residual_hat).square().mean()

    return mse + penalty * aux_mse


def gated_auxiliary_reconstruction_l1_loss(
        x,
        x_hat,
        pre_codes,
        z_sh,
        z_pr,
        D_sh,
        D_pr,
        g,
        penalty=0.1,
        enable_aux: bool = True,
):
    """Reconstruction error plus a gated auxiliary residual reconstruction term.
    
    Uses mean absolute errors. The auxiliary codes select half of the component
    count from ReLU(pre_codes) - (z_sh + z_pr), then apply g and 1-g
    before shared/private decoding. penalty scales the auxiliary error;
    enable_aux=False or g=None returns only the main reconstruction error.
    """
    residual = x - x_hat
    mse = residual.abs().mean()

    if (not enable_aux) or (g is None):
        return mse

    pre = torch.relu(pre_codes)
    codes_total = z_sh + z_pr
    unused = pre - codes_total

    if unused.max() == 0:
        return mse

    aux_topk = torch.topk(unused, k=unused.shape[1] // 2, dim=1)
    unused = torch.zeros_like(unused).scatter(-1, aux_topk.indices, aux_topk.values)

    g_row = g.unsqueeze(0).to(unused.dtype)
    unused_sh = unused * g_row
    unused_pr = unused * (1.0 - g_row)

    residual_hat = (unused_sh @ D_sh) + (unused_pr @ D_pr)
    aux = (residual - residual_hat).abs().mean()

    return mse + penalty * aux


def gate_binarization_loss(g: torch.Tensor) -> torch.Tensor:
    return (g * (1.0 - g)).mean()


def gate_ratio_loss(g: torch.Tensor, target: float) -> torch.Tensor:
    return (g.mean() - float(target)).pow(2)

def shared_latent_alignment_loss(
    z_sh_dict: Dict[str, torch.Tensor],
    eps: float = 1e-8,
    return_breakdown: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Average normalized continuous-Jaccard and overlap-weighted magnitude loss.
    
    Inputs are source-> [N,K] nonnegative shared codes with corresponding rows.
    Each source pair contributes mean(1 - sum(min(patterns))/sum(max(patterns)))
    plus 0.2 times the overlap-weighted raw-code absolute difference. Patterns
    are row-L1-normalized; overlap weights are detached for the magnitude term.
    Pairs are averaged. return_breakdown also returns support and unweighted mag.
    A single source returns zero; the source mapping must not be empty.
    """
    names = list(z_sh_dict.keys())
    if len(names) <= 1:
        zero = torch.tensor(
            0.0, device=next(iter(z_sh_dict.values())).device
        )
        if return_breakdown:
            return zero, {"support": zero, "mag": zero}
        return zero

    shapes = {name: tuple(z.shape) for name, z in z_sh_dict.items()}
    if any(len(shape) != 2 for shape in shapes.values()) or len(set(shapes.values())) != 1:
        raise ValueError(f"Shared-code alignment requires matching [N,K] shapes; got {shapes}")

    loss = 0.0
    cnt = 0
    lambda_mag = 0.2
    support_total = 0.0
    mag_total = 0.0

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            zi = z_sh_dict[names[i]]
            zj = z_sh_dict[names[j]]

            zi_pat = zi / zi.sum(dim=-1, keepdim=True).clamp_min(eps)
            zj_pat = zj / zj.sum(dim=-1, keepdim=True).clamp_min(eps)

            inter = torch.min(zi_pat, zj_pat).sum(dim=-1)
            union = torch.max(zi_pat, zj_pat).sum(dim=-1).clamp_min(eps)
            support_loss = (1.0 - inter / union).mean()

            overlap = torch.min(zi_pat, zj_pat).detach()
            mag_diff = (zi - zj).abs() * overlap
            mag_loss = (
                mag_diff.sum(dim=-1) / overlap.sum(dim=-1).clamp_min(eps)
            ).mean()

            loss += support_loss + lambda_mag * mag_loss
            support_total += support_loss
            mag_total += mag_loss
            cnt += 1

    loss = loss / max(cnt, 1)
    if return_breakdown:
        return loss, {
            "support": support_total / max(cnt, 1),
            "mag": mag_total / max(cnt, 1),
        }
    return loss


def shared_only_reconstruction_loss(
    x: torch.Tensor,
    z_sh: torch.Tensor,
    D,
    recon_loss: str = "l2",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (reconstruction loss, z_sh @ D) for the shared branch alone.
    
    recon_loss selects mean absolute error (l1) or mean squared error (l2).
    """
    x_hat_sh = z_sh @ D

    if recon_loss == "l1":
        loss = (x - x_hat_sh).abs().mean()
    elif recon_loss == "l2":
        loss = (x - x_hat_sh).square().mean()
    else:
        raise ValueError(f"Unknown recon_loss: {recon_loss}")
    return loss, x_hat_sh


def _orient_dictionary_for_orthogonality(D: torch.Tensor) -> torch.Tensor:
    """Apply the retained shape heuristic before row-wise Gram comparisons.
    
    Transpose a 2D matrix when rows exceed columns. The resulting rows are
    the shorter-axis vectors; for overcomplete [components,features] inputs,
    they are feature-axis vectors, not guaranteed dictionary atoms.
    """
    if D.dim() != 2:
        raise ValueError("Dictionary must be 2D")
    if D.shape[0] > D.shape[1]:
        D = D.t()
    return D


def _ortho_sh_sh(D_sh: torch.Tensor) -> torch.Tensor:
    D_sh = _orient_dictionary_for_orthogonality(D_sh)
    D_sh = F.normalize(D_sh, dim=-1)
    G = D_sh @ D_sh.t()
    I = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
    return (G - I).pow(2).mean()


def _ortho_sh_pr(D_sh: torch.Tensor, D_pr: torch.Tensor) -> torch.Tensor:
    D_sh = _orient_dictionary_for_orthogonality(D_sh)
    D_pr = _orient_dictionary_for_orthogonality(D_pr)
    D_sh = F.normalize(D_sh, dim=-1)
    D_pr = F.normalize(D_pr, dim=-1)
    C = D_sh @ D_pr.t()
    return C.pow(2).mean()


def orthogonality_loss(
    D_sh: torch.Tensor,
    D_pr: torch.Tensor | None = None,
    lambda_shsh: float = 1.0,
    lambda_shpr: float = 0.0,
) -> torch.Tensor:
    loss = lambda_shsh * _ortho_sh_sh(D_sh)

    if D_pr is not None and lambda_shpr > 0:
        loss = loss + lambda_shpr * _ortho_sh_pr(D_sh, D_pr)

    return loss


def bottleneck_sae_total_loss(
    *,
    x: torch.Tensor,
    x_hat: torch.Tensor,
    pre_codes: torch.Tensor,
    z_sh: torch.Tensor,
    z_pr: torch.Tensor,
    D_sh: torch.Tensor | None = None,
    D_pr: torch.Tensor | None = None,
    D: torch.Tensor | None = None,
    g: torch.Tensor | None = None,
    penalty: float = 0.1,
    enable_aux: bool = True,
    lambda_sh_only: float = 0.2,
    lambda_shsh: float = 1.0,
    lambda_shpr: float = 0.0,
    recon_loss: str = "l2",
    lambda_gate_bin: float = 0.0,
    lambda_gate_ratio: float = 0.0,
    gate_ratio_target: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return (per-source objective, diagnostic extras) for gated reconstruction.
    
    The active training path supplies D_sh, D_pr and g. It combines total/
    auxiliary reconstruction, optional shared-only reconstruction, orthogonality
    and gate penalties. Cross-source alignment is added by the trainer separately.
    D is a retained legacy argument; the standalone D-only/no-g path does not
    initialize the reconstruction loss in this implementation.
    """
    extras = {}

    if D_sh is not None:
        if recon_loss == "l1":
            if g is not None:
                loss = gated_auxiliary_reconstruction_l1_loss(
                    x, x_hat, pre_codes, z_sh, z_pr, D_sh, D_pr, g,
                    penalty=penalty, enable_aux=enable_aux
                )
                loss_recon = loss
                extras["loss/recon"] = loss_recon.detach()
        else:
            if g is not None:
                loss = gated_auxiliary_reconstruction_l2_loss(
                    x, x_hat, pre_codes, z_sh, z_pr, D_sh, D_pr, g,
                    penalty=penalty, enable_aux=enable_aux
                )
                loss_recon = loss
                extras["loss/recon"] = loss_recon.detach()

    if lambda_sh_only > 0:
        if D_sh is not None:
            sh_loss, x_hat_sh = shared_only_reconstruction_loss(
                x, z_sh, D_sh, recon_loss
            )
        else:
            sh_loss, x_hat_sh = shared_only_reconstruction_loss(
                x, z_sh, D, recon_loss
            )

        loss = loss + lambda_sh_only * sh_loss
        extras["x_hat_sh"] = x_hat_sh
        extras["loss/shared_only"] = (lambda_sh_only * sh_loss).detach()

    if D_sh is not None:
        ortho = orthogonality_loss(
            D_sh=D_sh,
            D_pr=D_pr,
            lambda_shsh=lambda_shsh,
            lambda_shpr=lambda_shpr,
        )
    elif D is not None:
        ortho = orthogonality_loss(
            D_sh=D,
            D_pr=None,
            lambda_shsh=lambda_shsh,
            lambda_shpr=0.0,
        )
    else:
        ortho = None

    if ortho is not None:
        loss = loss + ortho
        extras["orthogonality"] = ortho
        extras["loss/orthogonality"] = ortho.detach()

    if (g is not None) and (lambda_gate_bin > 0.0):
        gb = gate_binarization_loss(g)
        loss = loss + float(lambda_gate_bin) * gb
        extras["gate_bin"] = gb
        extras["loss/gate_bin"] = (lambda_gate_bin * gb).detach()

    if (g is not None) and (lambda_gate_ratio > 0.0):
        gr = gate_ratio_loss(g, target=gate_ratio_target)
        loss = loss + float(lambda_gate_ratio) * gr
        extras["gate_ratio"] = gr
        extras["gate_mean"] = g.mean()
        extras["loss/gate_ratio"] = (lambda_gate_ratio * gr).detach()

    return loss, extras
