"""Batch-derived cosine retrieval, including PROJECTED/SHARED/PRIVATE.

Zero-shot category Recall retains the paired image, as in the original batch
script. This differs from the separate trained-head downstream experiment.
"""
import math
from pathlib import Path
import numpy as np
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

from gbsae.models.config import build_sae, restore_sae_state
from gbsae.models.config import build_projectors
from gbsae.data.config import build_activation_dataset
from ..common import align_activations, model_names

SELECT_KS = [1, 5, 10, 50]

@torch.no_grad()
def l2_normalize(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def pool_tokens(x_tok, mode="mean"):
    if mode == "mean":
        return x_tok.mean(dim=0)
    if mode == "max":
        return x_tok.max(dim=0).values
    if mode == "sum":
        return x_tok.sum(dim=0)
    raise ValueError(f"Unknown pooling mode: {mode}")


@torch.no_grad()
def instance_recall_curve(query, gallery, max_k=50):
    """Return instance Recall@k for paired rows of CPU query/gallery embeddings.
    
    The diagonal item is the matching instance. Rank is one plus the number
    of strictly larger cosine scores, so score ties share the better rank.
    Returns NumPy arrays (k_values, recalls), capped at gallery size.
    """
    query = l2_normalize(query)
    gallery = l2_normalize(gallery)
    sim = query @ gallery.t()
    diag = sim.diag().unsqueeze(1)
    ranks = (sim > diag).sum(dim=1) + 1
    ks = torch.arange(1, min(max_k, sim.shape[1]) + 1)
    recall = torch.stack([(ranks <= k).float().mean() for k in ks])
    return ks.numpy(), recall.numpy()


@torch.no_grad()
def category_recall_curve(query, gallery, q_labels, g_labels, max_k=50):
    """Return category Recall@k using the first same-label gallery hit.
    
    Every query class must occur in the gallery; otherwise the retained argmax
    implementation yields a spurious rank-one hit. Paired same-image matches are
    included. Inputs are CPU tensors; returns NumPy k/recall arrays.
    """
    query = l2_normalize(query)
    gallery = l2_normalize(gallery)
    sim = query @ gallery.t()
    idx = sim.argsort(dim=1, descending=True)
    labels = g_labels[idx]
    hit = torch.argmax((labels == q_labels[:, None]).float(), dim=1) + 1
    ks = torch.arange(1, min(max_k, sim.shape[1]) + 1)
    recall = torch.stack([(hit <= k).float().mean() for k in ks])
    return ks.numpy(), recall.numpy()


def compute_instance_random_baseline(num_gallery, ks):
    ks = np.asarray(ks, dtype=np.int64)
    return {int(k): float(k / num_gallery) for k in ks}


def _log_comb(n, k):
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def compute_category_random_baseline(q_labels, g_labels, ks):
    """Expected Recall@k under uniform gallery sampling without replacement.
    
    Each query label must occur in the gallery. Average 1 - C(N-m,k)/C(N,k)
    over queries, where m is the gallery count for its class, including its
    paired image when present.
    """
    ks = np.asarray(ks, dtype=np.int64)
    g_labels = torch.as_tensor(g_labels, dtype=torch.long)
    q_labels = torch.as_tensor(q_labels, dtype=torch.long)
    num_gallery = int(g_labels.numel())
    unique_labels, counts = torch.unique(g_labels, return_counts=True)
    count_map = {
        int(label): int(count) for label, count in zip(unique_labels.tolist(), counts.tolist())
    }
    positives_per_query = [count_map[int(label)] for label in q_labels.tolist()]

    baseline = {}
    for k in ks:
        k = int(k)
        hit_probs = []
        for m in positives_per_query:
            if m <= 0:
                hit_probs.append(0.0)
            elif k > num_gallery - m:
                hit_probs.append(1.0)
            else:
                miss_logprob = _log_comb(num_gallery - m, k) - _log_comb(num_gallery, k)
                hit_probs.append(1.0 - math.exp(miss_logprob))
        baseline[k] = float(np.mean(hit_probs))
    return baseline


def build_retrieval_dataset(cfg, imagenet_root, activation_root, split, model_names):
    return build_activation_dataset(
        cfg, model_names, imagenet_root=imagenet_root, activation_root=activation_root,
        split=split, standardize=cfg["data"].get("standardize", False),
    )


def build_retrieval_sae(cfg, model_names, ckpt, device):
    sae = build_sae(cfg, model_names, device).eval()
    restore_sae_state(sae, ckpt["model_state_dict"])
    return sae


def build_retrieval_projectors(cfg, ckpt, model_names, device):
    projectors = build_projectors(cfg, model_names, device)
    for n in model_names:
        projectors[n].load_state_dict(ckpt["linear_projectors"][n])
        projectors[n].to(device).eval()
    return projectors


@torch.no_grad()
def extract_embeddings(
    dataloader,
    model_names,
    projectors,
    sae,
    device,
    repr_type,
    pool="mean",
    max_images=None,
    *,
    cfg,
):
    """Pool aligned projected, latent or branch-reconstructed tokens per image.
    
    repr_type selects proj, z_sh, z_pr, xhat_sh or xhat_pr. All paths execute
    the SAE forward pass; even proj requires compatible SAE inputs. Returns
    CPU source-> [images,features] embeddings and labels in dataset order.
    max_images truncates the final batch to the requested total.
    """
    if repr_type == "xhat_sh":
        D_sh = sae.decoder_sh.get_dictionary().detach()
        if D_sh.shape[0] != sae.n_components:
            D_sh = D_sh.t()
        D_sh = D_sh.to(device)

    if repr_type == "xhat_pr":
        D_pr = {}
        for name in model_names:
            d = sae.decoder_pr[name].get_dictionary().detach()
            if d.shape[0] != sae.n_components:
                d = d.t()
            D_pr[name] = d.to(device)

    out_buf = {n: [] for n in model_names}
    labels_buf = []
    n_seen = 0

    for batch_set, labels in tqdm(dataloader, desc=f"Extract {repr_type}", dynamic_ncols=True):
        B, T = batch_set[model_names[0]].shape[:2]
        if max_images is not None and n_seen >= max_images:
            break

        token_acts = {}
        for name in model_names:
            act = batch_set[name].to(device)
            act = align_activations(cfg, act, name)
            token_acts[name] = act

        # Pool each source independently; retained prefixes may change token counts.
        B, T = token_acts[model_names[0]].shape[:2]
        if any(tokens.shape[0] != B for tokens in token_acts.values()):
            raise ValueError("Model activations have different batch sizes")
        proj_acts = {}
        for name in model_names:
            tok = token_acts[name]
            tok_flat = rearrange(tok, "b t d -> (b t) d")
            proj_acts[name] = projectors[name](tok_flat)

        out = sae(proj_acts)
        take = B if max_images is None else min(B, max_images - n_seen)
        labels_buf.extend(labels[:take].tolist())

        for name in model_names:
            T = token_acts[name].shape[1]
            zsh = rearrange(out.z_sh[name], "(b t) k -> b t k", b=B, t=T)
            zpr = rearrange(out.z_pr[name], "(b t) k -> b t k", b=B, t=T)

            if repr_type == "proj":
                tok_repr = rearrange(proj_acts[name], "(b t) d -> b t d", b=B, t=T)
            elif repr_type == "z_sh":
                tok_repr = zsh
            elif repr_type == "z_pr":
                tok_repr = zpr
            elif repr_type == "xhat_sh":
                tok_repr = torch.matmul(zsh, D_sh)
            elif repr_type == "xhat_pr":
                tok_repr = torch.matmul(zpr, D_pr[name])
            else:
                raise ValueError(repr_type)

            for i in range(take):
                out_buf[name].append(pool_tokens(tok_repr[i], pool).cpu())

        n_seen += take

    return {n: torch.stack(out_buf[n]) for n in model_names}, torch.tensor(labels_buf)


def evaluate_retrieval(
    cfg,
    ckpt,
    device,
    imagenet_root,
    activation_root,
    split,
    batch_size,
    num_workers,
    max_images,
    pool,
    shared_repr="xhat_sh",
    private_repr="xhat_pr",
):
    names = model_names(cfg)
    if len(names) < 2:
        raise ValueError("Cross-model retrieval requires at least two configured sources")
    dataset = build_retrieval_dataset(cfg, imagenet_root, activation_root, split, names)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    projectors = build_retrieval_projectors(cfg, ckpt, names, device)
    sae = build_retrieval_sae(cfg, names, ckpt, device)

    emb_proj, labels = extract_embeddings(
        loader, names, projectors, sae, device, "proj", pool=pool, max_images=max_images, cfg=cfg
    )
    emb_sh, _ = extract_embeddings(
        loader, names, projectors, sae, device, shared_repr, pool=pool, max_images=max_images, cfg=cfg
    )
    emb_pr, _ = extract_embeddings(
        loader, names, projectors, sae, device, private_repr, pool=pool, max_images=max_images, cfg=cfg
    )

    methods = {"PROJECTED": emb_proj, "SHARED": emb_sh, "PRIVATE": emb_pr}
    pairs = [(a, b) for a in names for b in names if a != b]

    inst_curves = {k: [] for k in methods}
    cat_curves = {k: [] for k in methods}
    max_k = max(SELECT_KS)

    for a, b in pairs:
        for name, emb in methods.items():
            _, inst_curve = instance_recall_curve(emb[a], emb[b], max_k)
            _, cat_curve = category_recall_curve(emb[a], emb[b], labels, labels, max_k)
            inst_curves[name].append(inst_curve)
            cat_curves[name].append(cat_curve)

    ks = np.arange(1, min(max_k, len(labels)) + 1)
    inst_mean = {k: np.stack(v).mean(0) for k, v in inst_curves.items()}
    cat_mean = {k: np.stack(v).mean(0) for k, v in cat_curves.items()}
    random_instance = compute_instance_random_baseline(len(labels), ks)
    random_category = compute_category_random_baseline(labels, labels, ks)

    return {
        "instance": {
            "curves": inst_mean,
            "random": random_instance,
        },
        "category": {
            "curves": cat_mean,
            "random": random_category,
        },
    }


def build_retrieval_rows(ckpt_path, retrieval_metrics):
    """Build checkpoint/branch Recall rows while retaining the original CSV schema.
    
    The legacy accuracy field stores Recall@k as a fraction. Ranks exceeding
    gallery size are omitted; shared_vs_random_ratio is populated only for SHARED.
    """
    rows = []
    for task_name in ["instance", "category"]:
        curves = retrieval_metrics[task_name]["curves"]
        random_baseline = retrieval_metrics[task_name]["random"]
        for branch in ["PROJECTED", "SHARED", "PRIVATE"]:
            for k in SELECT_KS:
                if k > len(curves[branch]):
                    continue
                accuracy = float(curves[branch][k - 1])
                rand_value = float(random_baseline[k])
                row = {
                    "checkpoint_path": ckpt_path,
                    "checkpoint_name": Path(ckpt_path).name,
                    "task": task_name,
                    "branch": branch,
                    "recall_at_k": k,
                    "accuracy": accuracy,
                    "random_baseline": rand_value,
                    "shared_vs_random_ratio": (accuracy / rand_value) if branch == "SHARED" and rand_value > 0 else "",
                }
                rows.append(row)
    return rows

