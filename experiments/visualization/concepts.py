#!/usr/bin/env python3

"""Retrieve and visualize gate-filtered shared/private concepts across a dataset.

Rank candidate concepts by accumulated top-token strength times tokenwise
contrast, then track top-N images per selected concept and save heatmaps.
SAE forward passes still compute all components; only heaps, selected sample
storage and rendered overlays are restricted to selected concept IDs.
"""
from gbsae.config_cli import ConfigSchema, parse_config_arguments

import sys
import os
import math
from typing import Dict, List
import heapq

import numpy as np
import torch
from torch.utils.data import DataLoader
from einops import rearrange
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from gbsae.data.tokens import build_token_transform, validate_token_correspondence
from gbsae.models.config import build_sae, restore_sae_state
from gbsae.models.config import build_projectors
from gbsae.data.config import build_activation_dataset
from gbsae.utils.plotting import interpolate_cv2


def load_config(path):
    from gbsae.configuration import load_model_config
    return load_model_config(path)


def z_to_heatmap(z: torch.Tensor):
    """Reshape a one-dimensional patch score vector to its square token grid.
    
    The token count must be a perfect square; class/prefix tokens must already
    have been removed by the caller.
    """
    T = z.shape[0]
    s = int(math.sqrt(T))
    assert s * s == T
    return z.reshape(s, s)



def extract_patch_latents(latents, transform, batch_size):
    """Keep configured prefixes in SAE inference; exclude them from spatial maps."""
    patches = {}
    for name, values in latents.items():
        tokens = values.reshape(batch_size, -1, values.shape[-1])
        count = transform.policies[name].retained_prefix_tokens
        tokens = tokens[:, count:]
        side = math.isqrt(tokens.shape[1])
        if not tokens.shape[1] or side * side != tokens.shape[1]:
            raise ValueError(f"{name}: spatial visualization requires a square patch grid, got {tokens.shape[1]} patches")
        patches[name] = tokens.flatten(0, 1)
    return patches


def mean_shared_codes(codes):
    """Average spatially corresponding shared codes across configured sources."""
    if len({tuple(values.shape) for values in codes.values()}) != 1:
        raise ValueError("Shared concept visualization requires matching output patch grids")
    return torch.stack(list(codes.values()), dim=0).mean(dim=0)


def concept_energy(z, k=10):
    """Score each [N,K] code column by top-min(k,N) mean times sample std.
    
    N may span all tokens in a batch during concept selection, or one image's
    tokens during image ranking. The score is not an L1/L2 code energy.
    """
    # top-k mean over tokens
    topk_vals, _ = torch.topk(z, k=min(k, z.shape[0]), dim=0)  # (k, K)
    strength = topk_vals.mean(dim=0)                          # (K,)
    # spatial contrast
    contrast = z.std(dim=0)                                   # (K,)
    return strength * contrast                                # (K,)


def save_overlay(img_path, heatmap_hw, out_path, alpha=0.4):
    """
    heatmap_hw: (H_token, W_token) torch tensor
    """
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    H_img, W_img = img_np.shape[:2]

    hm = heatmap_hw.detach().cpu().numpy()
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)

    
    hm_resized = interpolate_cv2(hm, (W_img, H_img))

    plt.figure(figsize=(5,5))
    plt.imshow(img_np)
    plt.imshow(hm_resized, cmap="viridis", alpha=alpha)
    plt.axis("off")
    plt.savefig(out_path, dpi=160, bbox_inches="tight", pad_inches=0)
    plt.close()


# Concept Heap (only for selected concepts)
class ConceptHeap:
    def __init__(self, concept_ids: List[int], topN: int):
        self.topN = topN
        self.concept_ids = concept_ids
        self.heaps = {k: [] for k in concept_ids}

    def update(self, energy: torch.Tensor, sample_id: int):
        for k in self.concept_ids:
            e = float(energy[k])
            heap = self.heaps[k]

            if len(heap) < self.topN:
                heapq.heappush(heap, (e, sample_id))
            else:
                if e > heap[0][0]:
                    heapq.heapreplace(heap, (e, sample_id))

    def get_topk(self, k):
        return sorted(self.heaps[k], reverse=True)


def main():
    ap = ConfigSchema('Filtered Dataset Concept Visualization for BottleneckSAE')

    
    ap.add_field('sae_ckpt', path='inputs.sae_ckpt', type=str)

    ap.add_field('imagenet_root', path='data.imagenet_root', type=str, inherit=('base', 'data.imagenet_root'))
    ap.add_field('activation_root', path='data.activation_root', type=str, inherit=('base', 'data.activation_root'))
    ap.add_field('split', path='data.split', type=str, inherit=('base', 'data.split'))

    ap.add_field('batch_size', path='loader.batch_size', type=int)
    ap.add_field('num_workers', path='loader.num_workers', type=int)

    ap.add_field('topN', path='evaluation.topN', type=int)
    ap.add_field('top_shared_concepts', path='evaluation.top_shared_concepts', type=int)
    ap.add_field('top_private_concepts', path='evaluation.top_private_concepts', type=int)
    ap.add_field('concept_type', path='evaluation.concept_type', type=str, choices=['all', 'shared', 'private'], help='Which concept branch to run: all, shared, or private')

    ap.add_field('gate_shared_thresh', path='evaluation.gate_shared_thresh', type=float)
    ap.add_field('gate_private_thresh', path='evaluation.gate_private_thresh', type=float)

    ap.add_field('save_dir', path='output.save_dir', type=str)
    ap.add_field('device', path='runtime.device', type=str, inherit=('base', 'runtime.device'))

    args = parse_config_arguments(ap)
    os.makedirs(args.save_dir, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    do_shared = args.concept_type in {"all", "shared"}
    do_private = args.concept_type in {"all", "private"}

    # Load config & checkpoint
    cfg = load_config(args.config)
    transform = build_token_transform(cfg)
    ckpt = torch.load(args.sae_ckpt, map_location=device)

    model_names = cfg["models"]["sources"]

    standardize = cfg["data"].get("standardize", False)
    divide_norm = cfg["data"].get("divide_norm", False)
    if standardize and "standardization_stats" not in ckpt:
        raise ValueError("Checkpoint does not contain standardization_stats")

    std_stats = {
        name: {
            "mean": ckpt["standardization_stats"][name]["mean"].to(device),
            "std": ckpt["standardization_stats"][name]["std"].to(device),
        }
        for name in model_names if standardize
    }

    # Dataset
    dataset = build_activation_dataset(
        cfg, imagenet_root=args.imagenet_root, activation_root=args.activation_root,
        sources=model_names, split=args.split, standardize=False, divide_norm=divide_norm and not standardize,
    )

    if do_shared and len(model_names) > 1:
        if not len(dataset):
            raise ValueError("The shared visualization dataset is empty")
        sample, _ = dataset[0]
        aligned_sample = {name: transform(name, sample[name].unsqueeze(0)) for name in model_names}
        validate_token_correspondence(
            aligned_sample, policies=transform.policies, include_prefixes=False,
            context="Shared concept visualization",
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Projectors
    projectors = build_projectors(cfg, model_names, device)

    for name in model_names:
        projectors[name].load_state_dict(ckpt["linear_projectors"][name])
        projectors[name].to(device).eval()

    # SAE
    sae = build_sae(cfg, model_names, device).eval()

    restore_sae_state(sae, ckpt["model_state_dict"])
    gate_vec = sae.bottleneck.gate().detach().cpu()

    K = gate_vec.numel()

    # Step 1: gate filtering
    shared_ids = torch.where(gate_vec > args.gate_shared_thresh)[0] if do_shared else torch.empty(0, dtype=torch.long)
    private_ids = torch.where(gate_vec < args.gate_private_thresh)[0] if do_private else torch.empty(0, dtype=torch.long)

    if do_shared:
        print(f"[INFO] Gate-filtered shared: {len(shared_ids)}")
    if do_private:
        print(f"[INFO] Gate-filtered private: {len(private_ids)}")

    # Step 2: energy accumulation
    energy_sh_total = torch.zeros(K) if do_shared else None
    energy_pr_total = {name: torch.zeros(K) for name in model_names} if do_private else {}

    print("[INFO] Accumulating dataset energy...")

    for batch_set, _ in tqdm(dataloader):

        proj_acts = {}
        for name in model_names:
            act = batch_set[name].to(device)
            if standardize:
                mean = std_stats[name]["mean"]
                std = std_stats[name]["std"]
                act = (act - mean) / (std + 1e-5)

            act = transform(name, act)
            act = rearrange(act, "b t d -> (b t) d")
            proj_acts[name] = projectors[name](act)

        with torch.no_grad():
            out = sae(proj_acts)
        patch_shared = extract_patch_latents(out.z_sh, transform, batch_set[model_names[0]].shape[0])
        patch_private = extract_patch_latents(out.z_pr, transform, batch_set[model_names[0]].shape[0])

        if do_shared:
            z_sh_mean = mean_shared_codes(patch_shared)
            energy_sh_total += concept_energy(z_sh_mean, k=10).cpu()

        if do_private:
            for name in model_names:
                z_pr = patch_private[name]
                energy_pr_total[name] += concept_energy(z_pr, k=10).cpu()


    # Step 3: pick top-M concepts
    top_shared_ids = torch.empty(0, dtype=torch.long)
    if do_shared:
        top_shared_k = min(args.top_shared_concepts, len(shared_ids))
        top_shared_ids = shared_ids[torch.topk(energy_sh_total[shared_ids], top_shared_k).indices]

    top_private_ids = {}
    if do_private:
        top_private_ids = {
            name: private_ids[torch.topk(
                energy_pr_total[name][private_ids],
                min(args.top_private_concepts, len(private_ids)),
            ).indices]
            for name in model_names
        }

    # build index maps for submatrix caching
    shared_id_to_col = {int(k): i for i, k in enumerate(top_shared_ids.tolist())} if do_shared else {}
    private_id_to_col = {
        name: {int(k): i for i, k in enumerate(top_private_ids[name].tolist())}
        for name in model_names
    } if do_private else {}

    if do_shared:
        print(f"[INFO] Selected shared concepts: {len(top_shared_ids)}")
    if do_private:
        for name in model_names:
            print(f"[INFO] Selected private concepts ({name}): {len(top_private_ids[name])}")


    # Step 4: build heaps only for selected concepts
    heap_sh = ConceptHeap(top_shared_ids.tolist(), args.topN) if do_shared else None
    heap_pr = {name: ConceptHeap(top_private_ids[name].tolist(), args.topN) for name in model_names} if do_private else {}

    print("[INFO] Retrieving top-N samples...")

    for batch_idx, (batch_set, _) in enumerate(tqdm(dataloader)):

        proj_acts = {}
        for name in model_names:
            act = batch_set[name].to(device)

            if standardize:  
                mean = std_stats[name]["mean"]
                std = std_stats[name]["std"]
                act = (act - mean) / (std + 1e-5)

            act = transform(name, act)
            act = rearrange(act, "b t d -> (b t) d")
            proj_acts[name] = projectors[name](act)

        with torch.no_grad():
            out = sae(proj_acts)
        patch_shared = extract_patch_latents(out.z_sh, transform, batch_set[model_names[0]].shape[0])
        patch_private = extract_patch_latents(out.z_pr, transform, batch_set[model_names[0]].shape[0])

        B = batch_set[model_names[0]].shape[0]
        # Slice each image using the aligned SAE output's token count.
        T = patch_shared[model_names[0]].shape[0] // B

        for i in range(B):
            global_id = batch_idx * dataloader.batch_size + i

            if do_shared:
                z_sh_mean = torch.stack(
                    [patch_shared[name][i * T:(i + 1) * T] for name in model_names],
                    dim=0,
                ).mean(dim=0)
                heap_sh.update(concept_energy(z_sh_mean, k=10).cpu(), global_id)

            if do_private:
                for name in model_names:
                    source_tokens = patch_private[name].shape[0] // B
                    z_pr = patch_private[name][i * source_tokens:(i + 1) * source_tokens]

                    #heap_pr[name].update(z_pr.abs().sum(dim=0).cpu(), global_id)
                    heap_pr[name].update(concept_energy(z_pr, k=10).cpu(), global_id)
                

    print("[INFO] Collecting all Top-N sample indices...")

    top_sample_ids = set()

    if do_shared:
        for k in top_shared_ids.tolist():
            for _, idx in heap_sh.get_topk(k):
                top_sample_ids.add(idx)

    if do_private:
        for name in model_names:
            for k in top_private_ids[name].tolist():
                for _, idx in heap_pr[name].get_topk(k):
                    top_sample_ids.add(idx)

    top_sample_ids = sorted(list(top_sample_ids))

    print(f"[INFO] Total unique Top-N samples: {len(top_sample_ids)}")

    print("[INFO] Forwarding Top-N samples for visualization...")

    z_cache = {}

    for idx in tqdm(top_sample_ids):
        act_dict, _ = dataset[idx]

        proj_acts = {}
        for name in model_names:
            act = act_dict[name].to(device)

            if standardize:
                mean = std_stats[name]["mean"]
                std = std_stats[name]["std"]
                act = (act - mean) / (std + 1e-5)

            act = transform(name, act.unsqueeze(0)).squeeze(0)

            proj = projectors[name](act)
            proj_acts[name] = proj

        with torch.no_grad():
            out = sae(proj_acts)
        patch_shared = extract_patch_latents(out.z_sh, transform, 1)
        patch_private = extract_patch_latents(out.z_pr, transform, 1)

        z_cache[idx] = {}

        if do_shared:
            z_sh_mean_full = mean_shared_codes({name: patch_shared[name].detach().cpu() for name in model_names})
            z_cache[idx]["z_sh_mean"] = z_sh_mean_full[:, top_shared_ids]

        if do_private:
            for name in model_names:
                # full z
                z_pr_full = patch_private[name].detach().cpu()

                # slice only target concepts
                z_pr_sub = z_pr_full[:, top_private_ids[name]]  # (T, top_private_concepts)

                z_cache[idx][name] = {
                    "z_pr": z_pr_sub
                }

    # Save Shared Results
    print("[INFO] Saving results...")

    samples = dataset.samples

    if do_shared:
        shared_dir = os.path.join(args.save_dir, "shared")
        os.makedirs(shared_dir, exist_ok=True)

        for k in top_shared_ids.tolist():
            concept_dir = os.path.join(shared_dir, f"concept_{k:04d}")
            os.makedirs(concept_dir, exist_ok=True)

            topk = heap_sh.get_topk(k)

            with open(os.path.join(concept_dir, "summary.txt"), "w") as f:
                for rank, (score, idx) in enumerate(topk):
                    img_path, _ = samples[idx]
                    f.write(f"{rank + 1}: {img_path}  score={score:.4f}\n")

                    # ---------- get z_sh heatmap ----------
                    col = shared_id_to_col[int(k)]
                    z = z_cache[idx]["z_sh_mean"][:, col]
                    heatmap = z_to_heatmap(z)

                    # ---------- save overlay ----------
                    out_path = os.path.join(concept_dir, f"rank{rank + 1}_overlay.png")
                    save_overlay(img_path, heatmap, out_path)


    # Save Private Results
    if do_private:
        private_root = os.path.join(args.save_dir, "private")
        os.makedirs(private_root, exist_ok=True)

        print("[INFO] Saving private concept visualizations...")

        for name in model_names:
            print(f"[INFO] Processing private concepts for backbone: {name}")

            backbone_dir = os.path.join(private_root, name)
            os.makedirs(backbone_dir, exist_ok=True)

            for k in top_private_ids[name].tolist():
                concept_dir = os.path.join(backbone_dir, f"concept_{k:04d}")
                os.makedirs(concept_dir, exist_ok=True)

                topk = heap_pr[name].get_topk(k)

                with open(os.path.join(concept_dir, "summary.txt"), "w") as f:
                    for rank, (score, idx) in enumerate(topk):
                        img_path, _ = samples[idx]
                        f.write(f"{rank + 1}: {img_path}  score={score:.4f}\n")

                        # ---------- get z_pr heatmap ----------
                        col = private_id_to_col[name][int(k)]
                        z = z_cache[idx][name]["z_pr"][:, col]
                        heatmap = z_to_heatmap(z)

                        # ---------- save overlay ----------
                        out_path = os.path.join(concept_dir, f"rank{rank + 1}_overlay.png")
                        save_overlay(img_path, heatmap, out_path)


    print("[DONE] Filtered concept retrieval finished.")


if __name__ == "__main__":
    main()

