import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

from .specs import MethodSpec


@torch.inference_mode()
def evaluate_pair(
    query: torch.Tensor,
    gallery: torch.Tensor,
    labels: torch.Tensor,
    ks: Sequence[int],
    device: str,
    query_chunk_size: int,
) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    """Evaluate same-order paired cross-model embeddings in query chunks.
    
    Instance Recall keeps the diagonal matching image. Category Recall and
    full-ranking AP remove it and exclude singleton-class queries from their
    means. Ranks above gallery size are clipped, not omitted. Returns aggregate
    metrics plus per-query hit/AP tensors; no eligible category queries yields
    NaN category means under the retained implementation.
    """
    if query.shape[0] != gallery.shape[0] or query.shape[0] != labels.shape[0]:
        raise ValueError("Query, gallery, and labels must describe the same ordered samples")
    num_samples = labels.shape[0]
    if num_samples < 2:
        raise ValueError("At least two validation samples are required")

    ks = tuple(sorted(set(int(k) for k in ks)))
    max_instance_k = min(max(ks), num_samples)
    max_category_k = min(max(ks), num_samples - 1)
    query = F.normalize(query.float(), dim=-1)
    gallery = F.normalize(gallery.float(), dim=-1).to(device)
    labels_cpu = labels.long().cpu()
    gallery_labels = labels_cpu.to(device)
    class_counts = torch.bincount(labels_cpu)
    category_valid = class_counts[labels_cpu] > 1

    instance_hits = {k: torch.zeros(num_samples, dtype=torch.bool) for k in ks}
    category_hits = {k: torch.zeros(num_samples, dtype=torch.bool) for k in ks}
    average_precision = torch.zeros(num_samples, dtype=torch.float32)

    for start in range(0, num_samples, query_chunk_size):
        end = min(start + query_chunk_size, num_samples)
        q = query[start:end].to(device)
        similarity = q @ gallery.t()
        local_rows = torch.arange(end - start, device=device)
        target_cols = torch.arange(start, end, device=device)

        instance_order = similarity.topk(max_instance_k, dim=1, largest=True).indices
        for k in ks:
            effective_k = min(k, max_instance_k)
            instance_hits[k][start:end] = (
                instance_order[:, :effective_k].eq(target_cols[:, None]).any(dim=1).cpu()
            )

        # Category retrieval explicitly removes the paired same-image item.
        similarity[local_rows, target_cols] = -torch.inf
        order = similarity.argsort(dim=1, descending=True)
        ordered_labels = gallery_labels[order]
        matches = ordered_labels.eq(labels_cpu[start:end].to(device)[:, None])
        # The removed paired item sorts to the last position. Mask it
        # explicitly so it cannot leak into AP's positive count.
        matches &= order.ne(target_cols[:, None])

        for k in ks:
            effective_k = min(k, max_category_k)
            category_hits[k][start:end] = matches[:, :effective_k].any(dim=1).cpu()

        cumulative_matches = matches.cumsum(dim=1)
        positions = torch.arange(1, num_samples + 1, device=device, dtype=torch.float32)
        precision_at_rank = cumulative_matches.float() / positions
        positive_count = matches.sum(dim=1)
        valid = positive_count > 0
        ap = torch.zeros(end - start, device=device)
        ap[valid] = (precision_at_rank[valid] * matches[valid]).sum(dim=1) / positive_count[valid]
        average_precision[start:end] = ap.cpu()

    metrics: Dict[str, float] = {}
    for k in ks:
        metrics[f"instance_R@{k}"] = float(instance_hits[k].float().mean())
        metrics[f"category_R@{k}"] = float(category_hits[k][category_valid].float().mean())
    metrics["category_mAP"] = float(average_precision[category_valid].mean())
    metrics["num_queries"] = int(num_samples)
    metrics["num_category_queries"] = int(category_valid.sum())
    metrics["num_singleton_category_queries_excluded"] = int((~category_valid).sum())

    per_query = {
        **{f"instance_R@{k}": instance_hits[k] for k in ks},
        **{f"category_R@{k}": category_hits[k] for k in ks},
        "category_valid": category_valid,
        "category_AP": average_precision,
    }
    return metrics, per_query


@torch.inference_mode()
def evaluate_method(
    embeddings: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    spec: MethodSpec,
    ks: Sequence[int],
    device: str,
    query_chunk_size: int,
) -> Dict[str, object]:
    pair_results: Dict[str, Dict[str, float]] = {}
    pair_query_results: List[Dict[str, torch.Tensor]] = []

    for query_name, gallery_name in spec.eval_pairs:
        pair_name = f"{query_name}->{gallery_name}"
        metrics, per_query = evaluate_pair(
            embeddings[query_name],
            embeddings[gallery_name],
            labels,
            ks=ks,
            device=device,
            query_chunk_size=query_chunk_size,
        )
        pair_results[pair_name] = metrics
        pair_query_results.append(per_query)

    metric_names = [
        name
        for name in next(iter(pair_results.values()))
        if name.startswith("instance_R@") or name.startswith("category_R@") or name == "category_mAP"
    ]
    macro = {
        metric_name: sum(result[metric_name] for result in pair_results.values()) / len(pair_results)
        for metric_name in metric_names
    }
    macro["num_directions"] = len(pair_results)

    output: Dict[str, object] = {
        "method": spec.name,
        "summary_kind": spec.summary_kind,
        "pairs": pair_results,
        "macro": macro,
    }

    if spec.name == "gbsae":
        all_hit: Dict[str, float] = {}
        for k in ks:
            instance_stack = torch.stack([row[f"instance_R@{k}"] for row in pair_query_results])
            all_hit[f"instance_AllHit@{k}"] = float(instance_stack.all(dim=0).float().mean())

            category_stack = torch.stack([row[f"category_R@{k}"] for row in pair_query_results])
            valid_stack = torch.stack([row["category_valid"] for row in pair_query_results])
            jointly_valid = valid_stack.all(dim=0)
            all_hit[f"category_AllHit@{k}"] = float(
                category_stack[:, jointly_valid].all(dim=0).float().mean()
            )
        output["all_hit"] = all_hit
    return output


def write_evaluation_results(output_dir: Path, results: Mapping[str, object], stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{stem}.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    pair_rows = []
    for pair_name, metrics in results["pairs"].items():
        pair_rows.append({"pair": pair_name, **metrics})
    if pair_rows:
        fields = list(pair_rows[0])
        with open(output_dir / f"{stem}_pairs.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(pair_rows)

    summary_row = {"method": results["method"], **results["macro"]}
    summary_row.update(results.get("all_hit", {}))
    with open(output_dir / f"{stem}_summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_row))
        writer.writeheader()
        writer.writerow(summary_row)
