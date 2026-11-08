from dataclasses import dataclass
from itertools import permutations
from typing import Dict, Tuple


FUSED_VIEW = "FUSED"


@dataclass(frozen=True)
class MethodSpec:
    name: str
    views: Tuple[str, ...]
    head_by_view: Dict[str, str]
    train_pairs: Tuple[Tuple[str, str], ...]
    eval_pairs: Tuple[Tuple[str, str], ...]
    expected_head_count: int
    summary_kind: str


def get_method_spec(method: str, sources) -> MethodSpec:
    """Build directed views/head sharing for an ordered configured ensemble."""
    sources = tuple(sources)
    if len(sources) < 2 or len(sources) != len(set(sources)) or FUSED_VIEW in sources:
        raise ValueError("Retrieval requires at least two unique sources, excluding reserved name FUSED")
    method = method.lower()
    if method == "projected":
        pairs = tuple(permutations(sources, 2))
        return MethodSpec(
            name=method,
            views=sources,
            head_by_view={name: name for name in sources},
            train_pairs=pairs,
            eval_pairs=pairs,
            expected_head_count=len(sources),
            summary_kind="directed_pair_macro",
        )
    if method == "gbsae":
        # One tied retrieval head is applied to the fused query and to each
        # per-model shared reconstruction. Training is symmetric; the primary
        # evaluation directions are FUSED -> each single model.
        train_pairs = tuple(
            pair
            for model_name in sources
            for pair in ((FUSED_VIEW, model_name), (model_name, FUSED_VIEW))
        )
        eval_pairs = tuple((FUSED_VIEW, model_name) for model_name in sources)
        return MethodSpec(
            name=method,
            views=sources + (FUSED_VIEW,),
            head_by_view={name: "GBSAE_FUSED" for name in sources + (FUSED_VIEW,)},
            train_pairs=train_pairs,
            eval_pairs=eval_pairs,
            expected_head_count=1,
            summary_kind="fused_to_single_macro_and_all_hit",
        )
    raise ValueError(f"Unknown method {method!r}; expected projected or gbsae")



