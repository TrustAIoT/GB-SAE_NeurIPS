"""Shared construction and checkpoint checks for configured GB-SAE models."""
from gbsae.data.tokens import source_names
from .gb_sae import GatedBottleneckSAE
from .projectors import build_linear_projector


def configured_sources(cfg, sources=None):
    """Select an ordered nonempty subset of the configured feature sources."""
    available = source_names(cfg)
    selected = available if sources is None else list(sources)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("sources must be nonempty and unique")
    if any(name not in available for name in selected):
        raise ValueError(f"Selected sources {selected} are not a subset of {available}")
    return selected


def build_sae(cfg, sources=None, device="cpu"):
    """Construct the training architecture, optionally selecting private decoders."""
    names = configured_sources(cfg, sources)
    options = cfg["sae"]
    return GatedBottleneckSAE(
        input_dim=cfg["projector"]["output_dim"], n_components=options["n_components"],
        model_names=names, k=options["k"],
        encoder_module=options.get("encoder_module", "linear"),
        dictionary_norm=options.get("dictionary_norm", "l2"), device=device,
    ).to(device)


def build_projectors(cfg, sources=None, device="cpu"):
    """Construct source-ordered input projections with the training settings."""
    names = configured_sources(cfg, sources)
    options = cfg["projector"]
    projectors = build_linear_projector(
        options["type"], {s: cfg["models"]["input_dims"][s] for s in names},
        options["output_dim"], **options.get("kwargs", {}),
    )
    for projector in projectors.values():
        projector.to(device)
    return projectors


def restore_sae_state(sae, state_dict):
    """Require the full selected architecture; allow only unused private decoders.

    A single-source evaluator can load a multi-source checkpoint. All encoder,
    gate, shared dictionary and selected private keys must match; architecture
    mismatches never silently leave random shared parameters in the evaluator.
    """
    expected = set(sae.state_dict())
    missing = expected - state_dict.keys()
    unexpected = set(state_dict) - expected
    selected_prefixes = tuple(f"decoder_pr.{name}." for name in sae.model_names)
    invalid = {key for key in unexpected
               if not key.startswith("decoder_pr.") or key.startswith(selected_prefixes)}
    if missing or invalid:
        raise ValueError(f"SAE checkpoint does not match configured architecture: missing={sorted(missing)}, unexpected={sorted(invalid)}")
    return sae.load_state_dict(state_dict, strict=False)
