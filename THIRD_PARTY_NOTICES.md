# Third-party code and acknowledgments

This repository includes components adapted from the following open-source
projects. The notices below describe those incorporated components.

GB-SAE-specific code and additions use the PolyForm Noncommercial License 1.0.0
in [LICENSE](LICENSE). The original third-party components described below retain
their MIT licenses and permissions; the GB-SAE license does not replace them.

## Overcomplete

- Project: https://github.com/KempnerInstitute/overcomplete
- Copyright: Copyright (c) 2023 Thomas Fel
- License: MIT; see [licenses/Overcomplete.txt](licenses/Overcomplete.txt).

This repository adapts SAE encoder and dictionary components, together with
selected supporting utilities, from Overcomplete. Some components were adapted
from the Overcomplete version bundled with UniversalSAE.

Relevant code is located in `gbsae/nn/`, `gbsae/optimization/`, `gbsae/base.py`,
and selected portions of `gbsae/backbones/` and `gbsae/utils/`.

## UniversalSAE (USAE)

- Project: https://github.com/yorkucvil/UniversalSAE
- Copyright: Copyright (c) 2025 UniversalSAE Team
- License: MIT; see [licenses/USAE.txt](licenses/USAE.txt).

The feature-caching and activation-loading pipeline was adapted from
UniversalSAE and extended for this repository. The patch-token interpolation
utility was adapted from
`universal_sae/uni_analysis.py::interpolate_patch_tokens`.

Relevant code is located in `cache/`, `gbsae/data/activations.py`, and
`gbsae/data/tokens.py`. The interpolation utility underlies the configured
patch-grid resizing; the configuration and token-handling interfaces have been
extended for this repository.

## License notices

The original USAE and Overcomplete license texts and copyright notices are
retained in `licenses/`. The `LICENSE-overcomplete` reference in the upstream
USAE license corresponds to `licenses/Overcomplete.txt` in this repository.

## External models and datasets

Pretrained backbone weights and datasets are obtained separately and remain
subject to their respective licenses and terms. They are not included in this
source-code distribution.
