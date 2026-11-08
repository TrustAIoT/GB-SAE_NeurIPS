# GB-SAE training and model API

This package contains the shared/private SAE, token preprocessing, losses and
training components. Frozen backbone feature generation is in [cache/](../cache/README.md).

## Train with a YAML

```bash
python -m gbsae.train --config configs/train/config_i.yaml
```

Start from an ensemble recipe under `configs/train/`. Set source names and raw
widths, image/cache roots, token policies and output paths before training.
The complete parameter reference is in [configs/README.md](../configs/README.md).
The training entry assembles the dataset, DataLoader, token transform, projectors,
SAE, AdamW/cosine schedule and logger.

## Token layout and alignment

```yaml
models:
  sources: [TransformerA, ConvB]
  input_dims: {TransformerA: 768, ConvB: 1024}
data:
  prefix_tokens: {TransformerA: 1, ConvB: 0}
  use_class_tokens: false
  activation_layouts: {TransformerA: tokens, ConvB: channels_first}
  token_alignment:
    TransformerA: {input_grid: 16, output_grid: 7}
    ConvB: {input_grid: 7, output_grid: 7}
```

This fragment describes a transformer cache with 1 prefix + 256 patch tokens and
a prefix-free `[1024,7,7]` convolutional cache. Layout conversion and prefix
removal happen in the dataset, followed by original-grid standardization or
per-token L2 division. The batch token transform then interpolates patches,
preserves any retained prefixes, and passes `[B,T,D]` to projection and SAE.
The dataset does not resize tokens when estimating statistics.

Every source needs an explicit prefix count, retention choice and alignment
entry. `use_class_tokens` also accepts a source-to-bool mapping. A `null`
alignment entry skips resizing; otherwise input/output grids are square side
lengths excluding prefixes. Different input grids are allowed.

With multiple sources and `loss.lambda_align > 0`, all transformed patch grids
and retained prefix counts must match. `output_grid` values must therefore agree
for spatially aligned sources. Matching total token counts does not establish
correspondence when prefixes or patch grids differ. Training validates a sample
before construction and each aligned batch before SAE inference.

Shared spatial visualization also requires common output patch grids, but can
exclude different prefix counts. Single-source adapters, per-source evaluation
and separately pooled image representations do not require cross-source grid
equality. Each source must still satisfy its own declared layout and token count.

## Use the model directly

```python
import torch
from gbsae.models import GatedBottleneckSAE

sources = ["A", "B"]
model = GatedBottleneckSAE(
    input_dim=128, n_components=512, model_names=sources,
    k=16, encoder_module="mlp_bn_1", dictionary_norm="l2",
)
model.eval()
with torch.no_grad():
    out = model({name: torch.randn(8, 128) for name in sources})
    shared = out.z_sh["A"] @ out.D_sh
    private = out.z_pr["A"] @ out.D_pr["A"]
```

Direct inputs are projected token matrices `[N,input_dim]`, not raw backbone
features or images. `out` contains `pre_codes`, `z_sh`, `z_pr`, `recon`, `D_sh`,
`D_pr` and the shared gate `g`. Code matrices have `n_components` columns;
dictionaries have shape `[n_components,input_dim]`. `k` selects the total Top-K
support before the gate divides contributions into shared and private codes.

## Assemble or extend training

| Module | API / responsibility |
| --- | --- |
| `data.config` | `build_activation_dataset(cfg, sources=None, **data_overrides)` |
| `data.tokens` | `build_token_transform(cfg, sources=None)` and token-correspondence checks |
| `data.preprocessing` | Apply configured preprocessing to cached tensors; select patch tokens for spatial analysis |
| `models.config` | `build_projectors`, `build_sae`, `restore_sae_state` |
| `training.components` | Model, optimizer/scheduler and logger builders |
| `training.train` | `train_bottleneck_sae` and checkpoint writing |
| `nn.factory` | `EncoderFactory.register_module` and `list_modules` |

```python
from gbsae.train import load_config, train
from gbsae.data.config import build_activation_dataset
from gbsae.data.tokens import build_token_transform

cfg = load_config("configs/train/config_i.yaml")
dataset = build_activation_dataset(cfg)
transform = build_token_transform(cfg)
features, label = dataset[0]
aligned = {s: transform(s, x.unsqueeze(0)) for s, x in features.items()}
# Run the assembled training workflow when ready:
# train(cfg, device="cuda")
```

For custom data or joint projector optimization, build the components and supply
your own loader/optimizer to `train_bottleneck_sae`. Its
`token_transform(source, tokens)` callback receives `[B,T,D]` after loader
preprocessing and returns `[B,T',D]` before projection; `None` skips alignment.
A custom shared-alignment transform must preserve image and token correspondence.
The supplied main optimizer includes SAE parameters only.

## Checkpoints and evaluation

For the pretrained Config I checkpoint, download information and matching configuration,
see [the root README](../README.md#pretrained-config-i-checkpoint).

`run.save_dir` receives `training_config.yaml` and
`<run.name>_epoch_<zero-based-epoch>.pth`. Checkpoints store `model_state_dict`,
`linear_projectors`, `optimizer_state_dict`, `epoch`, `learning_rate` and optional
`standardization_stats`. Use the matching training YAML when constructing a model
for evaluation. `restore_sae_state` checks shared and selected-private weights;
a source subset may leave unrelated private dictionaries unused.

Experiment modules choose statistics and train/eval modes for their task. Use
[main metrics](../experiments/main_metrics/README.md),
[attribution](../experiments/attribution/README.md), or
[downstream classification](../experiments/downstream/downstream_classification/README.md)
for their complete run settings and outputs.
