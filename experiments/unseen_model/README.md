# Train an adapter for a new source

Connect one new cached-feature source to a trained GB-SAE. Train its input
projector and private dictionary while freezing encoder, gate and shared-dictionary
parameters. Backbone weights are neither loaded nor trained here.

## Run

Generate the new source's activation cache first. Copy
`configs/unseen_model/adapter_template.yaml`, edit it and run from the repository
root. The supplied CLIP jobs use the same entry:

```bash
python -m experiments.unseen_model.train_adapter --config configs/unseen_model/clip_sh05.yaml
python -m experiments.unseen_model.train_adapter --config configs/unseen_model/clip_sh08.yaml
```

The base model comes from `gbsae_config` and `adapter.shared_ckpt`. Its architecture,
projector output width, normalization strategy, component count and `k` are read
from the base training recipe. The adapter YAML supplies the new source and its
own optimizer/loss/training/output choices.

The CLIP recipes use `(lambda_sh_only, lambda_total) = (0.5, 0.1)` and
`(0.8, 0.1)`, respectively.

## CLIP training objective

The adapter weights Total reconstruction and auxiliary residual reconstruction
together:

\[
L=\lambda_{\mathrm{total}}(L_{\mathrm{MSE,total}}+0.1L_{\mathrm{aux}})
  +\lambda_{\mathrm{sh-only}}L_{\mathrm{MSE,shared}}+L_{\mathrm{regularization}}.
\]

`lambda_total` applies to those two reconstruction terms. Shared-only, dictionary
orthogonality and gate penalties retain their own coefficients. The adapter-local
training wrapper supplies this loss to the shared loop.

The two CLIP recipes use these common settings:

| Setting | Value |
| --- | --- |
| `loss.lambda_total` | 0.1 |
| `loss.recon_loss` | `l2` |
| `loss.lambda_shpr` | 0 |
| `loss.lambda_gate_bin` | 0.2 |
| `loss.lambda_gate_ratio`, `loss.gate_ratio_target`, `loss.lambda_align` | 0 |
| `optimizer.lr` | 0.0003 |
| `optimizer.final_lr`, `optimizer.warmup_lr` | 0.000001 |
| `optimizer.warmup_pct` | 0.05 |
| `optimizer.weight_decay` | 0.00001 |
| `schedule.warmup_pct`, `schedule.steady_pct` | 0 / 0.1 |
| `training.batch_size`, `training.epochs`, `training.save_every` | 32 / 10 / 1 |

The CLIP entry trains on the selected training split. Its default logger records
projected Total/Shared R-squared, losses, gate statistics and dictionary usage;
it performs no raw-space back-projection or pseudoinverse calculation. This entry
has no separate validation pass. Epoch R-squared values average the training
batch scores. Initialization and seed placement follow the existing adapter
training order.

## New-source configuration

```yaml
gbsae_config: ../train/config_i.yaml
models:
  sources: [NewModel]
  input_dims: {NewModel: 1024}
adapter:
  shared_ckpt: artifacts/config_i/weights/gbsae_config_i_epoch_29.pth
projector:
  type: learnable
  kwargs: {init: orthogonal}
data:
  imagenet_root: data/imagenet
  activation_root: data/activations/new_model
  split: train
  combined_npz: false
  use_class_tokens: false
  prefix_tokens: {NewModel: 1}
  activation_layouts: {NewModel: tokens}
  token_alignment:
    NewModel: {input_grid: 16, output_grid: 14}
```

This is a source-definition fragment; use the supplied template for a complete
job including loss, optimizer, schedule, training and output settings.

| Adapter field | Meaning / requirement |
| --- | --- |
| `gbsae_config` | Base GB-SAE training YAML, relative to the adapter YAML. |
| `models.sources` | Exactly one source identifier; used for cache keys and new projector/private dictionary names. |
| `models.input_dims` | Raw cached feature width for that source. |
| `adapter.shared_ckpt` | Checkpoint whose shared architecture matches the base recipe. |
| `projector.type` | Usually `learnable`; `frozen_random`/`identity` can be used for private-dictionary-only studies. Identity requires matching input/output widths. |
| `projector.kwargs.init` | Learnable projector initialization: `orthogonal` or `xavier`. Frozen-projector `mode`/`scale` are described in the common guide. |
| `data.imagenet_root`, `data.activation_root`, `data.split`, `data.target_class` | New source's image/cache selection and optional class filter. |
| `data.combined_npz` | Source array in `<stem>_combined.npz` when true, or `activation` in `<stem><Source>.npz` when false. |
| `data.activation_layouts` | `tokens`, `channels_first` or `channels_last`; spatial layout conversion only flattens. |
| `data.prefix_tokens` | Cached leading CLS/register token count, explicitly supplied for the new source. |
| `data.use_class_tokens` | Keep or remove those prefixes; bool or source-to-bool mapping. |
| `data.token_alignment` | Source entry `{input_grid: S, output_grid: G}` or `null` to skip interpolation. Grids are square side lengths, excluding prefixes. |
| `loss.lambda_align` | Must be 0: a single-source run has no cross-source alignment pair. |
| `loss.lambda_total` | Positive finite weight on Total reconstruction plus auxiliary residual reconstruction. |
| `run.save_dir`, `run.name` | Adapter output directory and checkpoint filename prefix. |
| `runtime.device` | Execution device. |

Other local loss weights, optimizer, gate schedule, loader/training and logger
fields have the meanings in [the training parameter reference](../../configs/README.md).
The base recipe supplies `sae` and `projector.output_dim`; do not duplicate those
shared fields in the adapter YAML.

## Token and checkpoint requirements

The cache source key, feature width, prefix count and input grid must match the
actual arrays. Built-in CLIP cache features contain 1 CLS + 196 patch tokens, so
its input grid is 14. Normalization uses the new source's original-grid tokens,
then its configured interpolation runs before projection.

The adapter's output grid does not have to equal the base ensemble's grid:
encoding and reconstruction operate on each token independently. Another
experiment that compares spatial token positions may impose correspondence.

Shared checkpoint keys must match the configured encoder/gate/shared dictionary.
A target private dictionary already present in the checkpoint is loaded;
otherwise it is initialized for this source. The projector is initialized from
its configuration for this training invocation. Parameters in shared modules stay
frozen; the trainer uses SAE `train()` mode, so BatchNorm running buffers can update.
Use `run.logger_type: full` with the supplied trainer.

## Outputs and reusable components

`run.save_dir` receives `adapter_config.yaml` and
`<run.name>_epoch_<zero-based-epoch>.pth`. The checkpoint contains SAE weights,
source projector, optimizer state, epoch, learning rate and dataset statistics.
Use the adapter configuration with its checkpoint for later new-source evaluation.

| Function in `adapter.py` | Responsibility |
| --- | --- |
| `get_adapter_source` | Validate single-source and zero-alignment settings |
| `build_adapter_dataset` | Build the configured cached-token dataset |
| `build_adapter_modules` | Build the projector/private dictionary and restore shared state |
| `configure_adapter_parameters` | Freeze shared parameters and select trainable projector/private weights |
| `build_adapter_token_transform` | Create the new source's token transform |

Resolve YAMLs with `gbsae.configuration.load_model_config` before calling these
components. `train_adapter.train_adapter(cfg, device=None, logger=None)` assembles
the run. For custom data/optimization, use the same components with
`experiments.unseen_model.training.train_weighted_adapter`; its token callback accepts and returns
`[B,T,D]` batches.
