# Configuration and parameter guide

All maintained commands use `python -m MODULE --config FILE.yaml`. Edit the YAML
to select data, checkpoints, model settings, hyperparameters, device and outputs.
Use actual numeric/boolean YAML values, not quoted strings such as `"32"` or
`"false"`. Field comments describe the supplied values; the tables below explain
how to change them.

## Which YAML owns which settings?

| Directory | Use |
| --- | --- |
| [train/](train/) | GB-SAE model, preprocessing, losses and training settings |
| [ablations/](ablations/) | GB-SAE training recipes with selected loss weights changed |
| [eval/](eval/) | Ready-to-run reconstruction/retrieval jobs for each ensemble |
| [experiments/](experiments/) | Head training, attribution, metrics and visualization jobs |
| [unseen_model/](unseen_model/) | A new source's adapter, referring to a base training recipe |
| [cache/](cache/) | Independent image/backbone cache extraction, combination and inspection |

A GB-SAE training YAML is the authoritative definition of its sources,
architecture and preprocessing. Other experiments read it with `gbsae_config`
and add their own checkpoint/data selection and experiment settings:

```yaml
gbsae_config: ../../train/config_i.yaml
inputs:
  ckpt: artifacts/config_i/weights/gbsae_config_i_epoch_29.pth
data:
  split: val
  activation_root: data/cache/validation/combined
```

`experiment_config` reuses another experiment YAML; local settings override
inherited settings. It is useful for evaluation-only jobs or method variants.
Configuration references resolve relative to the containing YAML. Dataset,
checkpoint and output paths resolve from the command's working directory.
Run project commands from the repository root.

An experiment may select different image/cache roots, split, target class and
NPZ storage format. Model widths, prefix policy, grid transformations and
normalization are read from the training recipe. `reference_data` identifies the
classifier's training dataset for WNID-to-label mapping when it differs from the
GB-SAE recipe. The cache pipeline owns a separate YAML and does not use
`gbsae_config`.

## GB-SAE training parameters

Start with [train/config_i.yaml](train/config_i.yaml), or another ensemble recipe.
The following fields are consumed by `gbsae.train`.

### Sources and cached data

| Field | Meaning and settings |
| --- | --- |
| `models.sources` | Ordered, unique source identifiers. They must match cache keys/filename suffixes and projector/private-dictionary names. Names do not select backbone types. |
| `models.input_dims` | Raw feature width D for every source, before projection. Must match each cached array. |
| `data.imagenet_root` | ImageNet root with metadata and the selected split; supplies image identities and class labels. |
| `data.activation_root` | Cache root directly containing `train/` or `val/`; use a cache job's `per_source/` or `combined/` directory. |
| `data.split` | Physical split directory to load, normally `train` or `val`. |
| `data.target_class` | `null`/`ALL` for all classes; otherwise a supported ImageNet class key or integer label. |
| `data.combined_npz` | `true`: `<stem>_combined.npz`, arrays keyed by source. `false`: `<stem><Source>.npz`, array named `activation`. |
| `data.activation_layouts` | Per-source layout: `tokens` = `[T,D]`, `channels_first` = `[D,H,W]`, `channels_last` = `[H,W,D]`. Spatial layouts are flattened. |
| `data.prefix_tokens` | Explicit number of cached leading non-patch tokens for each source, including CLS/register tokens. Use 0 for patch-only features. |
| `data.use_class_tokens` | `false` removes prefixes, `true` retains them; also accepts a source-to-bool mapping. Applies to all configured prefixes. |
| `data.token_alignment` | Entry per source: `{input_grid: S, output_grid: G}`, or `null` to skip grid resizing. Grid values are square side lengths, excluding prefixes. |
| `data.standardize` | Estimate per-channel mean/std on up to 1,000 sampled images and standardize original-grid tokens. |
| `data.divide_norm` | Per-token L2 division; applied only when `standardize` is false. |

The image tree must enumerate the available caches. A 10% cache therefore needs
the corresponding 10% image tree, not the full image directory. JSON sidecars
are not samples and are not read by this dataset.

**Grid requirements:** `input_grid ** 2` must equal the cached patch count after
separating prefixes. Equal input/output grids check shapes without resizing.
When `loss.lambda_align > 0` with multiple sources, output patch grids, retained
prefix counts and batch correspondence must match across sources. Input grids
and feature widths may differ. For example, a 16 x 16 DINO grid and a 14 x 14 ViT
grid can both use `output_grid: 14`. Setting different target grids is not valid
for token-wise shared alignment. `null` is suitable when resizing is unnecessary;
the participating sequences must still correspond. See [token examples](../gbsae/README.md).

### Projector and SAE

| Field | Meaning and settings |
| --- | --- |
| `projector.type` | `frozen_random`, `learnable` or `identity`. Identity requires every input width to equal `output_dim`. |
| `projector.output_dim` | Common feature width after projection, used as the SAE input width. |
| `projector.kwargs.mode` | For `frozen_random`: `orthogonal` or `gaussian` weight initialization. |
| `projector.kwargs.scale` | For `frozen_random`: divide initialized weights by the square root of the input width. |
| `projector.kwargs.init` | For `learnable`: `orthogonal` or `xavier` initialization. Supply only kwargs accepted by the chosen type. |
| `sae.n_components` | Number of latent concepts; shared and each private dictionary have this many atoms. |
| `sae.k` | Total per-token Top-K budget before gating. Use `0 < k < n_components` for sparse operation; `k <= 0` or `k >= n_components` leaves codes dense. |
| `sae.encoder_module` | Registered encoder name, e.g. `linear`, `mlp_bn_1` or `mlp_ln_1`; it must accept flattened `[N,output_dim]` inputs. |
| `sae.dictionary_norm` | Atom normalization, commonly `l2`, `l1` or `identity`; used by shared and private dictionaries. |

The main optimizer updates SAE parameters only. Selecting a `learnable` projector
class does not add its weights to that optimizer. For joint projector training,
assemble the components with your own optimizer; the single-source adapter
already includes its target projector in its optimizer.

### Losses and gate schedule

| Field | Meaning |
| --- | --- |
| `loss.recon_loss` | Reconstruction error type, `l1` or `l2`. |
| `loss.lambda_sh_only` | Weight of shared-only reconstruction in addition to full shared+private reconstruction. |
| `loss.lambda_shpr` | Shared/private dictionary orthogonality weight; 0 disables it. |
| `loss.lambda_align` | Cross-source shared-code alignment weight; 0 disables cross-source alignment. |
| `loss.lambda_gate_bin` | Gate binarization penalty weight. |
| `loss.lambda_gate_ratio` | Penalty encouraging the mean gate to match `gate_ratio_target`. |
| `loss.gate_ratio_target` | Desired mean shared gate value, in `[0,1]`, used by the ratio penalty. |
| `schedule.warmup_pct` | Fraction of total training steps before gate penalties start their linear ramp. |
| `schedule.steady_pct` | Fraction when gate penalties reach full weight. Require `0 <= warmup_pct < steady_pct <= 1`. |

Gate scheduling scales the gate penalties, not the optimizer learning rate.
Auxiliary reconstruction is handled by the loss implementation.

### Optimization, execution and outputs

| Field | Meaning |
| --- | --- |
| `optimizer.lr` | AdamW base learning rate. |
| `optimizer.final_lr` | Final cosine-schedule learning rate. |
| `optimizer.weight_decay` | AdamW weight decay. |
| `optimizer.warmup_pct` | Fraction of optimizer steps used for learning-rate warmup; independent of the gate schedule. |
| `optimizer.warmup_lr` | Learning rate at the start of optimizer warmup. |
| `training.batch_size` | Images per batch; each image contributes its retained/aligned tokens. |
| `training.num_workers` | DataLoader workers; 0 runs loading in the main process. |
| `training.epochs` | Number of training epochs. |
| `training.enc_clip` | Encoder/shared-decoder gradient norm limit; the private-decoder limit is scaled by source count. `null` disables clipping; use a positive limit otherwise (zero would zero the clipped gradients). |
| `training.save_every` | Save at zero-based epochs divisible by this value, plus the final epoch. Must be positive. |
| `training.seeded` | Seed Python/PyTorch to 10 inside the training loop. To control initialization and NumPy statistics too, seed before constructing data/models in a custom Python launcher. |
| `runtime.device` | `auto`, `cpu`, `cuda` or a CUDA device such as `cuda:0`. |
| `run.name` | Run label and checkpoint filename prefix. |
| `run.save_dir` | Checkpoint directory; also receives `training_config.yaml`. |
| `run.logger_type` | Use `full` for the supplied trainer. A custom Python caller may supply a compatible logger. |
| `run.logger_granularity` | 1 logs per-source metrics; 2 logs source means. |
| `run.use_wandb` | Enable Weights & Biases logging; install the `tracking` extra first. |
| `run.wandb_project` | W&B project name. |
| `run.wandb_online` | `true` uploads logs; `false` uses offline logging. |
| `run.wandb_save_dir` | Local W&B artifact directory. |

Checkpoints contain SAE/projector weights, optimizer state, zero-based epoch,
learning rate and standardization statistics when enabled. Construct evaluation
models with the same encoder, dimensions, component count, `k`, normalization and
source keys as training.

## Loss-weight recipes

Loss-weight studies use the same training entry and parameter meanings:

```bash
python -m gbsae.train --config configs/ablations/alignment_high.yaml
```

| Recipe | `lambda_sh_only` | `lambda_align` | `lambda_gate_bin` |
| --- | --- | --- | --- |
| `alignment_high.yaml` | 0.5 | 1.3 | 0.2 |
| `alignment_mid.yaml` | 0.5 | 1.0 | 0.2 |
| `equal_weights.yaml` | 1.0 | 1.0 | 0.2 |
| `shared_only_high.yaml` | 1.0 | 0.5 | 0.2 |
| `gate_high.yaml` | 0.5 | 1.3 | 0.5 |

Each is a complete training recipe. Give runs distinct names/output directories
when changing weights, and point subsequent experiments at the matching recipe.

## Experiment-specific parameters

These guides explain every field in their supplied jobs, including required
checkpoints, data selection, statistics modes, outputs and Python extension points:

| Workflow | Parameter guide |
| --- | --- |
| Reconstruction / zero-shot retrieval | [main metrics](../experiments/main_metrics/README.md) |
| Projected classifiers / concept attribution | [attribution](../experiments/attribution/README.md) |
| Fused classification | [downstream classification](../experiments/downstream/downstream_classification/README.md) |
| Learned retrieval | [downstream retrieval](../experiments/downstream/downstream_retrieval/README.md) |
| New-source adaptation | [unseen model](../experiments/unseen_model/README.md) |
| Concept visualization | [visualization](../experiments/visualization/README.md) |
| Backbone caches | [cache](../cache/README.md) |

`selection.sources: null` uses all sources where that field is supported.
`evaluation.max_images: null` or `max_batches: null` processes the complete
selected dataset. Sample counts for estimating statistics and top-image counts
for plotting are separate parameters. Keep classifier/retrieval-head optimizers
in their own job YAMLs; they do not change GB-SAE training settings.
