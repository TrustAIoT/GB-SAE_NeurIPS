# Frozen-feature classification

Train one fused classifier on GB-SAE shared features, or evaluate independent
PROJECTED classifiers trained by the attribution module. Backbones, projectors
and SAE parameters are frozen; only the linear classification head is optimized.

## Train and evaluate

```bash
# One fused shared GB-SAE classifier:
python -m experiments.downstream.downstream_classification.train_head --config configs/experiments/downstream_classification/train_head.yaml
python -m experiments.downstream.downstream_classification.evaluate --config configs/experiments/downstream_classification/evaluate.yaml

# One PROJECTED classifier per source:
python -m experiments.attribution.train_probe --config configs/experiments/attribution/train_probe.yaml
python -m experiments.downstream.downstream_classification.evaluate --config configs/experiments/downstream_classification/evaluate_projected.yaml
```

Run from the repository root after editing paths. The fused feature for each
image is the mean across sources of their mean-pooled shared reconstruction
`z_sh @ D_sh`. PROJECTED uses each source's mean-pooled projected input. Mean
pooling is fixed for these classifiers. Sources can have different token counts
because pooling happens before fusion.

## Training parameters

| Field | Meaning |
| --- | --- |
| `gbsae_config` | Training YAML supplying source order/dimensions, encoder, dictionaries, projectors and preprocessing. |
| `inputs.ckpt` | Frozen GB-SAE/projector checkpoint. |
| `selection.sources` | Source subset for fusion; null uses all configured sources. |
| `data.imagenet_root`, `data.activation_root`, `data.split` | Optional head-training data overrides. |
| `loader.batch_size`, `loader.num_workers` | Loader settings. |
| `training.epochs`, `training.lr`, `training.weight_decay` | Linear-head AdamW training settings. |
| `training.num_classes` | Head output classes; null uses configured/checkpoint metadata, otherwise 1,000. Labels must fit this range. |
| `training.save_every` | Save every this many one-based epochs plus the final epoch. Must be positive. |
| `evaluation.std_mode` | `ckpt` uses saved source statistics; `recompute` estimates statistics on the selected data. |
| `evaluation.max_batches` | Optional batch cap per training epoch; null uses all batches. |
| `runtime.device` | Device for frozen feature extraction and head training. |
| `output.save_dir` | Root containing `shared_mean/` checkpoints and training CSV. |

Frozen feature extraction uses checkpoint encoder BatchNorm statistics. If standardization is enabled,
`std_mode: ckpt` requires saved mean/std for every selected source. Reuse the same
SAE/projectors, source order, pooling and statistics mode when evaluating a head.
The shared dataset applies normalization before configured grid interpolation.

## Evaluation parameters

| Field | Meaning |
| --- | --- |
| `method` | `gbsae` evaluates the fused `shared_mean` head; `projected` evaluates one projector head per selected source. |
| `gbsae_config`, `inputs.ckpt`, `selection.sources` | Frozen representation model and selected sources. |
| `inputs.probe_root` | Head-training output root. PROJECTED jobs point to the attribution probe root. |
| `evaluation.probe_epoch` | One-based saved head epoch to load; filename and metadata must match. |
| `evaluation.std_mode` | Statistics mode matching the selected head and its filename. |
| `data.imagenet_root`, `data.activation_root`, `data.split` | Evaluation image/cache selection. |
| `reference_data.imagenet_root`, `reference_data.activation_root`, `reference_data.split` | Head-training dataset for WNID-to-label mapping; defaults to the GB-SAE training data. |
| `loader.batch_size`, `loader.num_workers` | Evaluation loader settings. |
| `evaluation.max_batches` | Optional evaluation cap; null processes the complete selected dataset. |
| `runtime.device` | Execution device. |
| `output.output_csv` | Evaluation CSV destination. |

`evaluate_projected.yaml` inherits shared evaluation options with
`experiment_config` and changes the method/probe root. Configuration references
resolve beside the containing YAML; asset paths use the working directory.
Evaluation maps class labels by WNID and skips classes absent from head training.
Acc@1/Acc@5 use all remaining samples, with Top-5 clipped to the class count.

## Outputs and extension points

The fused checkpoint path is
`<save_dir>/shared_mean/probe_shared_mean_pool-mean_std-<mode>_epoch-<epoch>.pth`.
PROJECTED paths are listed in the [probe guide](../../attribution/README.md).
Head checkpoints contain `probe_state_dict`, epoch, representation metadata,
class count and source/fusion information. Training CSV accuracies are
percentages; evaluation CSV accuracies are fractions.

`train_head.extract_features` defines shared reconstruction and fusion.
`train_fused_head` assembles the training workflow; `evaluate.evaluate_heads`
loads matching classifiers and computes metrics. For custom data, feature
functions, heads, optimizers or callbacks, use the
[classification-head API](../../classification_head/README.md).
