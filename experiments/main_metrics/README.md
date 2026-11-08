# Reconstruction and zero-shot retrieval

Evaluate saved GB-SAE checkpoints using reconstruction R-squared and cross-source
image retrieval. These evaluators use frozen representations without training a
classification or retrieval head.

## Run

Edit the selected job's `gbsae_config`, checkpoint selection, image/cache paths
and output directory, then run from the repository root:

```bash
# Both metrics; the ensemble is selected by the referenced training YAML.
python -m experiments.main_metrics.batch_checkpoint_eval --config configs/eval/config_i.yaml
python -m experiments.main_metrics.batch_checkpoint_eval --config configs/eval/config_ii.yaml
python -m experiments.main_metrics.batch_checkpoint_eval --config configs/eval/config_iii.yaml

# Individual tasks:
python -m experiments.main_metrics.reconstruction.evaluate --config configs/experiments/main_metrics/reconstruction.yaml
python -m experiments.main_metrics.retrieval.evaluate --config configs/experiments/main_metrics/retrieval.yaml
```

All three entry points share the same evaluator functions and YAML schema.
The supplied standalone jobs select their corresponding metric in
`evaluation.metrics`. To evaluate another trained ensemble, change the training
reference and checkpoint/data paths, rather than copying model fields.

## Parameters

| Field | Meaning |
| --- | --- |
| `gbsae_config` | Training YAML defining architecture, source order/dimensions, projectors, normalization and token policies. |
| `inputs.ckpt` | List of explicit SAE checkpoint paths. |
| `inputs.ckpt_glob` | List of checkpoint glob patterns. |
| `inputs.ckpt_dir` | Recursively search this directory for `.pth`, `.pt` and `.ckpt` files. If all three checkpoint selections are empty, search the training `run.save_dir`. |
| `inputs.projector_from` | Optional fallback projector key in the same checkpoint if the requested source projector is absent. Its dimensions must fit that source. |
| `selection.sources` | Reconstruction source subset; `null` means all. Retrieval always uses all configured sources and needs at least two. |
| `data.imagenet_root`, `data.activation_root`, `data.split` | Evaluation data selection; omitted values inherit the training recipe. Set val roots explicitly when they differ from train roots. |
| `loader.batch_size`, `loader.num_workers` | Common image batch size and workers; null values use training loader settings. |
| `retrieval_data.imagenet_root`, `retrieval_data.activation_root`, `retrieval_data.split` | Optional retrieval-only data overrides; null values reuse `data`. |
| `retrieval_loader.batch_size`, `retrieval_loader.num_workers` | Optional retrieval-only loader overrides. |
| `evaluation.metrics` | A list containing `reconstruction`, `retrieval`, or both. |
| `evaluation.pool` | Retrieval token pooling: `mean`, `max` or `sum`. |
| `evaluation.shared_repr` | Shared retrieval representation: decoded `xhat_sh` or sparse codes `z_sh`. |
| `evaluation.private_repr` | Private retrieval representation: decoded `xhat_pr` or sparse codes `z_pr`. |
| `evaluation.max_images` | Retrieval image limit; `null` evaluates the complete selected dataset. This does not cap reconstruction. |
| `runtime.device` | Execution device, e.g. `auto`, `cpu` or `cuda:0`. |
| `output.output_dir` | Directory for cumulative metric CSVs. |

Model/preprocessing fields are explained in [the configuration guide](../../configs/README.md).
Image/cache identity, feature widths, prefix counts and native input grids must
match the referenced recipe. Image-level retrieval pools each source separately,
so native or output token counts may differ across sources.

## BatchNorm statistics and evaluation objectives

Reconstruction evaluates how the learned encoder and shared/private dictionaries
preserve each backbone's representations. Backbones have distinct activation
distributions, and the shared encoder processes each source separately. We use
current-batch BatchNorm statistics to retain this source-wise normalization when
measuring reconstruction fidelity. Learned affine parameters, model weights and
BatchNorm running buffers remain fixed throughout evaluation.

Retrieval evaluates whether the learned shared space supports matching
independently extracted representations. A gallery may be indexed in advance,
while queries arrive separately; the representation of an image must therefore
remain consistent across extraction batches. Checkpoint BatchNorm running
statistics provide a fixed mapping for both query and gallery features.

These BatchNorm statistics are separate from the dataset-computed statistics used
to standardize input features.

## Reconstruction values

For each source, evaluate full, shared-only, private-only and private-versus-residual
reconstruction. The residual is input minus shared reconstruction.

- `r2_linear_*` compares representations in the projected feature space.
- `r2_raw_*` maps reconstructions back with the projector pseudoinverse and compares
  them with the loader-preprocessed features. It does not undo standardization.

Each value is the arithmetic mean of per-batch R-squared scores. Each batch uses
one global feature mean in its denominator. Reconstruction uses shuffled,
source-specific loaders and current-batch encoder BatchNorm statistics.
Input standardization statistics are estimated by the dataset on original-grid
tokens; reconstruction passes build their datasets independently.

## Retrieval values

PROJECTED pools projected input tokens. SHARED/PRIVATE pool the selected decoded
representations or codes. Features are compared with cosine similarity across
all ordered source pairs, and directional recalls are averaged.

Instance Recall checks the paired image. Category Recall checks matching labels
and includes the paired image in the gallery. Retrieval uses checkpoint
BatchNorm running statistics; input standardization uses
dataset-computed statistics. Supported Recall@1/5/10/50 ranks
are reported when the gallery is large enough. A strict greater-than rank
comparison determines how instance ties are counted.

Learned-head retrieval is a separate workflow with contrastive training and
paired-image exclusion for category metrics; see [its guide](../downstream/downstream_retrieval/README.md).

## Outputs and Python APIs

`output.output_dir` receives `reconstruction_metrics.csv` and/or
`retrieval_metrics.csv`, updated after each checkpoint. Reconstruction rows identify
the checkpoint and source. Retrieval rows identify checkpoint, task, branch and
rank. Its `accuracy` column stores Recall@k as a fraction; `random_baseline` and
`shared_vs_random_ratio` provide chance-reference columns.

- `reconstruction.evaluator.evaluate_reconstruction` returns per-source projected
  and input-space metrics.
- `retrieval.evaluator.evaluate_retrieval` returns the three representation branches.
- `batch_checkpoint_eval.evaluate_checkpoints` accepts a resolved model config,
  checkpoint list, task selection and output directory.

Use [gbsae.train.load_config](../../gbsae/train.py) to resolve a training YAML
before calling Python APIs. Change representation extraction or metric aggregation
inside its respective evaluator when defining a new research metric.
