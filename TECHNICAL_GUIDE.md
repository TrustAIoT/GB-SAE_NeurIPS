# Gated Bottleneck Sparse Autoencoders (GB-SAE)

GB-SAE learns shared and source-specific concepts from frozen model features.
Each source has an input projector and a private dictionary; sources share the
encoder, learned gate and shared dictionary.

## Install

Use Python 3.10 or newer with a compatible PyTorch/torchvision installation.
Run the following commands from this repository root:

```bash
python -m pip install -e .
# Optional Weights & Biases tracking:
python -m pip install -e ".[tracking]"
```

All maintained jobs take a single command-line argument, `--config`. Put paths,
model choices, hyperparameters, device and execution modes in that YAML.

Configuration-file references are relative to the YAML containing them. Image,
activation, checkpoint and output paths are relative to the working directory.
Run commands from the repository root.

## 1. Prepare images and frozen features

The supplied training and evaluation recipes use ImageNet. GB-SAE operates on
cached features and can be applied to other datasets by adapting the activation
dataset loader.

Prepare ImageNet with `meta.bin` and `train/<wnid>/*.JPEG` / `val/<wnid>/*.JPEG`.
GB-SAE loaders use images to enumerate samples and labels, and load features from
a separate activation root. A subset image tree must contain exactly the images
whose activation files are available.

```bash
python -m cache.run --config configs/cache/config.yaml
python -m cache.run --config configs/cache/combine.yaml
python -m cache.run --config configs/cache/validation.yaml
```

These jobs extract ViT/SigLIP/DinoV2 train features, combine the train caches, and
extract both separate and combined val caches. Cache parameters, ConvNeXt/CLIP
jobs, subset selection and other combinations are explained in the
[cache guide](cache/README.md). Caches retain native token grids and prefix tokens;
GB-SAE preprocessing is configured at training time.

## 2. Configure and train GB-SAE

Edit the image/cache/output paths and model settings in a training YAML:

| Recipe | Sources | Configured output patch grid |
| --- | --- | --- |
| [config_i.yaml](configs/train/config_i.yaml) | ViT, SigLIP, DinoV2 | 14 x 14 |
| [config_ii.yaml](configs/train/config_ii.yaml) | ViT, ConvNeXt, DinoV2 | 7 x 7 |
| [config_iii.yaml](configs/train/config_iii.yaml) | ViT, SigLIP, DinoV2, ConvNeXt | 7 x 7 |

```bash
python -m gbsae.train --config configs/train/config_i.yaml
```

For the cache commands above, set training `data.activation_root` to
`data/cache/config_i_merged/combined`. Set the evaluation job's activation root to
`data/cache/validation/combined`. Sample YAML paths are editable inputs; cache jobs
do not rewrite downstream YAMLs.

The training YAML is the model definition used by subsequent experiments.
It specifies ordered source names and dimensions, token layout/prefix/grid
policies, normalization, projectors, SAE architecture, losses and optimizer.
See [all training parameters](configs/README.md) and the
[training API and token requirements](gbsae/README.md).

When cross-source shared-code alignment is enabled, sources must produce the
same output patch grid and retained prefix count. `output_grid` is a square-grid
side length, not the number of tokens. Native input grids and feature widths may
differ. A single-source adapter or independently pooled image representation
does not require matching grids across sources.

Training writes checkpoints and `training_config.yaml` to `run.save_dir`.
Keep that configuration and its checkpoints together when running other jobs.

### Pretrained Config I checkpoint

The pretrained Config I checkpoint will be provided separately via Google Drive
(download link to be added). Save it anywhere and set the checkpoint path in your
experiment YAML. Use [configs/train/config_i.yaml](configs/train/config_i.yaml)
as the matching GB-SAE configuration.

## 3. Select an experiment

Each experiment YAML references the GB-SAE training YAML through `gbsae_config`.
Set its checkpoint, data split/roots, local training/evaluation parameters and
output paths, then run the corresponding module.

| Task | Guide | Example command/configuration |
| --- | --- | --- |
| Reconstruction and zero-shot retrieval | [main metrics](experiments/main_metrics/README.md) | `python -m experiments.main_metrics.batch_checkpoint_eval --config configs/eval/config_i.yaml` |
| Train per-source projected classifiers | [attribution](experiments/attribution/README.md) | `python -m experiments.attribution.train_probe --config configs/experiments/attribution/train_probe.yaml` |
| Evaluate concept sufficiency/comprehensiveness and prediction agreement, with a random orthogonal baseline | [attribution](experiments/attribution/README.md) | `python -m experiments.attribution.evaluate --config configs/experiments/attribution/evaluate.yaml` |
| Train/evaluate a fused shared classifier | [classification](experiments/downstream/downstream_classification/README.md) | `python -m experiments.downstream.downstream_classification.train_head --config configs/experiments/downstream_classification/train_head.yaml` |
| Train/evaluate retrieval heads | [learned retrieval](experiments/downstream/downstream_retrieval/README.md) | `python -m experiments.downstream.downstream_retrieval.run_experiment --config configs/experiments/downstream_retrieval/config.server.yaml` |
| Adapt one new feature source | [adapter](experiments/unseen_model/README.md) | `python -m experiments.unseen_model.train_adapter --config configs/unseen_model/clip_sh05.yaml` |
| Visualize concept heatmaps | [visualization](experiments/visualization/README.md) | `python -m experiments.visualization.concepts --config configs/experiments/visualization/concepts.yaml` |

For another ensemble, select its training YAML in the experiment configuration
and use a matching checkpoint. Each module guide documents its parameters,
evaluation definitions, outputs and Python APIs.

## APIs and configuration

- [GB-SAE components](gbsae/README.md): datasets, token transforms, projectors,
  SAE construction and a custom training loop.
- [Classification-head API](experiments/classification_head/README.md): custom
  datasets, frozen feature functions, classifiers, optimizers and epoch callbacks.
- [Cache APIs](cache/README.md): augmentation factories and backbone extractors.
- [Configuration guide](configs/README.md): ownership, path rules and parameters.

Datasets, weights and runtime artifacts belong in
configured data/output directories rather than the source package.

## Repository layout

```text
cache/          Backbone activation extraction, combination and inspection
gbsae/          Models, losses, datasets, token transforms and training
experiments/    Metrics, heads, adaptation and visualization
configs/        Training recipes and experiment/cache jobs
licenses/       Third-party license texts
```

## License

GB-SAE-specific code and additions are available under the
[PolyForm Noncommercial License 1.0.0](LICENSE).
Copyright (c) 2026 Jinchi Zhu and Tie Luo.

The license permits noncommercial uses, including research and evaluation,
and other uses expressly allowed by its terms. Commercial uses outside those
permissions require a separate written agreement; contact Jinchi Zhu and Tie Luo
for commercial licensing.

Third-party components, including code adapted from USAE and Overcomplete,
retain their original MIT licenses. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [licenses/](licenses/).
