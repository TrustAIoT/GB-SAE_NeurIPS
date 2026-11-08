# Backbone activation caching

Extract frozen backbone features upstream of GB-SAE training. This package owns
its data selection, augmentation, backbone extraction, storage, combination and
inspection settings. It reads an independent YAML, not a GB-SAE training recipe.
Backbone implementations live in `gbsae/backbones/`.

## Run the supplied jobs

All commands run from the repository root:

```bash
# Base source train cache, train combination, and full val cache:
python -m cache.run --config configs/cache/config.yaml
python -m cache.run --config configs/cache/combine.yaml
python -m cache.run --config configs/cache/validation.yaml

# Additional source caches; run the sources/splits you need:
python -m cache.run --config configs/cache/convnext_train.yaml
python -m cache.run --config configs/cache/convnext_val.yaml
python -m cache.run --config configs/cache/clip_train.yaml
python -m cache.run --config configs/cache/clip_val.yaml

# Combine the prepared base and ConvNeXt caches:
python -m cache.run --config configs/cache/combine_configii_train.yaml
python -m cache.run --config configs/cache/combine_configii_val.yaml
python -m cache.run --config configs/cache/combine_configiii_train.yaml
python -m cache.run --config configs/cache/combine_configiii_val.yaml

python -m cache.run --config configs/cache/inspect.yaml
```

| YAML | Image selection / sources | Default storage |
| --- | --- | --- |
| `config.yaml` | Full train; ViT, SigLIP, DinoV2 | `data/cache/config_i/per_source` |
| `validation.yaml` | Full val; ViT, SigLIP, DinoV2 | `data/cache/validation/{per_source,combined}` |
| `convnext_train.yaml` | Approximately 10% train; ConvNeXt | `data/cache/convnext_train/per_source` |
| `convnext_val.yaml` | Approximately 10% val; ConvNeXt | `data/cache/convnext_val/per_source` |
| `clip_train.yaml` | Full train; CLIP | `data/cache/clip_train/per_source` |
| `clip_val.yaml` | Full val; CLIP | `data/cache/clip_val/per_source` |
| `combine.yaml` | Full train; ViT, SigLIP, DinoV2 | `data/cache/config_i_merged/combined` |
| `combine_configii_train.yaml` / `combine_configii_val.yaml` | ConvNeXt 10% manifest; ViT, ConvNeXt, DinoV2 | `data/cache/config_ii_{train,val}/combined` |
| `combine_configiii_train.yaml` / `combine_configiii_val.yaml` | ConvNeXt 10% manifest; ViT, SigLIP, DinoV2, ConvNeXt | `data/cache/config_iii_{train,val}/combined` |
| `inspect.yaml` | Checks `combine.yaml` outputs by default | `data/cache/config_i_merged/inspection.json` |

All train extraction jobs enable augmentation; val jobs disable it. ConvNeXt
jobs export selected originals to `data/imagenet_10percent/{train,val}`. Config
II/III use the same ConvNeXt selection per split and retrieve the matching base
arrays from the full caches. CLIP is optional and is not needed for those combinations.

`config.yaml` saves separate features, so run `combine.yaml` when combined train
features are needed. `validation.yaml` uses `both`; its combined val cache already
exists while its separate files remain available for other combinations.

## Extraction parameters

`action: extract` selects extraction. All paths use the working directory.

| Field | Meaning |
| --- | --- |
| `models` | Ordered mapping from source identifiers to extractor specifications. Identifiers become NPZ keys/filename suffixes; their count/order controls view assignment. |
| `models.<source>.type` | Built-in `ViT`, `SigLIP`, `DinoV2`, `ConvNeXt`, `ResNet`, `CLIP`, or a registered name / `module:factory`. |
| `models.<source>.kwargs` | Constructor options passed to that extractor factory. Built-in wrappers accept `use_half`. |
| `models.<source>.kwargs.use_half` | `false` keeps default FP32; `true` casts model/input to FP16. This changes extraction precision and generally the saved feature dtype. |
| `models.<source>.storage_layout` | `tokens` (default `[T,D]`), `channels_first` (`[D,H,W]`) or `channels_last` (`[H,W,D]`). |
| `models.<source>.feature_dim` | Optional assertion of raw feature width D. |
| `data.type` | `imagenet` for torchvision ImageNet metadata, or `image_folder` for `root/split/class/images`. |
| `data.root` | Original image root shared by all sources. |
| `data.split` | Physical split directory name. |
| `data.export_root` | Optional destination for selected originals and ImageNet `meta.bin`; null copies no images. |
| `sampling.mode` | `all`, `balanced` or `manifest`, described below. |
| `sampling.fraction` | Fraction in `(0,1]` for `balanced` only. Ignored by `all`/`manifest`. |
| `sampling.manifest` | Input CSV when mode is `manifest`. |
| `sampling.exclude_manifest` | Optional same-split CSV of images to remove from the selection. |
| `augmentation.enabled` | Enable augmented views; false gives every source the original image. |
| `augmentation.type` | Built-in `slight`, a registered augmentation, or `module:factory` returning a PIL-to-PIL transform. |
| `augmentation.kwargs` | Arguments for the augmentation factory. Built-in `image_size` sets its crop size; backbone preprocessing follows it. |
| `loader.batch_size`, `loader.num_workers` | Images per extraction batch and decoding workers. |
| `runtime.seed` | Cache-stage seed for dataset sampling, augmentations and view assignment. |
| `runtime.device` | `auto`, `cpu` or an explicit CUDA device. |
| `output.root` | Root containing metadata, manifests and activation subdirectories. Use a distinct root for each train/val extraction job. |
| `output.mode` | `per_source` (default), `both` or `combined`. |
| `output.compressed` | Lossless NPZ compression; changes storage/I/O cost, not values or dtype. |
| `output.overwrite` | False validates/skips compatible existing artifacts and fills gaps; true regenerates and replaces artifacts touched by this job. It does not clear the directory. |

### Image selection and views

- `all` uses the complete split.
- `balanced` uses seeded near-class-balanced allocation. Classes with too few
  images are capped and their shortfall is not redistributed, so the actual
  fraction may be smaller. On imbalanced data, `fraction: 1` need not equal `all`.
- `manifest` reuses an existing CSV in its recorded order, checking image paths,
  class labels and split. Use this when separately cached sources must cover
  exactly the same subset. CSVs with a `target` column are accepted as well.

One original-image manifest is shared by every source in an extraction job.
With augmentation enabled, N >= 2 sources receive the original plus N-1 augmented
views with deterministic per-image random assignment. A single source receives
one augmented view. With augmentation disabled, every source receives the original.

Per-image randomness uses the filename/basename and base seed. Equal basenames
share seeds even across class directories. Batch size, decoding-worker count
and resuming missing files do not change assigned views. Keep the same source
order and augmentation settings when resuming.

### Storage and layout requirements

| Mode | Files written |
| --- | --- |
| `per_source` | Separate source NPZs only |
| `both` | Finish all separate caches, then create combined files |
| `combined` | Combined files only, without individual NPZ intermediates |

Default features retain prefixes and native grids as `[T,D]`. Cache extraction
performs backbone image preprocessing and feature extraction, but no GB-SAE
standardization, per-token L2 division or token-grid interpolation.

Square spatial layouts require prefix-free features. Features with CLS/register
tokens must use `tokens`. Rectangular maps are not supported. A combined NPZ may
contain different layouts, widths and native grid sizes across sources.

```text
<output.root>/
  cache_metadata.json
  manifests/train.csv
  per_source/train/<class>/<stem>ViT.npz
  per_source/train/<class>/<stem>ViT.npz.json
  combined/train/<class>/<stem>_combined.npz
  combined/train/<class>/<stem>_combined.npz.json
```

Per-source NPZs contain `activation`, `label` and `view_id`. Combined NPZs contain
only source feature arrays. Per-image JSON sidecars record identity/label, source
views, layout, prefix counts, native grid, shape, dtype and extraction identity.
Run metadata records extraction settings, backbone weights/feature definition,
preprocessing and selected-input fingerprints. The manifest is the readable image list.

GB-SAE uses the original image tree for sample labels and directly opens the NPZ
paths; JSON files are not loaded as samples. Set downstream `activation_root` to
`<output.root>/per_source` with `combined_npz: false`, or `.../combined` with true.
Explicitly set downstream `activation_layouts`, prefix counts and token grids;
DataLoaders do not infer these settings from sidecars. See [training parameters](../configs/README.md).

## Combine selected sources

`action: combine` reads existing features without loading backbones. Use
`models` to choose sources and `inputs.source_roots` to locate them. For example,
to add CLIP to full ViT/SigLIP/DinoV2 features, give each source its own root and
select a manifest covered by every source. A 10% source can be combined with
full-source caches by choosing the 10% source's manifest.

| Combination / inspection field | Meaning |
| --- | --- |
| `action` | `combine` writes combined NPZs; `inspect` checks existing caches. |
| `models` | Sources required for each selected image. `type` describes each source; no extractor is loaded for these actions. |
| `models.<source>.prefix_tokens`, `storage_layout`, `feature_dim` | Explicit format declarations for imported NPZs without metadata. New cache sidecars already describe their arrays. |
| `data.split` | Split shared by the selected manifest and input paths. |
| `inputs.manifest` | CSV defining selected images/order. Null discovers the union of input identities and reports incomplete images. |
| `inputs.source_roots` | Exact per-source roots containing `<split>/<class>/*.npz`. |
| `inputs.combined_root` | Optional existing combined directory when adding sources; use a different output root. For inspection, points to the combined cache being checked. |
| `output.root` | Combination run root, or directory for the inspection report. |
| `output.mode`, `output.compressed`, `output.overwrite` | Combination uses `combined`; compression and replacement have the extraction meanings above. |

The combiner checks image identity, labels, source completeness and array formats.
Missing/mismatched inputs are reported without writing partial combined images.
It does not change values, dtype, views or token grids. Its report is
`combination_report.json`; errors produce a nonzero CLI exit status.

To extend an existing combined cache, set `inputs.combined_root`, list only added
sources in `source_roots` and choose a new `output.root`. Existing arrays are
preserved, including extra sources. Do not supply a source in both inputs.
Imported files without metadata need explicit prefix/layout declarations and a
manifest when image identity cannot be recovered from standard filenames.

`action: inspect` writes `inspection.json`, listing valid images and issues.
It checks sidecars and arrays; use `source_roots` for separate caches, or
`combined_root` for combined files. Required source metadata must be present.

## Resume and extend

Writes are atomic. With `overwrite: false`, extraction validates the run recipe
and artifacts before skipping inference; missing, corrupted or uncommitted files
are regenerated. Changing extraction-relevant settings/images requires a new
output root or explicit overwrite. Existing separate caches can recreate missing
combined outputs in `both` mode without another backbone forward pass.

```python
from cache.configuration import load_config
from cache.pipeline import extract_cache
from cache.combine import combine_cache
from cache.validation import inspect_cache

cfg = load_config("configs/cache/config.yaml")
# report = extract_cache(cfg)
# Use an action-appropriate config for combine_cache / inspect_cache.
```

| Module | Responsibility |
| --- | --- |
| `augmentation.py` | Augmentation factories and view generation |
| `utils.py` | Global/per-image seeds, view assignment and factory imports |
| `data.py`, `manifest.py` | Shared image selection, manifest I/O and optional exports |
| `features.py` | Backbone registry, extraction and layout conversion |
| `pipeline.py` | Batched extraction and storage-mode orchestration |
| `storage.py` | Artifact paths, metadata, atomic writes and reuse checks |
| `combine.py`, `validation.py` | Combination and inspection APIs |

An augmentation factory returns a PIL-to-PIL callable. A ready-made transform can
be selected with `augmentation.type: torchvision.transforms:RandomHorizontalFlip`
and `kwargs: {p: 0.5}`. For compositions, provide your own `module:factory` or
register it with `register_augmentation(name, factory)`. Keep stochastic image
operations in augmentation and deterministic preprocessing in the extractor.

A custom backbone factory has signature `factory(device=..., **kwargs)` and
returns `FeatureExtractor(model, preprocess, forward, native_layout, prefix_tokens,
provenance)`. `preprocess` maps PIL to a tensor; `forward` returns a selected
feature tensor `[B,T,D]`, `[B,D,H,W]` or `[B,H,W,D]`. Declare its native layout,
nonnegative prefix count and weight/feature/preprocessing metadata. Select the
factory with `models.<source>.type: my_module:factory`, or use
`register_backbone(name, factory)`. Extend native feature extraction inside that
adapter; the pipeline does not guess layers from dictionary outputs.
