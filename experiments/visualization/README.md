# Concept heatmaps

Rank shared/private concepts by response energy, retrieve highly activating
images and overlay their spatial activations on the original images.

```bash
python -m experiments.visualization.concepts --config configs/experiments/visualization/concepts.yaml
```

Edit the training reference, SAE checkpoint, image/cache roots and output path.
Run from the repository root. The script reads cached features; original image
files are needed to render overlays.

## Parameters

| Field | Meaning |
| --- | --- |
| `gbsae_config` | Training YAML defining model/source order and token preprocessing. |
| `inputs.sae_ckpt` | SAE/projector weights and saved normalization statistics. |
| `data.imagenet_root`, `data.activation_root`, `data.split` | Images and corresponding cached features to rank. Unset paths inherit the training recipe. |
| `loader.batch_size`, `loader.num_workers` | Ranking loader batch size and workers. |
| `evaluation.concept_type` | `shared`, `private` or `all`. |
| `evaluation.gate_shared_thresh` | Select shared candidates whose gate is strictly above this value. |
| `evaluation.gate_private_thresh` | Select private candidates whose gate is strictly below this value. |
| `evaluation.top_shared_concepts` | Maximum number of shared candidates retained by energy ranking. |
| `evaluation.top_private_concepts` | Maximum private candidates retained per source. |
| `evaluation.topN` | Highest-response images saved for each selected concept. |
| `runtime.device` | Execution device. |
| `output.save_dir` | Root for shared/private concept directories and overlays. |

## Input requirements and processing

Cached shapes, layouts, raw widths and prefix counts must match the training
recipe. When standardization is enabled, the checkpoint must contain statistics
for all configured sources. Normalization precedes configured token interpolation.
Prefixes may participate in SAE inference when configured, but are excluded from
spatial patch maps.

Shared visualization averages corresponding patch codes across sources, so their
output patch grids must match. Retained prefix counts may differ because prefixes
are excluded before that average. Private-only visualization works with each
source's own square patch grid. Feature dimensions may differ before projection.
Concept selection, image ranking and heatmap rendering use checkpoint BatchNorm
running statistics, keeping each image's latent responses consistent across passes.

Energy ranking uses the highest-response patch tokens. The workflow ranks
candidates, selects top images and then renders only those images. If no concepts
pass a branch's gate threshold, that branch has no concepts to render; adjust the
thresholds or requested branch for the model being inspected.

## Outputs and customization

```text
<save_dir>/
  shared/concept_<id>/summary.txt
  shared/concept_<id>/rank1_overlay.png
  private/<source>/concept_<id>/summary.txt
  private/<source>/concept_<id>/rank1_overlay.png
```

Summaries list ranked image paths and scores. Modify `concept_energy`, concept
selection or `save_overlay` in `concepts.py` to change scoring and presentation.
`gbsae.data.preprocessing.select_patch_tokens` and configured token transforms
can be reused for custom spatial analyses. Heatmap resizing is a plotting step;
it does not change the backbone activation cache.
