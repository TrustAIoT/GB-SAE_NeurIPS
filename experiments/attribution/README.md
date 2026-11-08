# Projected classifiers and concept attribution

Train one linear classifier per source on mean-pooled projected backbone features.
Use those frozen classifiers as probes to measure the effect of keeping/removing
selected shared, private or total concept contributions, with a uniform random
orthogonal baseline. Evaluation writes numerical results only.

## 1. Train the probes

```bash
python -m experiments.attribution.train_probe --config configs/experiments/attribution/train_probe.yaml
```

Edit the GB-SAE training reference, SAE checkpoint and training image/cache paths
first. The checkpoint supplies frozen source projectors; this step trains only
the classification heads. `selection.sources: null` creates one head for every
source in the training recipe. Pooling is always the token mean.

| Training field | Meaning |
| --- | --- |
| `gbsae_config` | Authoritative source, projector, architecture and preprocessing recipe. |
| `inputs.ckpt` | GB-SAE checkpoint containing the frozen projectors and optional normalization statistics. |
| `selection.sources` | Nonempty unique source subset, or `null` for all configured sources. |
| `data.imagenet_root`, `data.activation_root`, `data.split` | Optional head-training data selection; otherwise inherit the recipe. |
| `loader.batch_size`, `loader.num_workers` | Images per batch and loading workers. |
| `training.epochs`, `training.lr`, `training.weight_decay` | Linear-head AdamW training settings. |
| `training.num_classes` | Output class count; null uses configured/checkpoint class metadata, falling back to 1,000. Targets must fit this range. |
| `training.save_every` | Save every this many one-based epochs and at the final epoch; must be positive. |
| `evaluation.std_mode` | `ckpt` uses checkpoint mean/std, which must exist for selected sources; `recompute` estimates them from the selected data. Used only when standardization is enabled. |
| `evaluation.max_batches` | Optional per-epoch batch cap; null uses the complete loader. |
| `runtime.device` | Execution device. |
| `output.save_dir` | Root for source-specific probe checkpoints and training CSVs. |

Heads are saved as:

```text
<save_dir>/<source>/projector/
  linear_probe_metrics.csv
  probe_<source>_projector_pool-mean_std-<mode>_epoch-<epoch>.pth
```

Training CSV loss and accuracy are sample-weighted; Acc@1/Acc@5 are percentages.
These same probes can be evaluated for PROJECTED classification with the
[downstream classifier evaluator](../downstream/downstream_classification/README.md).

## 2. Evaluate concept interventions

```bash
python -m experiments.attribution.evaluate --config configs/experiments/attribution/evaluate.yaml
```

Use the same SAE/projector weights and preprocessing as probe training. Select
the held-out data and the probe directory. If probe-training roots differ from
the GB-SAE recipe, provide `reference_data` for the correct WNID label mapping.

| Evaluation field | Meaning |
| --- | --- |
| `gbsae_config`, `inputs.ckpt`, `selection.sources` | Model definition, SAE/projector checkpoint and sources to evaluate. |
| `inputs.probe_root` | Directory recursively searched for the trained per-source probes. |
| `evaluation.probe_match` | Optional list of substrings that must all occur in a probe path, e.g. `[projector, epoch-50]`. Exactly one probe must match per source. |
| `evaluation.probe_std_mode` | `ckpt` or `recompute` normalization statistics, consistent with the trained probe. |
| `data.imagenet_root`, `data.activation_root`, `data.split` | Evaluation data; unset values inherit the recipe. |
| `reference_data.imagenet_root`, `reference_data.activation_root`, `reference_data.split` | Probe-training dataset for class-label mapping; defaults to the training recipe's data. |
| `loader.batch_size`, `loader.num_workers` | Evaluation loader settings. |
| `evaluation.max_batches` | Optional evaluation batch cap; null processes all batches. |
| `evaluation.prediction_agreement_k` | Concept count for prediction agreement, 10 in the example; must lie in `1..sae.k`. |
| `random_baseline.seeds` | Distinct seeds for random orthogonal bases, ten in the example; at least two are required for the sample standard deviation. |
| `random_baseline.mask_seed_offset` | Each repeat's per-image subset generator uses its basis seed plus this offset. |
| `runtime.device` | Execution device. |
| `runtime.seed` | Python/NumPy/PyTorch evaluation seed, reset per source; 10 in the example. |
| `output.output_csv` | Destination for shared/private/total/random Suff/Comp curves. Companion CSVs and metadata are written in the same directory. |

For each image, the reference is the complete mean-pooled reconstruction
`R_full = z_sh @ D_sh + z_pr @ D_pr`. The evaluator ranks ground-truth-class
contributions after clamping negative contributions to zero. Shared and private
contributions use their respective activation and dictionary; total contribution
is their sum at each latent index. Evaluation covers `k=1..sae.k`.

For the selected indices:

- **Shared:** keep only the selected shared contributions; remove only those
  shared contributions from `R_full`.
- **Private:** keep only the selected private contributions; remove only those
  private contributions from `R_full`.
- **Total:** keep the selected indices' shared-plus-private contributions;
  remove those same contributions from `R_full`.

For every group, `R_keep + R_remove = R_full`. Shared/private keep interventions
zero all contributions outside the selected branch and indices. Total selects
paired shared/private indices, rather than ranking two concatenated dictionaries.

The frozen probe supplies the ground-truth logits (including bias):

```text
Suff@k = mean((logit_full - logit_keep) / (abs(logit_full) + 1e-6))
Comp@k = mean((logit_full - logit_remove) / (abs(logit_full) + 1e-6))
```

Scores average batch means. Smaller Suff and larger Comp indicate stronger
selected evidence. These normalized logit changes need not lie in `[0, 1]`.

## Random orthogonal baseline

For each repeat seed, a Gaussian `d x d` matrix undergoes QR decomposition with
sign correction to produce a fixed orthonormal basis `Q`. Its columns are
independent of images and labels. The same seed gives the same basis for every
source of the same representation dimension. Project the full reconstruction
onto this basis, `alpha = R_full @ Q`; these signed coefficients are recomputed
from the representation, rather than copied from the SAE activations.

Each image independently receives a uniform sample without replacement of
`sae.k` directions. The first `k` indices define a nested subset `S_k`:

```text
R_random_keep   = sum(alpha_j * q_j for j in S_k)
R_random_remove = R_full - R_random_keep
```

Keep and remove use the same subset. There is no label-contribution ranking,
coefficient clipping or norm matching: only the selected component count `k`
is matched to the learned concepts. This produces one random baseline per
source, shared by the three learned-group comparisons. `sae.k` must not exceed
the representation dimension. Separate CPU generators for bases and subsets
leave the evaluation loader's RNG sequence unchanged.

The example runs ten repeats. Each repeat first averages its per-batch scores;
the reported random mean and sample standard deviation are then computed across
repeats (`ddof=1`).

## Prediction agreement and outputs

At `evaluation.prediction_agreement_k`, count how often the keep/remove probe's
predicted class matches the full reconstruction's predicted class. This uses
argmax over all classes, independently of the ground-truth class used for
ranking. All evaluated images contribute, including incorrect full predictions.
Agreement pools matching image counts across batches. Random agreement is
reported as a mean and sample standard deviation across repeats.

Use a separate output directory for each evaluation. It contains:

| File | Contents |
| --- | --- |
| Configured `output.output_csv` | `model, group, k, suff, comp, suff_std, comp_std, repeats, num_batches, num_samples`; groups are `sh`, `pr`, `total`, `random`. Learned-group standard deviations are zero because their evaluation is not repeated. |
| `random_repeats.csv` | Per-source, seed and k random Suff/Comp, before aggregation across repeats. |
| `prediction_agreement.csv` | Per-source/group keep/remove agreement at the configured k, with random standard deviations and sample counts. |
| `prediction_agreement_random_repeats.csv` | Per-seed agreement rates and matching-image counts. |
| `evaluation_metadata.json` | Seeds, subset policy, per-source dimensions/counts, normalization and aggregation settings. |

The evaluator uses a shuffled loader with `drop_last=True`; `num_samples`
records the images actually processed, excluding any incomplete final batch.
The example fixes batch size 32 and data seed 10. These affect current-batch
encoder statistics and must be preserved when reproducing a run. No plots are
generated.

Evaluation classes map to the probe-training labels; class count is read from
the probe.

## BatchNorm statistics and attribution objective

Attribution measures the evidence carried by selected concepts for a fixed
source-specific classifier. Each backbone's representations are normalized using
current-batch BatchNorm statistics within the shared encoder. This source-wise
normalization preserves the context in which the concepts encode that backbone's
features. The resulting codes are reused for the full reconstruction and all
keep/remove interventions, so their logit differences measure the selected
concepts' contributions within the same normalized representation.

The SAE and probe parameters, learned BatchNorm affine parameters and running
buffers remain fixed. The loader's batch size and composition define the
normalization context. Input-feature standardization follows
`evaluation.probe_std_mode` separately from encoder BatchNorm.

## Extend

`train_probe.extract_features` defines the representation; `train_projector_heads`
assembles the loader, frozen projector, head and checkpoint callback.
`interventions.evaluate_concept_interventions` defines branch masks and scores;
`random_baseline` defines uniform orthogonal selection. Change these pieces
for another attribution study, while reusing the
[classification-head framework](../classification_head/README.md).
Prefix/layout/grid policies and normalization are described in
[configs/README.md](../../configs/README.md).
