# Train retrieval heads on frozen GB-SAE features

Train and evaluate embedding heads for cross-source image/category retrieval.
The representation model stays frozen. Use `method: gbsae` for shared decoded
features or `method: projected` for projected backbone features.

## Run

```bash
python -m experiments.downstream.downstream_retrieval.run_experiment --config configs/experiments/downstream_retrieval/config.server.yaml
python -m experiments.downstream.downstream_retrieval.run_experiment --config configs/experiments/downstream_retrieval/projected.yaml
python -m experiments.downstream.downstream_retrieval.run_experiment --config configs/experiments/downstream_retrieval/evaluate.yaml
```

The first job trains GB-SAE's tied retrieval head. `projected.yaml` inherits the
shared settings and trains PROJECTED heads. `evaluate.yaml` inherits settings and
loads a saved head without training. Run from the repository root; all settings
belong in YAML.

## Representations and directions

For N configured sources:

| Method | Representation / heads | Training directions | Reported directions |
| --- | --- | --- | --- |
| `gbsae` | Pool each source's shared reconstruction, average sources into FUSED; one tied head for FUSED and single-source views | `FUSED -> source` and `source -> FUSED`, 2N directions | Mean of N `FUSED -> source` directions, plus AllHit metrics |
| `projected` | Pool projected input features; one head per source | Every ordered cross-source pair, N(N-1) directions | Mean across the same directed pairs |

Source identities/order and preprocessing are read from `gbsae_config`; use at
least two unique sources and do not name a source `FUSED`. Standardization uses
checkpoint statistics when enabled. Each source is pooled before fusion, so grids
may differ while their input metadata must remain correct.

## Model, data and frozen-feature cache parameters

| Field | Meaning |
| --- | --- |
| `method` | `gbsae` or `projected`. |
| `gbsae_config` | Authoritative training YAML for sources, dimensions, projectors, SAE and preprocessing. |
| `checkpoints.gbsae` | Matching representation checkpoint, including source projectors. |
| `data.train`, `data.val` | Each contains `imagenet_root`, `activation_root` and physical `split`; optional `target_class`/`combined_npz` select data/storage. |
| `data.expected_train_classes` | Assert the training class count; null disables this count assertion. |
| `data.alignment_checks` | Number of image/cache spot checks per split. WNID class-mapping checks always run independently of this count. |
| `preprocessing.token_pooling` | `mean`, `max` or `sum` over each source's tokens. |
| `cache.batch_size`, `cache.num_workers` | Frozen-feature extraction loader settings. |
| `cache.dtype` | Storage precision of the extracted image features, e.g. `float32`. |

Train/val must identify separate sample sets and use compatible class labels.
Training labels must form `0..C-1`; validation WNIDs must be covered by training.
The supplied examples use distinct roots whose physical directory is named
`train` for both subsets; for a conventional layout set val's root/split to the
actual `val` data. Each image tree must match its activation files.

This workflow creates an additional image-feature cache for head training; it is
different from the upstream token activation cache. Feature caches record sample
order, labels, checkpoint and representation settings. After changing those
settings, set `execution.rebuild_cache: true` to regenerate them.

## Head and optimization parameters

| Field | Meaning |
| --- | --- |
| `head.type` | `linear` or `mlp` embedding head. |
| `head.output_dim` | Width of the normalized retrieval embedding. |
| `head.hidden_dim`, `head.dropout` | MLP hidden width/dropout; unused by a linear head. |
| `training.epochs`, `training.lr`, `training.weight_decay` | Head AdamW settings; an epoch cosine schedule is applied. |
| `training.classes_per_batch` | P classes sampled per training batch. |
| `training.samples_per_class` | K images sampled per class; batch size is P x K. |
| `training.batches_per_epoch` | Explicit batch count, or null to derive it from dataset size. |
| `training.temperature` | Positive temperature dividing contrastive cosine logits. |
| `training.instance_loss_weight` | Weight of paired-image contrastive loss. |
| `training.category_loss_weight` | Weight of same-class contrastive loss. |
| `training.gradient_clip_norm` | Maximum head gradient norm. |
| `training.eval_every` | Validate/select the best head every this many epochs; final epoch is evaluated too. |
| `seed` | Seed for this head experiment's initialization/sampling. |
| `device` | Frozen-feature extraction and head-training device. |

`classes_per_batch` cannot exceed the number of training classes. Sampling uses
replacement within a class if it has fewer than `samples_per_class` images.
Use at least two samples per class for same-class positives beyond the paired image.

## Evaluation, execution and outputs

| Field | Meaning |
| --- | --- |
| `evaluation.ks` | Recall ranks, e.g. `[1,5,10,50]`; choose ranks supported by the gallery. |
| `evaluation.embedding_batch_size` | Cached image features encoded by the head per inference batch. |
| `evaluation.query_chunk_size` | Queries compared with the gallery per scoring chunk; reduce it to lower scoring memory. |
| `evaluation.device` | Device for metric scoring; may differ from head-training device. |
| `execution.cache_only` | Build/audit frozen-feature caches, then stop. |
| `execution.eval_only` | Load a head and evaluate without optimizer updates. Training and val datasets/caches are still required for audits. |
| `execution.rebuild_cache` | Regenerate the frozen-feature caches. |
| `execution.head_checkpoint` | Head checkpoint for eval-only; null uses the run's `best.pth`. |
| `output_root` | Artifact root; outputs are grouped under the method name. |
| `run_name_by_method` | Optional method-to-output-subdirectory mapping; each value must be a simple directory name. |

Instance Recall includes the paired image. Category Recall and category mAP
exclude it. Best-head selection uses validation macro category mAP. Training
automatically evaluates at `eval_every` and evaluates the best saved head at the
end; an extra evaluation job is optional.

Outputs include frozen-feature caches, `method_spec.json`, `dataset_audit.json`,
`training_history.csv`, `best.pth`, `last.pth` and validation metric JSON/CSV files.
The frozen GB-SAE checkpoint is not overwritten by head training.

## Extend

`configuration.py` resolves representation/data settings; `feature_providers.py`
extracts frozen features; `specs.py` defines views, head sharing and directions.
Change the head/loss in `losses.py`, metrics in `metrics.py`, or outer workflow in
`run_experiment.py` independently when creating a retrieval experiment.
