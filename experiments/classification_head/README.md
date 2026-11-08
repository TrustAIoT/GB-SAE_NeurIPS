# Reusable classification-head training

Train a caller-supplied classifier on frozen features. The core accepts ordinary
PyTorch datasets/loaders, a feature callback, head, optimizer and loss. ImageNet
and GB-SAE support is provided by separate optional adapters.

## Minimal example

```python
import torch
from torch.utils.data import TensorDataset
from experiments.classification_head import (
    LinearProbe, TrainingConfig, fit, make_dataloader,
)

dataset = TensorDataset(torch.randn(32, 128), torch.randint(0, 10, (32,)))
loader = make_dataloader(dataset, batch_size=8, num_workers=0, pin_memory=False)
head = LinearProbe(dim=128, num_classes=10).to("cpu")
optimizer = torch.optim.SGD(head.parameters(), lr=0.01)
history = fit(head, loader, config=TrainingConfig(epochs=2), optimizer=optimizer)
print(history[-1].loss, history[-1].acc1)
```

The loader yields `(inputs, targets)`. Without a feature callback, `inputs` must
be a `[B,D]` tensor. Targets are integer class indices in `[0,num_classes)`.
`LinearProbe` is one biased linear layer; another `nn.Module` returning `[B,C]`
logits can be supplied instead. Put the head on its desired device before `fit`.

## Parameters and return values

| API / parameter | Meaning |
| --- | --- |
| `LinearProbe(dim, num_classes)` | Input feature width and classifier output class count. |
| `make_dataloader(dataset, batch_size=...)` | Construct a standard loader; accepts `shuffle`, `num_workers`, `pin_memory`, `generator`, `drop_last` and extra DataLoader kwargs such as `collate_fn`. |
| `TrainingConfig.epochs` | Positive number of epochs. |
| `TrainingConfig.lr` | AdamW learning rate when `fit` constructs the optimizer. |
| `TrainingConfig.weight_decay` | AdamW weight decay when `fit` constructs the optimizer. |
| `TrainingConfig.max_batches` | Optional per-epoch batch limit; `None` processes the complete loader. |
| `fit(..., feature_fn=...)` | Callback from loader inputs to `[B,D]` frozen features. `None` uses the input tensor directly. |
| `fit(..., optimizer=...)` | Optional caller-owned optimizer. Its settings take precedence over the default AdamW settings. |
| `fit(..., criterion=...)` | Optional loss; default is cross entropy. Must return a scalar batch-mean loss for correct aggregation. |
| `fit(..., on_epoch_end=...)` | Callback `(epoch, metrics, head)` after each one-based epoch. Use it for saves, logging or scheduling. |
| `train_one_epoch(...)` | Run a single epoch with explicit loader, feature function, optimizer, criterion, device and optional batch cap. |
| `CSVLogger(path)` | Callable `(epoch, metrics)` writing an epoch CSV; creates its parent directory. |
| `save_head(path, head, epoch=..., metadata=...)` | Write `probe_state_dict`, epoch and caller-supplied metadata. |

`fit` returns a list of `EpochMetrics(loss, acc1, acc5, num_samples)`.
Loss/accuracy are sample-weighted; accuracies are percentages. Top-5 uses
`min(5,num_classes)`. The core trainer does not write files unless a callback
uses an output helper.

## Frozen feature functions

`feature_fn` receives the loader's input object, including nested dictionaries
when desired. It owns device transfer, extraction, pooling and fusion:

```python
feature_model.eval()

def feature_fn(inputs):
    tokens = feature_model(inputs["images"].to("cuda"))
    return tokens.mean(dim=1)

# head.to("cuda") must accept the feature width returned above.
# history = fit(head, loader, feature_fn=feature_fn,
#               config=TrainingConfig(epochs=50))
```

The training loop runs feature extraction under `torch.no_grad()` and optimizes
the head. The feature callback defines the frozen representation and its
normalization policy. `mean_pool_tokens` averages a `[B,T,D]` sequence over T.
Experiments may implement another representation inside their own feature callback.

## Save and inspect an experiment

```python
from pathlib import Path
from experiments.classification_head import CSVLogger, save_head

output = Path("artifacts/my_classifier")
logger = CSVLogger(output / "training.csv")

def on_epoch_end(epoch, metrics, trained_head):
    logger(epoch, metrics)
    if epoch % 10 == 0:
        save_head(output / f"epoch_{epoch}.pth", trained_head, epoch=epoch,
                  metadata={"representation": "my_features"})

# history = fit(head, loader, config=TrainingConfig(epochs=50),
#               on_epoch_end=on_epoch_end)
```

For debugging or a custom scheduler, call `train_one_epoch` from your own outer
loop and inspect features, gradients and metrics between calls.

## Modules and GB-SAE adapters

| Module | Responsibility |
| --- | --- |
| `model.py` | Linear head definition |
| `training.py` | Training config, epoch metrics, `fit` and `train_one_epoch` |
| `data.py` | Generic DataLoader helper |
| `pooling.py` | Mean token pooling |
| `io.py` | CSV/checkpoint helpers |
| `cli.py` | YAML field groups used by experiment-owned entry schemas |
| `activation_data.py` | Cached ImageNet loaders, statistics modes and WNID label mapping |
| `gbsae_features.py` | Frozen projectors/SAE and projected-token extraction |

`make_activation_loader` selects sources and checkpoint/recomputed statistics.
`projected_tokens` applies an optional `(source,tokens)` transform and returns
source-to-`[B,T,D]` tensors; pooling/fusion remains with the caller.

[Attribution](../attribution/README.md) assembles per-source projected classifiers.
[Downstream classification](../downstream/downstream_classification/README.md)
assembles one fused shared classifier. Those guides document complete YAML jobs,
representation choices, saved-head paths and evaluation commands.
