"""Current-batch encoder normalization for source-wise evaluation."""
from types import MethodType

import torch
from torch.nn import functional as F


def _batch_statistics_forward(layer, inputs):
    """Apply current-batch normalization with the learned affine transform."""
    layer._check_input_dim(inputs)
    return F.batch_norm(
        inputs,
        running_mean=None,
        running_var=None,
        weight=layer.weight,
        bias=layer.bias,
        training=True,
        momentum=0.0,
        eps=layer.eps,
    )


def use_current_batch_statistics(encoder):
    """Use current-batch statistics while retaining encoder BatchNorm buffers."""
    for layer in encoder.modules():
        if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
            layer.forward = MethodType(_batch_statistics_forward, layer)
