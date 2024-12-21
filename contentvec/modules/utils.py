# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import collections
import contextlib
import copy
import importlib
import logging
import os
import sys
import warnings
from itertools import accumulate
from typing import Callable, Dict, List, Optional

import math
import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


def get_incremental_state(
    module: "MultiheadAttention",
    incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
    key: str,
) -> Optional[Dict[str, Optional[Tensor]]]:
    """Helper for getting incremental state for an nn.Module."""
    return module.get_incremental_state(incremental_state, key)


def set_incremental_state(
    module: "MultiheadAttention",
    incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
    key: str,
    value: Dict[str, Optional[Tensor]],
) -> Optional[Dict[str, Dict[str, Optional[Tensor]]]]:
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        result = module.set_incremental_state(incremental_state, key, value)
        if result is not None:
            incremental_state = result
    return incremental_state


def strip_pad(tensor, pad):
    return tensor[tensor.ne(pad)]


def buffered_arange(max, device="cpu"):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor().to(device)
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


def softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)


def log_softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.log_softmax(x.float(), dim=dim)
    else:
        return F.log_softmax(x, dim=dim, dtype=torch.float32)


def relu_squared(x: torch.Tensor):
    return F.relu(x).pow(2)

def gelu_accurate(x):
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return (
        0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))
    )


def gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x.float()).type_as(x)

def get_activation_fn(activation: str) -> Callable:
    """Returns the activation function corresponding to `activation`"""

    if activation == "relu":
        return F.relu
    elif activation == "relu_squared":
        return relu_squared
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_fast":
        deprecation_warning(
            "--activation-fn=gelu_fast has been renamed to gelu_accurate"
        )
        return gelu_accurate
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "swish":
        return torch.nn.SiLU
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


def get_available_activation_fns() -> List:
    return [
        "relu",
        "gelu",
        "gelu_fast",  # deprecated
        "gelu_accurate",
        "tanh",
        "linear",
    ]

def is_xla_tensor(tensor):
    return torch.is_tensor(tensor) and tensor.device.type == "xla"


def index_put(tensor, indices, value):
    if is_xla_tensor(tensor):
        for _ in range(indices.dim(), tensor.dim()):
            indices = indices.unsqueeze(-1)
        if indices.size(-1) < tensor.size(-1):
            indices = indices.expand_as(tensor)
        tensor = torch.mul(tensor, ~indices) + torch.mul(value, indices)
    else:
        tensor[indices] = value
    return tensor