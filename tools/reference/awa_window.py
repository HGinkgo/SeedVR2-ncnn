"""Tensor-only AWA window transforms for export and backend comparison."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import nn

from .seedvr2_baseline import _index_maps, make_windows


Size3 = Tuple[int, int, int]


def _size3(value: Sequence[int], name: str) -> Size3:
    if len(value) != 3 or any(int(item) <= 0 for item in value):
        raise ValueError(f"{name} dimensions must be positive triples")
    return tuple(int(item) for item in value)


class AwaWindowPartition(nn.Module):
    """Partition a fixed ``(T,H,W,C)`` tensor in upstream AWA order.

    The source shape and window configuration are model attributes. The
    generated index is a persistent buffer, so the exported graph contains no
    Python window list or runtime slice construction.
    """

    def __init__(
        self,
        size: Sequence[int],
        num_windows: Sequence[int],
        shifted: bool = False,
    ) -> None:
        super().__init__()
        self.source_size = _size3(size, "size")
        self.num_windows = _size3(num_windows, "num_windows")
        self.shifted = bool(shifted)
        windows = make_windows(self.source_size, self.num_windows, shifted=self.shifted)
        target_index, _ = _index_maps(self.source_size, windows)
        self.register_buffer("target_index", target_index)
        self.window_count = len(windows)
        self.partitioned_tokens = int(target_index.numel())

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        flat = source.reshape(self.partitioned_tokens, source.shape[-1])
        return flat.index_select(0, self.target_index)


class AwaWindowReverse(nn.Module):
    """Reverse an :class:`AwaWindowPartition` result to ``(T,H,W,C)``."""

    def __init__(
        self,
        size: Sequence[int],
        num_windows: Sequence[int],
        shifted: bool = False,
    ) -> None:
        super().__init__()
        self.source_size = _size3(size, "size")
        self.num_windows = _size3(num_windows, "num_windows")
        self.shifted = bool(shifted)
        windows = make_windows(self.source_size, self.num_windows, shifted=self.shifted)
        _, reverse_index = _index_maps(self.source_size, windows)
        self.register_buffer("reverse_index", reverse_index)
        self.partitioned_tokens = int(reverse_index.numel())

    def forward(self, partitioned: torch.Tensor) -> torch.Tensor:
        restored = partitioned.index_select(0, self.reverse_index)
        return restored.reshape(
            self.source_size[0], self.source_size[1], self.source_size[2], partitioned.shape[-1]
        )


__all__ = ["AwaWindowPartition", "AwaWindowReverse"]
