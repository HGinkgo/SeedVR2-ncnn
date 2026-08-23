"""Tensor-only AWA attention sequence layout for export comparison."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .seedvr2_baseline import _index_maps, make_windows


class AwaAttentionPack(nn.Module):
    """Interleave each video window with one copy of the text sequence.

    This is the fixed-shape equivalent of upstream ``repeat_concat_idx``.
    Inputs use ``(T,H,W,3,heads,head_dim)`` for video QKV and
    ``(text_tokens,3,heads,head_dim)`` for text QKV. The output is a single
    ragged-attention sequence with cumulative lengths for each window.
    """

    def __init__(
        self,
        size: Sequence[int],
        num_windows: Sequence[int],
        text_tokens: int,
        shifted: bool = False,
    ) -> None:
        super().__init__()
        if len(size) != 3 or any(int(item) <= 0 for item in size):
            raise ValueError("size dimensions must be positive triples")
        if len(num_windows) != 3 or any(int(item) <= 0 for item in num_windows):
            raise ValueError("num_windows dimensions must be positive triples")
        if int(text_tokens) <= 0:
            raise ValueError("text_tokens must be positive")

        self.source_size = tuple(int(item) for item in size)
        self.num_windows = tuple(int(item) for item in num_windows)
        self.text_tokens = int(text_tokens)
        self.shifted = bool(shifted)
        windows = make_windows(self.source_size, self.num_windows, shifted=self.shifted)
        target_index, reverse_index = _index_maps(self.source_size, windows)
        source_token_count = int(target_index.numel())

        sequence_index = []
        video_positions = []
        text_positions = []
        cursor = 0
        window_lengths = []
        source_index = torch.arange(source_token_count, dtype=torch.long).reshape(*self.source_size)
        for window in windows:
            time, height, width = window
            window_index = source_index[
                time[0] : time[1], height[0] : height[1], width[0] : width[1]
            ].reshape(-1)
            sequence_index.extend(window_index.tolist())
            video_positions.extend(range(cursor, cursor + len(window_index)))
            cursor += len(window_index)
            sequence_index.extend(
                (source_token_count + torch.arange(self.text_tokens, dtype=torch.long)).tolist()
            )
            text_positions.append(torch.arange(cursor, cursor + self.text_tokens, dtype=torch.long))
            cursor += self.text_tokens
            window_lengths.append(len(window_index) + self.text_tokens)

        self.register_buffer("sequence_index", torch.tensor(sequence_index, dtype=torch.long))
        self.register_buffer("video_positions", torch.tensor(video_positions, dtype=torch.long))
        self.register_buffer("text_positions", torch.stack(text_positions))
        self.register_buffer(
            "cu_seqlens",
            torch.tensor([0] + list(torch.tensor(window_lengths).cumsum(0).tolist()), dtype=torch.long),
        )
        self.register_buffer("reverse_index", reverse_index)
        self.source_token_count = source_token_count
        self.window_count = len(windows)

    def forward(
        self, vid_qkv: torch.Tensor, txt_qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        heads = vid_qkv.shape[-2]
        head_dim = vid_qkv.shape[-1]
        vid_flat = vid_qkv.reshape(self.source_token_count, 3, heads, head_dim)
        txt_flat = txt_qkv.reshape(self.text_tokens, 3, heads, head_dim)
        source = torch.cat((vid_flat, txt_flat), dim=0)
        return source.index_select(0, self.sequence_index), self.cu_seqlens


class AwaAttentionUnpack(nn.Module):
    """Recover video tokens and coalesce repeated text tokens."""

    def __init__(
        self,
        size: Sequence[int],
        num_windows: Sequence[int],
        text_tokens: int,
        shifted: bool = False,
    ) -> None:
        super().__init__()
        pack = AwaAttentionPack(size, num_windows, text_tokens, shifted=shifted)
        self.source_size = pack.source_size
        self.text_tokens = pack.text_tokens
        self.window_count = pack.window_count
        self.source_token_count = pack.source_token_count
        self.register_buffer("video_positions", pack.video_positions)
        self.register_buffer("text_positions", pack.text_positions)
        self.register_buffer("reverse_index", pack.reverse_index)

    def forward(self, packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        video_partitioned = packed.index_select(0, self.video_positions)
        video_flat = video_partitioned.index_select(0, self.reverse_index)
        video = video_flat.reshape(*self.source_size, packed.shape[-2], packed.shape[-1])
        text_repeated = packed.index_select(0, self.text_positions.reshape(-1))
        text = text_repeated.reshape(
            self.window_count, self.text_tokens, packed.shape[-2], packed.shape[-1]
        ).mean(0)
        return video, text


__all__ = ["AwaAttentionPack", "AwaAttentionUnpack"]
