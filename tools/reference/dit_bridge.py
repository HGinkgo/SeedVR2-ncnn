"""Fixed projection boundaries around the one-block SeedVR2 bridge."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .dit_pipeline import DIT_PIPELINE_CONTRACT


def _load_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    state = torch.load(Path(checkpoint), map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"unexpected SeedVR2 checkpoint structure: {checkpoint}")
    return state


def _copy(state: dict[str, torch.Tensor], name: str, target: torch.Tensor) -> None:
    source = state.get(name)
    if not isinstance(source, torch.Tensor) or tuple(source.shape) != tuple(target.shape):
        raise ValueError(
            f"checkpoint tensor mismatch for {name}: expected {tuple(target.shape)}, "
            f"got {getattr(source, 'shape', None)}"
        )
    with torch.no_grad():
        target.copy_(source.to(dtype=target.dtype))


class FixedDitInputProjection(nn.Module):
    """Project C++-assembled patches and encoded text into DiT token width."""

    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        self.vid_in = nn.Linear(132, 2560)
        self.txt_in = nn.Linear(5120, 2560)
        state = _load_state(checkpoint)
        _copy(state, "vid_in.proj.weight", self.vid_in.weight)
        _copy(state, "vid_in.proj.bias", self.vid_in.bias)
        _copy(state, "txt_in.weight", self.txt_in.weight)
        _copy(state, "txt_in.bias", self.txt_in.bias)
        self.eval()

    def forward(self, video_patches: torch.Tensor, text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.vid_in(video_patches), self.txt_in(text)


class FixedDitOutputProjection(nn.Module):
    """Project block video tokens back to flattened 2x2x16 latent patches."""

    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        self.vid_out = nn.Linear(2560, 64)
        state = _load_state(checkpoint)
        _copy(state, "vid_out.proj.weight", self.vid_out.weight)
        _copy(state, "vid_out.proj.bias", self.vid_out.bias)
        self.eval()

    def forward(self, video_tokens: torch.Tensor) -> torch.Tensor:
        return self.vid_out(video_tokens)


__all__ = ["FixedDitInputProjection", "FixedDitOutputProjection", "DIT_PIPELINE_CONTRACT"]
