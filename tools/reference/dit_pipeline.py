"""Fixed VAE-latent to one SeedVR2 DiT block bridge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .dit_block import FixedDitBlock, load_official_block_weights


@dataclass(frozen=True)
class DitPipelineContract:
    """The fixed one-frame bridge used by the first C++ integration smoke."""

    source_shape: tuple[int, int, int] = (1, 8, 8)
    text_tokens: int = 58
    video_channels: int = 33
    latent_channels: int = 16
    text_input_dim: int = 5120
    hidden_dim: int = 2560
    embedding_dim: int = 15360

    @property
    def video_shape(self) -> tuple[int, int, int, int, int]:
        t, h, w = self.source_shape
        return (1, self.video_channels, t, h * 2, w * 2)

    @property
    def text_shape(self) -> tuple[int, int]:
        return (self.text_tokens, self.text_input_dim)

    @property
    def embedding_shape(self) -> tuple[int, int]:
        return (1, self.embedding_dim)

    @property
    def output_shape(self) -> tuple[int, int, int, int, int]:
        t, h, w = self.source_shape
        return (1, self.latent_channels, t, h * 2, w * 2)

    @property
    def video_tokens(self) -> int:
        t, h, w = self.source_shape
        return t * h * w


DIT_PIPELINE_CONTRACT = DitPipelineContract()


class FixedDitPipeline(nn.Module):
    """Patch and run one official DiT block for a fixed VAE latent grid.

    The bridge deliberately accepts the already encoded text and timestep
    embedding.  Text encoding, timestep scheduling, and the remaining 31 DiT
    blocks are separate model-boundary work; keeping them outside this probe
    makes its ncnn contract explicit and reproducible.
    """

    def __init__(
        self,
        checkpoint: Path,
        *,
        contract: DitPipelineContract = DIT_PIPELINE_CONTRACT,
    ) -> None:
        super().__init__()
        self.contract = contract
        self.vid_in = nn.Linear(contract.video_channels * 1 * 2 * 2, contract.hidden_dim)
        self.txt_in = nn.Linear(contract.text_input_dim, contract.hidden_dim)
        self.block = FixedDitBlock(
            source_shape=contract.source_shape,
            text_tokens=contract.text_tokens,
        )
        self.vid_out = nn.Linear(contract.hidden_dim, contract.latent_channels * 1 * 2 * 2)

        state = torch.load(Path(checkpoint), map_location="cpu", mmap=True, weights_only=False)
        if not isinstance(state, dict):
            raise ValueError(f"unexpected SeedVR2 checkpoint structure: {checkpoint}")
        self._copy_tensor(state, "vid_in.proj.weight", self.vid_in.weight)
        self._copy_tensor(state, "vid_in.proj.bias", self.vid_in.bias)
        self._copy_tensor(state, "txt_in.weight", self.txt_in.weight)
        self._copy_tensor(state, "txt_in.bias", self.txt_in.bias)
        self._copy_tensor(state, "vid_out.proj.weight", self.vid_out.weight)
        self._copy_tensor(state, "vid_out.proj.bias", self.vid_out.bias)
        load_official_block_weights(self.block, Path(checkpoint), block_index=0)
        self.eval()

    @staticmethod
    def _copy_tensor(state: dict[str, torch.Tensor], name: str, target: torch.Tensor) -> None:
        source = state.get(name)
        if not isinstance(source, torch.Tensor) or tuple(source.shape) != tuple(target.shape):
            raise ValueError(
                f"checkpoint tensor mismatch for {name}: expected {tuple(target.shape)}, "
                f"got {getattr(source, 'shape', None)}"
            )
        with torch.no_grad():
            target.copy_(source.to(dtype=target.dtype))

    def forward(
        self,
        video: torch.Tensor,
        text: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        b, channels, frames, height, width = video.shape
        if (b, channels, frames, height, width) != self.contract.video_shape:
            raise ValueError(f"expected video shape {self.contract.video_shape}, got {tuple(video.shape)}")
        if tuple(text.shape) != self.contract.text_shape:
            raise ValueError(f"expected text shape {self.contract.text_shape}, got {tuple(text.shape)}")
        if tuple(embedding.shape) != self.contract.embedding_shape:
            raise ValueError(
                f"expected embedding shape {self.contract.embedding_shape}, got {tuple(embedding.shape)}"
            )

        patch_t, patch_h, patch_w = 1, 2, 2
        tokens = video.permute(0, 2, 3, 4, 1).reshape(
            b, self.contract.video_tokens, channels * patch_t * patch_h * patch_w
        )
        vid = self.vid_in(tokens).reshape(self.contract.video_tokens, self.contract.hidden_dim)
        txt = self.txt_in(text)
        vid, _ = self.block(vid, txt, embedding)

        patch = self.vid_out(vid).reshape(
            b,
            frames // patch_t,
            height // patch_h,
            width // patch_w,
            patch_t,
            patch_h,
            patch_w,
            self.contract.latent_channels,
        )
        return patch.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(
            b, self.contract.latent_channels, frames, height, width
        )


__all__ = ["DIT_PIPELINE_CONTRACT", "DitPipelineContract", "FixedDitPipeline"]
