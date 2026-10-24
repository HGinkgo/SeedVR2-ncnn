"""Shape-parameterized SeedVR2 DiT stack reference used by the ncnn exporter."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import ClassVar, Iterator

import torch
from torch import nn

from .dit_block import DIT_BLOCK_CONTRACT, FixedDitBlock


@dataclass(frozen=True)
class DitStackContract:
    supported_text_tokens: ClassVar[tuple[int, int]] = (58, 64)
    source_shape: tuple[int, int, int] = (1, 8, 8)
    window_shape: tuple[int, int, int] = (4, 3, 3)
    text_tokens: int = 58
    video_channels: int = 33
    text_input_dim: int = 5120
    hidden_dim: int = 2560
    embedding_dim: int = 15360
    block_count: int = 32
    latent_channels: int = 16

    def __post_init__(self) -> None:
        if int(self.text_tokens) not in self.supported_text_tokens:
            supported = " or ".join(str(value) for value in self.supported_text_tokens)
            raise ValueError(f"text_tokens must be {supported}, got {self.text_tokens}")
        for name, value in (("source_shape", self.source_shape), ("window_shape", self.window_shape)):
            if len(value) != 3 or any(int(item) <= 0 for item in value):
                raise ValueError(f"{name} dimensions must be positive triples")

    @property
    def video_tokens(self) -> int:
        time, height, width = self.source_shape
        return time * height * width

    @property
    def patch_width(self) -> int:
        return self.video_channels * 1 * 2 * 2

    @property
    def video_patch_shape(self) -> tuple[int, int]:
        return (self.video_tokens, self.patch_width)

    @property
    def text_input_shape(self) -> tuple[int, int]:
        return (self.text_tokens, self.text_input_dim)

    @property
    def embedding_shape(self) -> tuple[int, int]:
        return (1, self.embedding_dim)

    @property
    def output_width(self) -> int:
        return self.latent_channels * 1 * 2 * 2

    @property
    def output_shape(self) -> tuple[int, int]:
        return (self.video_tokens, self.output_width)

    @property
    def graph_names(self) -> tuple[str, ...]:
        return (
            "dit_input",
            "dit_embedding",
            *(f"dit_block_{index:02d}" for index in range(self.block_count)),
            "dit_output",
        )

    @staticmethod
    def block_shifted(block_index: int) -> bool:
        if int(block_index) < 0:
            raise ValueError("block index must be non-negative")
        return bool(int(block_index) % 2)


DIT_STACK_CONTRACT = DitStackContract()

# The pinned reference's one-layer output Ada cannot consume its 6*D
# embedding. Preserve the two-layer, per-feature packing used by the body.
# This is an explicit compatibility contract, not an upstream acceptance claim.
DIT_OUTPUT_MODULATION = "per-feature-six-first-triplet-v1"


def make_dit_stack_contract(
    text_tokens: int,
    *,
    source_shape: tuple[int, int, int] = DIT_STACK_CONTRACT.source_shape,
    window_shape: tuple[int, int, int] = DIT_STACK_CONTRACT.window_shape,
) -> DitStackContract:
    """Return a contract for one text length and runtime video source grid."""

    return replace(
        DIT_STACK_CONTRACT,
        text_tokens=int(text_tokens),
        source_shape=tuple(int(item) for item in source_shape),
        window_shape=tuple(int(item) for item in window_shape),
    )


def load_dit_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SeedVR2 DiT checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"unexpected SeedVR2 checkpoint structure: {checkpoint}")
    return state


def _copy_tensor(state: dict[str, torch.Tensor], name: str, target: torch.Tensor) -> None:
    source = state.get(name)
    if not isinstance(source, torch.Tensor) or tuple(source.shape) != tuple(target.shape):
        raise ValueError(
            f"checkpoint tensor mismatch for {name}: expected {tuple(target.shape)}, "
            f"got {getattr(source, 'shape', None)}"
        )
    with torch.no_grad():
        target.copy_(source.to(device=target.device, dtype=target.dtype))


def load_dit_block_weights(
    block: FixedDitBlock,
    state: dict[str, torch.Tensor],
    *,
    block_index: int,
) -> int:
    prefix = f"blocks.{int(block_index)}."
    parameters = dict(block.named_parameters())
    missing = [name for name in parameters if prefix + name not in state]
    if missing:
        shared_missing = []
        for name in missing:
            if ".vid." in name:
                candidate = name.replace(".vid.", ".all.")
            elif ".txt." in name:
                candidate = name.replace(".txt.", ".all.")
            else:
                candidate = name
            if prefix + candidate not in state:
                shared_missing.append(name)
        if shared_missing:
            raise ValueError(f"checkpoint is missing block {block_index} tensors: {shared_missing[:5]}")
    with torch.no_grad():
        for name, parameter in parameters.items():
            source_name = prefix + name
            if source_name not in state:
                if ".vid." in name:
                    source_name = prefix + name.replace(".vid.", ".all.")
                elif ".txt." in name:
                    source_name = prefix + name.replace(".txt.", ".all.")
            _copy_tensor(state, source_name, parameter)
    return len(parameters)


def iter_loaded_blocks(
    state: dict[str, torch.Tensor],
    *,
    contract: DitStackContract = DIT_STACK_CONTRACT,
) -> Iterator[tuple[int, FixedDitBlock]]:
    """Yield one populated block at a time to keep peak host memory bounded."""

    for block_index in range(contract.block_count):
        block = FixedDitBlock(
            source_shape=contract.source_shape,
            window_shape=contract.window_shape,
            text_tokens=contract.text_tokens,
            shifted=contract.block_shifted(block_index),
            is_last_layer=block_index == contract.block_count - 1,
        ).float().eval()
        load_dit_block_weights(block, state, block_index=block_index)
        yield block_index, block


def timestep_embedding(timestep: torch.Tensor, embedding_dim: int = 256) -> torch.Tensor:
    """Match diffusers.get_timestep_embedding(..., flip_sin_to_cos=False)."""

    if timestep.ndim == 0:
        timestep = timestep.reshape(1)
    if timestep.ndim != 1:
        raise ValueError(f"expected a scalar or rank-1 timestep, got {tuple(timestep.shape)}")
    # Keep the traced input rank explicit for ncnn.  A direct ``[None]`` on a
    # rank-1 input makes PNNX materialize the timestep as a rank-3 tensor
    # before the first Linear.  ncnn's Vulkan InnerProduct flatten path then
    # selects its sum8 kernel and corrupts this particular 3-D activation.
    # Reshaping to the same one-dimensional shape preserves the PyTorch
    # contract while keeping the exported Linear input rank-2.
    timestep = timestep.reshape(-1)
    half_dim = embedding_dim // 2
    exponent = -math.log(10000.0) * torch.arange(
        half_dim, device=timestep.device, dtype=torch.float32
    )
    exponent = exponent / half_dim
    values = timestep.float()[:, None] * torch.exp(exponent)[None, :]
    return torch.cat((torch.sin(values), torch.cos(values)), dim=-1)


class FixedDitEmbedding(nn.Module):
    def __init__(self, state: dict[str, torch.Tensor], *, contract: DitStackContract = DIT_STACK_CONTRACT) -> None:
        super().__init__()
        self.contract = contract
        self.proj_in = nn.Linear(256, contract.hidden_dim)
        self.proj_hid = nn.Linear(contract.hidden_dim, contract.hidden_dim)
        self.proj_out = nn.Linear(contract.hidden_dim, contract.embedding_dim)
        # These frequencies are independent of the runtime timestep. Lift
        # them as a buffer so PT2 serializes their exact values as weights,
        # rather than requiring a runtime arange/device-creation operator.
        exponent = -math.log(10000.0) * torch.arange(128, dtype=torch.float32)
        self.register_buffer("frequencies", torch.exp(exponent / 128).reshape(1, 128))
        _copy_tensor(state, "emb_in.proj_in.weight", self.proj_in.weight)
        _copy_tensor(state, "emb_in.proj_in.bias", self.proj_in.bias)
        _copy_tensor(state, "emb_in.proj_hid.weight", self.proj_hid.weight)
        _copy_tensor(state, "emb_in.proj_hid.bias", self.proj_hid.bias)
        _copy_tensor(state, "emb_in.proj_out.weight", self.proj_out.weight)
        _copy_tensor(state, "emb_in.proj_out.bias", self.proj_out.bias)
        self.eval()

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        values = timestep.reshape(-1, 1).float() * self.frequencies
        emb = torch.cat((torch.sin(values), torch.cos(values)), dim=-1)
        emb = emb.to(dtype=self.proj_in.weight.dtype)
        emb = torch.nn.functional.silu(self.proj_in(emb))
        emb = torch.nn.functional.silu(self.proj_hid(emb))
        return self.proj_out(emb)


class FixedDitInput(nn.Module):
    def __init__(self, state: dict[str, torch.Tensor], *, contract: DitStackContract = DIT_STACK_CONTRACT) -> None:
        super().__init__()
        self.video = nn.Linear(contract.patch_width, contract.hidden_dim)
        self.text = nn.Linear(contract.text_input_dim, contract.hidden_dim)
        _copy_tensor(state, "vid_in.proj.weight", self.video.weight)
        _copy_tensor(state, "vid_in.proj.bias", self.video.bias)
        _copy_tensor(state, "txt_in.weight", self.text.weight)
        _copy_tensor(state, "txt_in.bias", self.text.bias)
        self.eval()

    def forward(self, video_patches: torch.Tensor, text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.video(video_patches), self.text(text)


class FixedDitOutput(nn.Module):
    def __init__(self, state: dict[str, torch.Tensor], *, contract: DitStackContract = DIT_STACK_CONTRACT) -> None:
        super().__init__()
        self.contract = contract
        if contract.embedding_dim != contract.hidden_dim * 6:
            raise ValueError("output Ada requires embedding_dim == 6 * hidden_dim")
        self.norm = nn.Parameter(torch.empty(contract.hidden_dim), requires_grad=False)
        self.shift = nn.Parameter(torch.empty(contract.hidden_dim), requires_grad=False)
        self.scale = nn.Parameter(torch.empty(contract.hidden_dim), requires_grad=False)
        self.proj = nn.Linear(contract.hidden_dim, contract.output_width)
        _copy_tensor(state, "vid_out_norm.weight", self.norm)
        _copy_tensor(state, "vid_out_ada.out_shift", self.shift)
        _copy_tensor(state, "vid_out_ada.out_scale", self.scale)
        _copy_tensor(state, "vid_out.proj.weight", self.proj.weight)
        _copy_tensor(state, "vid_out.proj.bias", self.proj.bias)
        self.eval()

    def forward(self, video: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        if video.ndim != 2 or video.shape[1] != self.contract.hidden_dim:
            raise ValueError("output Ada requires a rank-2 video matrix with hidden_dim features")
        if tuple(embedding.shape) != self.contract.embedding_shape:
            raise ValueError("output Ada requires one packed 6 * hidden_dim embedding")
        variance = video.float().pow(2).mean(dim=-1, keepdim=True)
        video = video * torch.rsqrt(variance + 1.0e-5)
        video = video * self.norm
        # Retain six adjacent values per feature before selecting the first
        # triplet. Slicing the first 3*D values mixes different features.
        modulation = embedding.reshape(self.contract.hidden_dim, 6)[:, :3]
        shift_a, scale_a, _ = modulation.unbind(dim=-1)
        video = video * (scale_a + self.scale) + (shift_a + self.shift)
        return self.proj(video)


class FixedDitStack(nn.Module):
    """Complete fixed one-step stack, loading one DiT block at a time."""

    def __init__(self, checkpoint: Path, *, contract: DitStackContract = DIT_STACK_CONTRACT) -> None:
        super().__init__()
        self.contract = contract
        self._state = load_dit_state(checkpoint)
        self.input = FixedDitInput(self._state, contract=contract)
        self.embedding = FixedDitEmbedding(self._state, contract=contract)
        self.output = FixedDitOutput(self._state, contract=contract)

    def forward(
        self,
        video_patches: torch.Tensor,
        text: torch.Tensor,
        timestep: torch.Tensor,
        *,
        collect_intermediates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        if tuple(video_patches.shape) != self.contract.video_patch_shape:
            raise ValueError(f"expected video patches {self.contract.video_patch_shape}, got {tuple(video_patches.shape)}")
        if tuple(text.shape) != self.contract.text_input_shape:
            raise ValueError(f"expected text {self.contract.text_input_shape}, got {tuple(text.shape)}")
        vid, txt = self.input(video_patches, text)
        emb = self.embedding(timestep)
        traces: list[dict[str, torch.Tensor]] = []
        for block_index, block in iter_loaded_blocks(self._state, contract=self.contract):
            with torch.inference_mode():
                vid, txt = block(vid, txt, emb)
            if collect_intermediates:
                traces.append({"block_index": block_index, "vid": vid.detach().clone(), "txt": txt.detach().clone()})
            del block
        output = self.output(vid, emb)
        if collect_intermediates:
            return output, traces
        return output


__all__ = [
    "DIT_STACK_CONTRACT",
    "DitStackContract",
    "FixedDitEmbedding",
    "FixedDitInput",
    "FixedDitOutput",
    "FixedDitStack",
    "iter_loaded_blocks",
    "make_dit_stack_contract",
    "load_dit_block_weights",
    "load_dit_state",
    "timestep_embedding",
]
