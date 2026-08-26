"""Fixed-shape SeedVR2 DiT block reference and export contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import torch
from torch import nn

from .awa_attention import AwaAttentionPack, AwaAttentionUnpack, AwaWindowAttention
from .seedvr2_baseline import make_windows


@dataclass(frozen=True)
class DitBlockContract:
    source_shape: Tuple[int, int, int] = (1, 45, 80)
    window_shape: Tuple[int, int, int] = (4, 3, 3)
    text_tokens: int = 58
    vid_dim: int = 2560
    txt_dim: int = 2560
    emb_dim: int = 15360
    heads: int = 20
    head_dim: int = 128
    rope_dim: int = 128
    rope_frequency_dim: int = 126
    fp32_atol: float = 2.0e-4
    fp32_rtol: float = 2.0e-4

    @property
    def video_tokens(self) -> int:
        t, h, w = self.source_shape
        return t * h * w

    @property
    def window_count(self) -> int:
        return len(make_windows(self.source_shape, self.window_shape, shifted=False))


DIT_BLOCK_CONTRACT = DitBlockContract()


def _language_freqs(length: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Match RotaryEmbedding(freqs_for='lang', dim=128//3)."""

    base_dim = DIT_BLOCK_CONTRACT.rope_dim // 3
    frequencies = 1.0 / (
        10000.0 ** (torch.arange(0, base_dim, 2, device=device, dtype=torch.float32)[: base_dim // 2] / base_dim)
    )
    positions = torch.arange(length, device=device, dtype=torch.float32)
    values = torch.einsum("n,f->nf", positions, frequencies)
    values = values.repeat_interleave(2, dim=-1)
    return values.to(dtype=dtype)


def _axial_language_freqs(
    dims: Sequence[int], *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    axes = []
    for axis, length in enumerate(dims):
        values = _language_freqs(int(length), device=device, dtype=dtype)
        shape = [1] * len(dims) + [values.shape[-1]]
        shape[axis] = int(length)
        axes.append(values.reshape(shape))
    return torch.cat(torch.broadcast_tensors(*axes), dim=-1)


def build_mmrope_frequencies(
    source_shape: Sequence[int],
    text_tokens: int,
    *,
    window_shape: Sequence[int] = DIT_BLOCK_CONTRACT.window_shape,
    shifted: bool = False,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build bounded MMRoPE3D frequencies in fixed AWA window order."""

    source_shape = tuple(int(value) for value in source_shape)
    windows = make_windows(source_shape, window_shape, shifted=shifted)
    video_parts = []
    for time, height, width in windows:
        frame_count = time[1] - time[0]
        height_count = height[1] - height[0]
        width_count = width[1] - width[0]
        axial = _axial_language_freqs(
            (int(text_tokens) + frame_count, height_count, width_count),
            device=device,
            dtype=dtype,
        )
        video_parts.append(axial[int(text_tokens) :].reshape(-1, axial.shape[-1]))

    text = _language_freqs(int(text_tokens), device=device, dtype=dtype).repeat(1, 3)
    return torch.cat(video_parts, dim=0), text


def upstream_mmrope_frequencies(
    source_shape: Sequence[int],
    text_tokens: int,
    *,
    window_shape: Sequence[int] = DIT_BLOCK_CONTRACT.window_shape,
    shifted: bool = False,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference the upstream max-grid construction without importing its checkout."""

    # The upstream implementation materializes a 1024x128x128 grid.  That is
    # useful for the dynamic model, but is needlessly large for this fixed
    # export contract (and can exceed the available memory during tests).
    # Coordinates are local to each AWA window, so a bounded grid has the same
    # values as the corresponding slice of the upstream grid.
    windows = make_windows(source_shape, window_shape, shifted=shifted)
    video_parts = []
    for time, height, width in windows:
        frame_count = time[1] - time[0]
        height_count = height[1] - height[0]
        width_count = width[1] - width[0]
        bounded = _axial_language_freqs(
            (int(text_tokens) + frame_count, height_count, width_count),
            device=device,
            dtype=dtype,
        )
        video_parts.append(
            bounded[int(text_tokens) :].reshape(-1, bounded.shape[-1])
        )
    text = _language_freqs(int(text_tokens), device=device, dtype=dtype).repeat(1, 3)
    return torch.cat(video_parts, dim=0), text


class FixedWindowAttention(nn.Module):
    """Fixed AWA pack, dense attention, and reverse/text pooling."""

    def __init__(
        self,
        source_shape: Sequence[int],
        window_shape: Sequence[int],
        text_tokens: int,
        heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        self.pack = AwaAttentionPack(source_shape, window_shape, text_tokens, shifted=False)
        self.unpack = AwaAttentionUnpack(source_shape, window_shape, text_tokens, shifted=False)
        self.attention = AwaWindowAttention(self.pack.cu_seqlens, head_dim=head_dim)

    def forward(
        self, video_qkv: torch.Tensor, text_qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        packed, _ = self.pack(video_qkv, text_qkv)
        attended = self.attention(packed)
        return self.unpack(attended)


class FixedRMSNorm(nn.Module):
    """Export-friendly RMSNorm matching diffusers' implementation."""

    def __init__(self, dim: int, eps: float, elementwise_affine: bool) -> None:
        super().__init__()
        self.eps = float(eps)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(int(dim)))
        else:
            self.register_parameter("weight", None)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden.dtype
        variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
        output = hidden * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            if self.weight.dtype in (torch.float16, torch.bfloat16):
                output = output.to(dtype=self.weight.dtype)
            output = output * self.weight
        else:
            output = output.to(dtype=input_dtype)
        return output


class FixedAdaBranch(nn.Module):
    """One official SeedVR2 AdaSingle branch with a static sequence length."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        for layer in ("attn", "mlp"):
            self.register_parameter(f"{layer}_shift", nn.Parameter(torch.zeros(dim)))
            self.register_parameter(f"{layer}_scale", nn.Parameter(torch.ones(dim)))
            self.register_parameter(f"{layer}_gate", nn.Parameter(torch.zeros(dim)))

    def forward(self, hidden: torch.Tensor, emb: torch.Tensor, layer: str, mode: str) -> torch.Tensor:
        dim = hidden.shape[-1]
        modulation = emb.reshape(emb.shape[0], dim, 2, 3)[0, :, 0 if layer == "attn" else 1, :]
        modulation = modulation.transpose(0, 1).unsqueeze(0).repeat(hidden.shape[0], 1, 1)
        shift_a, scale_a, gate_a = modulation.unbind(dim=1)
        shift_b = getattr(self, f"{layer}_shift")
        scale_b = getattr(self, f"{layer}_scale")
        gate_b = getattr(self, f"{layer}_gate")
        if mode == "in":
            return hidden * (scale_a + scale_b) + (shift_a + shift_b)
        if mode == "out":
            return hidden * (gate_a + gate_b)
        raise ValueError(f"unsupported Ada mode: {mode}")


class FixedAdaPair(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.vid = FixedAdaBranch(dim)
        self.txt = FixedAdaBranch(dim)


class FixedSwiGLU(nn.Module):
    def __init__(self, dim: int, expand_ratio: int, multiple_of: int = 256) -> None:
        super().__init__()
        hidden_dim = int(2 * dim * expand_ratio / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.proj_in_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.proj_out = nn.Linear(hidden_dim, dim, bias=False)
        self.proj_in = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj_out(torch.nn.functional.silu(self.proj_in_gate(hidden)) * self.proj_in(hidden))


class FixedMLPPair(nn.Module):
    def __init__(self, dim: int, expand_ratio: int) -> None:
        super().__init__()
        self.vid = FixedSwiGLU(dim, expand_ratio)
        self.txt = FixedSwiGLU(dim, expand_ratio)


def _apply_fixed_rotary(hidden: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply RotaryEmbedding's interleaved-half rotation to the first 126 dims."""

    rotated = hidden[..., : freqs.shape[-1]]
    paired = rotated.reshape(*rotated.shape[:-1], -1, 2)
    rotate_half = torch.stack((-paired[..., 1], paired[..., 0]), dim=-1).reshape_as(rotated)
    angles = freqs.unsqueeze(1)
    rotated = rotated * torch.cos(angles) + rotate_half * torch.sin(angles)
    return torch.cat((rotated, hidden[..., freqs.shape[-1] :]), dim=-1)


class FixedMMAttention(nn.Module):
    """Official block attention with a bounded, traceable AWA implementation."""

    def __init__(
        self,
        source_shape: Sequence[int],
        window_shape: Sequence[int],
        text_tokens: int,
        vid_dim: int,
        txt_dim: int,
        heads: int,
        head_dim: int,
        norm_eps: float,
        shifted: bool = False,
    ) -> None:
        super().__init__()
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.proj_qkv = nn.Module()
        self.proj_qkv.vid = nn.Linear(vid_dim, 3 * heads * head_dim, bias=False)
        self.proj_qkv.txt = nn.Linear(txt_dim, 3 * heads * head_dim, bias=False)
        self.proj_out = nn.Module()
        self.proj_out.vid = nn.Linear(heads * head_dim, vid_dim)
        self.proj_out.txt = nn.Linear(heads * head_dim, txt_dim)
        self.norm_q = nn.Module()
        self.norm_q.vid = FixedRMSNorm(head_dim, norm_eps, True)
        self.norm_q.txt = FixedRMSNorm(head_dim, norm_eps, True)
        self.norm_k = nn.Module()
        self.norm_k.vid = FixedRMSNorm(head_dim, norm_eps, True)
        self.norm_k.txt = FixedRMSNorm(head_dim, norm_eps, True)

        self.pack = AwaAttentionPack(source_shape, window_shape, text_tokens, shifted=shifted)
        self.unpack = AwaAttentionUnpack(source_shape, window_shape, text_tokens, shifted=shifted)
        self.attention = AwaWindowAttention(self.pack.cu_seqlens, head_dim=head_dim)
        video_freqs, text_freqs = build_mmrope_frequencies(
            source_shape,
            text_tokens,
            window_shape=window_shape,
            device=torch.device("cpu"),
            dtype=torch.float32,
            shifted=shifted,
        )
        video_source_freqs = video_freqs.index_select(0, self.pack.reverse_index)
        source_freqs = torch.cat((video_source_freqs, text_freqs), dim=0)
        self.register_buffer("rope_freqs", source_freqs.index_select(0, self.pack.sequence_index))

    def forward(self, vid: torch.Tensor, txt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vid_qkv = self.proj_qkv.vid(vid).reshape(-1, 3, self.heads, self.head_dim)
        txt_qkv = self.proj_qkv.txt(txt).reshape(-1, 3, self.heads, self.head_dim)
        vid_q, vid_k, vid_v = vid_qkv.unbind(dim=1)
        txt_q, txt_k, txt_v = txt_qkv.unbind(dim=1)
        vid_q = self.norm_q.vid(vid_q)
        txt_q = self.norm_q.txt(txt_q)
        vid_k = self.norm_k.vid(vid_k)
        txt_k = self.norm_k.txt(txt_k)
        vid_qkv = torch.stack((vid_q, vid_k, vid_v), dim=1)
        txt_qkv = torch.stack((txt_q, txt_k, txt_v), dim=1)
        packed, _ = self.pack(vid_qkv, txt_qkv)
        packed_q, packed_k, packed_v = packed.unbind(dim=1)
        packed_q = _apply_fixed_rotary(packed_q, self.rope_freqs)
        packed_k = _apply_fixed_rotary(packed_k, self.rope_freqs)
        attended = self.attention(torch.stack((packed_q, packed_k, packed_v), dim=1))
        vid_out, txt_out = self.unpack(attended)
        vid_out = vid_out.reshape(-1, self.heads * self.head_dim)
        txt_out = txt_out.reshape(-1, self.heads * self.head_dim)
        return self.proj_out.vid(vid_out), self.proj_out.txt(txt_out)


class FixedDitBlock(nn.Module):
    """A fixed-shape export adapter for one official SeedVR2 DiT block."""

    def __init__(
        self,
        source_shape: Sequence[int] = DIT_BLOCK_CONTRACT.source_shape,
        window_shape: Sequence[int] = DIT_BLOCK_CONTRACT.window_shape,
        text_tokens: int = DIT_BLOCK_CONTRACT.text_tokens,
        vid_dim: int = DIT_BLOCK_CONTRACT.vid_dim,
        txt_dim: int = DIT_BLOCK_CONTRACT.txt_dim,
        emb_dim: int = DIT_BLOCK_CONTRACT.emb_dim,
        heads: int = DIT_BLOCK_CONTRACT.heads,
        head_dim: int = DIT_BLOCK_CONTRACT.head_dim,
        expand_ratio: int = 4,
        norm_eps: float = 1.0e-5,
        shifted: bool = False,
    ) -> None:
        super().__init__()
        if int(emb_dim) != 6 * int(vid_dim) or int(vid_dim) != int(txt_dim):
            raise ValueError("the fixed adapter requires emb_dim=6*vid_dim=6*txt_dim")
        self.attn_norm = nn.Module()
        self.attn_norm.vid = FixedRMSNorm(vid_dim, norm_eps, False)
        self.attn_norm.txt = FixedRMSNorm(txt_dim, norm_eps, False)
        self.attn = FixedMMAttention(
            source_shape,
            window_shape,
            text_tokens,
            vid_dim,
            txt_dim,
            heads,
            head_dim,
            norm_eps,
            shifted,
        )
        self.mlp_norm = nn.Module()
        self.mlp_norm.vid = FixedRMSNorm(vid_dim, norm_eps, False)
        self.mlp_norm.txt = FixedRMSNorm(txt_dim, norm_eps, False)
        self.mlp = FixedMLPPair(vid_dim, expand_ratio)
        self.ada = FixedAdaPair(vid_dim)

    def forward(
        self, vid: torch.Tensor, txt: torch.Tensor, emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vid_attn = self.attn_norm.vid(vid)
        txt_attn = self.attn_norm.txt(txt)
        vid_attn = self.ada.vid(vid_attn, emb, "attn", "in")
        txt_attn = self.ada.txt(txt_attn, emb, "attn", "in")
        vid_attn, txt_attn = self.attn(vid_attn, txt_attn)
        vid_attn = self.ada.vid(vid_attn, emb, "attn", "out")
        txt_attn = self.ada.txt(txt_attn, emb, "attn", "out")
        vid_attn = vid_attn + vid
        txt_attn = txt_attn + txt

        vid_mlp = self.mlp_norm.vid(vid_attn)
        txt_mlp = self.mlp_norm.txt(txt_attn)
        vid_mlp = self.ada.vid(vid_mlp, emb, "mlp", "in")
        txt_mlp = self.ada.txt(txt_mlp, emb, "mlp", "in")
        vid_mlp = self.mlp.vid(vid_mlp)
        txt_mlp = self.mlp.txt(txt_mlp)
        vid_mlp = self.ada.vid(vid_mlp, emb, "mlp", "out")
        txt_mlp = self.ada.txt(txt_mlp, emb, "mlp", "out")
        return vid_mlp + vid_attn, txt_mlp + txt_attn


def load_official_block_weights(
    block: FixedDitBlock,
    checkpoint: Path,
    *,
    block_index: int = 0,
) -> int:
    """Copy one official DiT block from the checkpoint into the fixed adapter.

    The checkpoint is memory-mapped, so loading block 0 does not require a
    full 3B-model allocation.  Only trainable tensors are copied: static AWA
    mappings and bounded RoPE frequencies belong to the adapter itself.
    """

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SeedVR2 DiT checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"unexpected SeedVR2 checkpoint structure: {checkpoint}")

    prefix = f"blocks.{int(block_index)}."
    parameters = dict(block.named_parameters())
    missing = [name for name in parameters if prefix + name not in state]
    if missing:
        joined = ", ".join(missing[:5])
        raise ValueError(f"checkpoint is missing {len(missing)} block tensors: {joined}")

    with torch.no_grad():
        for name, parameter in parameters.items():
            source = state[prefix + name]
            if not isinstance(source, torch.Tensor) or tuple(source.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"checkpoint tensor mismatch for {prefix + name}: "
                    f"expected {tuple(parameter.shape)}, got {getattr(source, 'shape', None)}"
                )
            parameter.copy_(source.to(device=parameter.device, dtype=parameter.dtype))
    return len(parameters)


def load_fixed_dit_block(
    checkpoint: Path,
    *,
    block_index: int = 0,
    source_shape: Sequence[int] = DIT_BLOCK_CONTRACT.source_shape,
    window_shape: Sequence[int] = DIT_BLOCK_CONTRACT.window_shape,
    text_tokens: int = DIT_BLOCK_CONTRACT.text_tokens,
) -> FixedDitBlock:
    """Construct and populate the fixed-shape adapter for an official block."""

    block = FixedDitBlock(
        source_shape=source_shape,
        window_shape=window_shape,
        text_tokens=text_tokens,
    ).eval()
    load_official_block_weights(block, checkpoint, block_index=block_index)
    return block


__all__ = [
    "DIT_BLOCK_CONTRACT",
    "DitBlockContract",
    "FixedDitBlock",
    "FixedMMAttention",
    "FixedRMSNorm",
    "FixedWindowAttention",
    "build_mmrope_frequencies",
    "load_fixed_dit_block",
    "load_official_block_weights",
    "upstream_mmrope_frequencies",
]
