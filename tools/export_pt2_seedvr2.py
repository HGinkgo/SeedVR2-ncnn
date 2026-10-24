# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 SeedVR2-ncnn contributors

"""Export small, representative SeedVR2 subgraphs to PT2.

The generated files are experiment artifacts.  The production application still
consumes the existing TorchScript/ncnn package.  This command deliberately keeps
the PT2 surface small so every graph can be compared with an eager reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

# Make ``python tools/export_pt2_seedvr2.py`` work from a source checkout.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.export_awa_attention import AwaAttentionGraph
from tools.reference.dit_block import (DIT_BLOCK_CONTRACT, DIT_LAST_BLOCK_CONTRACT, FixedDitBlock,
                                        FixedSwiGLU, build_mmrope_frequencies,
                                        load_fixed_dit_block)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class SeedVR2MLPProbe(nn.Module):
    """A reduced-size instance of the repository's SeedVR2 DiT SwiGLU.

    The module implementation is shared with ``FixedDitBlock``. Fixture callers
    seed PyTorch before construction; the optional checkpoint path injects the
    trained full-width block-0 MLP into this same wrapper.
    """

    def __init__(self, dim: int, *, mlp: FixedSwiGLU | None = None) -> None:
        super().__init__()
        self.mlp = mlp or FixedSwiGLU(dim, expand_ratio=4, multiple_of=dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.mlp(tokens)


class SeedVR2MLPResidualProbe(nn.Module):
    """The block-0 video MLP branch with its pre-norm and residual."""

    def __init__(self, dim: int, *, mlp: FixedSwiGLU | None = None) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps=1e-5, elementwise_affine=False)
        self.mlp = mlp or FixedSwiGLU(dim, expand_ratio=4, multiple_of=dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.mlp(self.norm(tokens))


class SeedVR2AdaProbe(nn.Module):
    """Video Ada modulation with materialized embedding-derived activations.

    The reshape/slice that turns the 15360-wide timestep embedding into
    per-token modulation is intentionally outside this first native boundary.
    This probe verifies the broadcast and checkpoint parameter contract used by
    the block, while keeping all tensors rank-2 for the restricted frontend.
    """

    def __init__(self, dim: int, *, shift: torch.Tensor | None = None,
                 scale: torch.Tensor | None = None) -> None:
        super().__init__()
        self.attn_shift = nn.Parameter(torch.zeros(dim) if shift is None else shift.detach().clone())
        self.attn_scale = nn.Parameter(torch.ones(dim) if scale is None else scale.detach().clone())

    def forward(
        self,
        tokens: torch.Tensor,
        modulation_shift: torch.Tensor,
        modulation_scale: torch.Tensor,
    ) -> torch.Tensor:
        scale = modulation_scale + self.attn_scale
        shift = modulation_shift + self.attn_shift
        return tokens * scale + shift


class SeedVR2AdaEmbeddingProbe(nn.Module):
    """Video attention Ada branch starting from the packed timestep embedding."""

    def __init__(self, dim: int, *, shift: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        self.dim = dim
        self.attn_shift = nn.Parameter(shift.detach().clone())
        self.attn_scale = nn.Parameter(scale.detach().clone())

    def forward(self, tokens: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        modulation = embedding.reshape(self.dim, 6)[:, :3]
        shift = modulation.select(1, 0)
        scale = modulation.select(1, 1)
        return tokens * (scale + self.attn_scale) + (shift + self.attn_shift)


class SeedVR2AdaMLPEmbeddingProbe(nn.Module):
    """Video MLP branch with fixed-layout packed Ada modulation.

    This starts at the hidden state after the attention residual.  The packed
    embedding layout is the official ``[batch, dim, 2, 3]`` layout flattened
    to ``[1, dim * 6]``; the second triplet contains the MLP shift, scale and
    gate values.
    """

    def __init__(
        self,
        dim: int,
        *,
        mlp: FixedSwiGLU,
        shift: torch.Tensor,
        scale: torch.Tensor,
        gate: torch.Tensor,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.norm = nn.RMSNorm(dim, eps=1e-5, elementwise_affine=False)
        self.mlp = mlp
        self.mlp_shift = nn.Parameter(shift.detach().clone())
        self.mlp_scale = nn.Parameter(scale.detach().clone())
        self.mlp_gate = nn.Parameter(gate.detach().clone())

    def forward(self, tokens: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        modulation = embedding.reshape(self.dim, 6)[:, 3:6]
        shift = modulation.select(1, 0)
        scale = modulation.select(1, 1)
        gate = modulation.select(1, 2)
        hidden = self.norm(tokens)
        hidden = hidden * (scale + self.mlp_scale) + (shift + self.mlp_shift)
        hidden = self.mlp(hidden)
        return tokens + hidden * (gate + self.mlp_gate)


class SeedVR2AdaAttentionGateEmbeddingProbe(nn.Module):
    """The block-0 video attention output gate and residual boundary."""

    def __init__(self, dim: int, *, gate: torch.Tensor) -> None:
        super().__init__()
        self.dim = int(dim)
        self.attn_gate = nn.Parameter(gate.detach().clone())

    def forward(
        self,
        attention_output: torch.Tensor,
        residual: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        modulation = embedding.reshape(self.dim, 6)[:, :3]
        gate = modulation.select(1, 2)
        return residual + attention_output * (gate + self.attn_gate)


class SeedVR2AttentionCoreProbe(nn.Module):
    """A static video self-attention core, before AWA/text integration."""

    def __init__(self, tokens: int = 7, dim: int = 16, heads: int = 2,
                 frequencies: torch.Tensor | None = None) -> None:
        super().__init__()
        if dim <= 0 or heads <= 0 or dim % heads:
            raise ValueError("attention core requires positive dimensions divisible by heads")
        self.tokens = int(tokens)
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        if self.head_dim % 2:
            raise ValueError("attention core rotary head dimension must be even")
        self.proj_qkv = nn.Linear(self.dim, 3 * self.dim, bias=False)
        self.norm_q = nn.RMSNorm(self.head_dim, eps=1e-5)
        self.norm_k = nn.RMSNorm(self.head_dim, eps=1e-5)
        self.proj_out = nn.Linear(self.dim, self.dim)

        positions = torch.arange(self.tokens, dtype=torch.float32)
        inverse_frequencies = 1.0 / (
            10000.0 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        angles = (positions[:, None] * inverse_frequencies[None, :]
                  if frequencies is None else frequencies[:, ::2])
        if angles.shape[0] != self.tokens or angles.shape[1] * 2 > self.head_dim:
            raise ValueError("rotary frequencies do not match the attention core")
        cosine = torch.cos(angles)
        sine = torch.sin(angles)
        rotary = torch.eye(self.head_dim).repeat(self.tokens, 1, 1)
        even = torch.arange(0, angles.shape[1] * 2, 2)
        odd = even + 1
        rotary[:, even, even] = cosine
        rotary[:, even, odd] = sine
        rotary[:, odd, even] = -sine
        rotary[:, odd, odd] = cosine
        self.register_buffer("rotary_matrix", rotary)

    def _apply_rope(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.matmul(hidden, self.rotary_matrix)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        qkv = self.proj_qkv(tokens).reshape(
            self.tokens, 3, self.heads, self.head_dim
        )
        query = self._apply_rope(self.norm_q(qkv.select(1, 0))).permute(1, 0, 2)
        key = self._apply_rope(self.norm_k(qkv.select(1, 1))).permute(1, 0, 2)
        value = qkv.select(1, 2).permute(1, 0, 2)
        attended = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        attended = attended.permute(1, 0, 2).reshape(self.tokens, self.dim)
        return self.proj_out(attended)


class SeedVR2AdaAttentionProbe(nn.Module):
    """Connect pre-norm/Ada input, video-only attention, and gated residual."""

    def __init__(self, attention: SeedVR2AttentionCoreProbe, *,
                 shift: torch.Tensor, scale: torch.Tensor, gate: torch.Tensor) -> None:
        super().__init__()
        self.attention = attention
        self.dim = attention.dim
        self.norm = nn.RMSNorm(self.dim, eps=1e-5, elementwise_affine=False)
        self.attn_shift = nn.Parameter(shift.detach().clone())
        self.attn_scale = nn.Parameter(scale.detach().clone())
        self.attn_gate = nn.Parameter(gate.detach().clone())

    def forward(self, tokens: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        modulation = embedding.reshape(self.dim, 6)[:, :3]
        shift = modulation.select(1, 0)
        scale = modulation.select(1, 1)
        gate = modulation.select(1, 2)
        hidden = self.norm(tokens) * (scale + self.attn_scale) + (shift + self.attn_shift)
        return tokens + self.attention(hidden) * (gate + self.attn_gate)


def load_checkpoint_video_attention_core(
    checkpoint: Path, tokens: int = 7,
) -> tuple[SeedVR2AttentionCoreProbe, str]:
    """Load block-0 video QKV/QK-norm/output weights and fixed MMRoPE3D."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    frequencies, _ = build_mmrope_frequencies(
        (1, 1, tokens), 58, window_shape=(1, 1, tokens),
        device=torch.device("cpu"), dtype=torch.float32)
    model = SeedVR2AttentionCoreProbe(tokens, 2560, 20, frequencies).eval()
    for name, parameter in model.named_parameters():
        layer, field = name.rsplit(".", 1)
        target = f"blocks.0.attn.{layer}.vid.{field}"
        value = state.get(target)
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                or value.shape != parameter.shape):
            raise ValueError(f"checkpoint attention tensor shape/dtype mismatch: {target}")
        with torch.no_grad():
            parameter.copy_(value)
    del state
    return model, _sha256(checkpoint)


def load_checkpoint_video_ada_attention(
    checkpoint: Path, tokens: int = 7,
) -> tuple[SeedVR2AdaAttentionProbe, str]:
    attention, digest = load_checkpoint_video_attention_core(checkpoint, tokens)
    state = torch.load(Path(checkpoint).expanduser().resolve(), map_location="cpu",
                       mmap=True, weights_only=True)
    tensors = {}
    for name in ("shift", "scale", "gate"):
        target = f"blocks.0.ada.vid.attn_{name}"
        value = state.get(target)
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                or tuple(value.shape) != (2560,)):
            raise ValueError(f"checkpoint tensor must be float32 with shape (2560,): {target}")
        tensors[name] = value
    return SeedVR2AdaAttentionProbe(attention, **tensors).eval(), digest


def _load_checkpoint_video_mlp(checkpoint: Path) -> tuple[FixedSwiGLU, str]:
    """Load the trained video SwiGLU weights from DiT block 0."""

    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SeedVR2 checkpoint not found: {checkpoint}")
    mlp = FixedSwiGLU(dim=2560, expand_ratio=4).eval()
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"unexpected SeedVR2 checkpoint structure: {checkpoint}")
    prefix = "blocks.0.mlp.vid."
    parameters = dict(mlp.named_parameters())
    missing = [prefix + name for name in parameters if prefix + name not in state]
    if missing:
        raise ValueError("checkpoint is missing video MLP tensors: " + ", ".join(missing))
    with torch.no_grad():
        for name, parameter in parameters.items():
            source = state[prefix + name]
            if not isinstance(source, torch.Tensor) or source.shape != parameter.shape:
                raise ValueError(
                    f"checkpoint tensor mismatch for {prefix + name}: expected {tuple(parameter.shape)}, "
                    f"got {getattr(source, 'shape', None)}"
                )
            if source.dtype != torch.float32:
                raise ValueError(
                    f"checkpoint tensor must be float32 for the FP32 export, got {source.dtype}: {prefix + name}"
                )
            parameter.copy_(source)
    del state
    return mlp, _sha256(checkpoint)


def load_checkpoint_video_mlp(checkpoint: Path) -> tuple[SeedVR2MLPProbe, str]:
    mlp, digest = _load_checkpoint_video_mlp(checkpoint)
    return SeedVR2MLPProbe(2560, mlp=mlp).eval(), digest


def load_checkpoint_video_mlp_residual(checkpoint: Path) -> tuple[SeedVR2MLPResidualProbe, str]:
    mlp, digest = _load_checkpoint_video_mlp(checkpoint)
    return SeedVR2MLPResidualProbe(2560, mlp=mlp).eval(), digest


def load_checkpoint_video_ada(checkpoint: Path) -> tuple[SeedVR2AdaProbe, str]:
    """Load block-0 video attention Ada shift/scale parameters."""

    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SeedVR2 checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"unexpected SeedVR2 checkpoint structure: {checkpoint}")
    prefix = "blocks.0.ada.vid."
    shift = state.get(prefix + "attn_shift")
    scale = state.get(prefix + "attn_scale")
    if not isinstance(shift, torch.Tensor) or not isinstance(scale, torch.Tensor):
        raise ValueError(f"checkpoint is missing video Ada tensors under {prefix}")
    if shift.dtype != torch.float32 or scale.dtype != torch.float32 or shift.ndim != 1 or scale.shape != shift.shape:
        raise ValueError("checkpoint Ada shift/scale must be matching rank-1 float32 tensors")
    model = SeedVR2AdaProbe(int(shift.shape[0]), shift=shift, scale=scale).eval()
    del state
    return model, _sha256(checkpoint)


def load_checkpoint_video_ada_mlp(checkpoint: Path) -> tuple[SeedVR2AdaMLPEmbeddingProbe, str]:
    """Load block-0 video MLP weights and its Ada shift/scale/gate tensors."""

    mlp, checkpoint_sha256 = _load_checkpoint_video_mlp(checkpoint)
    checkpoint = Path(checkpoint).expanduser().resolve()
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"unexpected SeedVR2 checkpoint structure: {checkpoint}")
    prefix = "blocks.0.ada.vid."
    tensors = {
        name: state.get(prefix + "mlp_" + name)
        for name in ("shift", "scale", "gate")
    }
    if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
        raise ValueError(f"checkpoint is missing video MLP Ada tensors under {prefix}")
    shift, scale, gate = (tensors[name] for name in ("shift", "scale", "gate"))
    if any(value.dtype != torch.float32 or value.ndim != 1 for value in (shift, scale, gate)):
        raise ValueError("checkpoint MLP Ada shift/scale/gate must be rank-1 float32 tensors")
    if not (shift.shape == scale.shape == gate.shape == (2560,)):
        raise ValueError(
            "checkpoint MLP Ada shift/scale/gate must have shape (2560,), got "
            f"{tuple(shift.shape)}, {tuple(scale.shape)}, {tuple(gate.shape)}"
        )
    model = SeedVR2AdaMLPEmbeddingProbe(
        2560, mlp=mlp, shift=shift, scale=scale, gate=gate
    ).eval()
    del state
    return model, checkpoint_sha256


def load_checkpoint_video_ada_attention_gate(
    checkpoint: Path,
) -> tuple[SeedVR2AdaAttentionGateEmbeddingProbe, str]:
    """Load the trained block-0 video attention output gate."""

    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SeedVR2 checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"unexpected SeedVR2 checkpoint structure: {checkpoint}")
    target = "blocks.0.ada.vid.attn_gate"
    gate = state.get(target)
    if not isinstance(gate, torch.Tensor) or gate.dtype != torch.float32 or tuple(gate.shape) != (2560,):
        raise ValueError(f"checkpoint tensor must be float32 with shape (2560,): {target}")
    model = SeedVR2AdaAttentionGateEmbeddingProbe(2560, gate=gate).eval()
    del state
    return model, _sha256(checkpoint)


def _outputs(value: object) -> tuple[torch.Tensor, ...]:
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)  # type: ignore[arg-type]


def _positive_shape(value: str, count: int) -> tuple[int, ...]:
    parts = tuple(int(item) for item in value.split(","))
    if len(parts) != count or any(item <= 0 for item in parts):
        raise argparse.ArgumentTypeError(
            f"expected {count} positive comma-separated integers"
        )
    return parts


def _make_export(
    args: argparse.Namespace,
) -> tuple[nn.Module, tuple[torch.Tensor, ...], dict[str, object], dict[str, object]]:
    if args.model == "awa":
        if args.dynamic:
            raise SystemExit("--dynamic is only supported by projection and checkpoint-mlp fixtures")
        if min(args.text_tokens, args.heads, args.head_dim) <= 0:
            raise SystemExit("--text-tokens, --heads and --head-dim must be positive")
        size = _positive_shape(args.size, 3)
        windows = _positive_shape(args.windows, 3)
        model = AwaAttentionGraph(
            size,
            windows,
            args.text_tokens,
            args.heads,
            args.head_dim,
            args.shifted,
        ).eval()
        inputs = (
            torch.randn(math.prod(size), 3, args.heads, args.head_dim),
            torch.randn(args.text_tokens, 3, args.heads, args.head_dim),
        )
        metadata = {
            "model": "awa_attention",
            "dynamic": False,
            "input_shapes": [list(item.shape) for item in inputs],
            "output_shapes": [list(item.shape) for item in _outputs(model(*inputs))],
            "size": list(size),
            "windows": list(windows),
            "text_tokens": args.text_tokens,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "shifted": args.shifted,
        }
        dynamic_shapes = None
    elif args.model == "projection":
        dim = int(args.dim)
        if dim <= 0:
            raise SystemExit("--dim must be positive")
        model = SeedVR2MLPProbe(dim).eval()
        token_count = int(args.tokens)
        if token_count <= 0:
            raise SystemExit("--tokens must be positive")
        if args.dynamic and token_count > args.max_tokens:
            raise SystemExit("--tokens must not exceed --max-tokens for a dynamic export")
        inputs = (torch.randn(token_count, dim),)
        metadata = {
            "model": "dit_mlp_probe",
            "dynamic": bool(args.dynamic),
            "input_shapes": [list(item.shape) for item in inputs],
            "output_shapes": [list(item.shape) for item in _outputs(model(*inputs))],
            "dim": dim,
            "tokens": token_count,
            "max_tokens": args.max_tokens if args.dynamic else None,
        }
        dynamic_shapes = (
            {"tokens": {0: torch.export.Dim("tokens", min=1, max=args.max_tokens)}}
            if args.dynamic
            else None
        )
    elif args.model == "checkpoint-ada":
        if not args.dynamic:
            raise SystemExit("checkpoint fixtures require --dynamic to verify their token-axis contract")
        if args.dim != 8:
            raise SystemExit("--dim applies only to the projection fixture")
        token_count = int(args.tokens)
        if token_count <= 0 or token_count > args.max_tokens:
            raise SystemExit("--tokens must be positive and not exceed --max-tokens")
        model, checkpoint_sha256 = load_checkpoint_video_ada(args.checkpoint)
        inputs = (torch.randn(token_count, 2560), torch.randn(token_count, 2560), torch.randn(token_count, 2560))
        metadata = {
            "model": "dit_block0_video_ada_attn",
            "dynamic": True,
            "input_shapes": [list(item.shape) for item in inputs],
            "output_shapes": [list(item.shape) for item in _outputs(model(*inputs))],
            "dim": 2560,
            "tokens": token_count,
            "max_tokens": args.max_tokens,
            "checkpoint_file": Path(args.checkpoint).name,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_parameter_prefix": "blocks.0.ada.vid.",
            "checkpoint_tensor_count": len(dict(model.named_parameters())),
            "checkpoint_parameter_elements": sum(parameter.numel() for parameter in model.parameters()),
            "modulation_boundary": "materialized rank-2 shift/scale tensors",
        }
        token_dim = torch.export.Dim("tokens", min=1, max=args.max_tokens)
        dynamic_shapes = ({0: token_dim}, {0: token_dim}, {0: token_dim})
    elif args.model == "checkpoint-ada-embedding":
        if not args.dynamic:
            raise SystemExit("checkpoint fixtures require --dynamic to verify their token-axis contract")
        if args.dim != 8:
            raise SystemExit("--dim applies only to the projection fixture")
        token_count = int(args.tokens)
        if token_count <= 0 or token_count > args.max_tokens:
            raise SystemExit("--tokens must be positive and not exceed --max-tokens")
        ada, checkpoint_sha256 = load_checkpoint_video_ada(args.checkpoint)
        model = SeedVR2AdaEmbeddingProbe(2560, shift=ada.attn_shift, scale=ada.attn_scale).eval()
        inputs = (torch.randn(token_count, 2560), torch.randn(1, 2560 * 6))
        metadata = {
            "model": "dit_block0_video_ada_attn_embedding",
            "dynamic": True,
            "input_shapes": [list(item.shape) for item in inputs],
            "output_shapes": [list(item.shape) for item in _outputs(model(*inputs))],
            "dim": 2560,
            "tokens": token_count,
            "max_tokens": args.max_tokens,
            "checkpoint_file": Path(args.checkpoint).name,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_parameter_prefix": "blocks.0.ada.vid.",
            "checkpoint_tensor_count": len(dict(model.named_parameters())),
            "checkpoint_parameter_elements": sum(parameter.numel() for parameter in model.parameters()),
            "modulation_boundary": "packed [1,dim*6] embedding selected by fixed reshape/slice/select",
        }
        token_dim = torch.export.Dim("tokens", min=1, max=args.max_tokens)
        dynamic_shapes = ({0: token_dim}, {})
    elif args.model == "checkpoint-ada-mlp-embedding":
        if not args.dynamic:
            raise SystemExit("checkpoint fixtures require --dynamic to verify their token-axis contract")
        if args.dim != 8:
            raise SystemExit("--dim applies only to the projection fixture")
        token_count = int(args.tokens)
        if token_count <= 0 or token_count > args.max_tokens:
            raise SystemExit("--tokens must be positive and not exceed --max-tokens")
        model, checkpoint_sha256 = load_checkpoint_video_ada_mlp(args.checkpoint)
        inputs = (torch.randn(token_count, 2560), torch.randn(1, 2560 * 6))
        metadata = {
            "model": "dit_block0_video_ada_mlp_embedding",
            "dynamic": True,
            "input_shapes": [list(item.shape) for item in inputs],
            "output_shapes": [list(item.shape) for item in _outputs(model(*inputs))],
            "dim": 2560,
            "tokens": token_count,
            "max_tokens": args.max_tokens,
            "checkpoint_file": Path(args.checkpoint).name,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_parameter_prefix": "blocks.0.mlp.vid. + blocks.0.ada.vid.mlp_*",
            "checkpoint_tensor_count": len(dict(model.named_parameters())),
            "checkpoint_parameter_elements": sum(parameter.numel() for parameter in model.parameters()),
            "modulation_boundary": "packed [1,dim*6] embedding selected by fixed reshape/slice/select",
        }
        token_dim = torch.export.Dim("tokens", min=1, max=args.max_tokens)
        dynamic_shapes = ({0: token_dim}, {})
    elif args.model == "checkpoint-ada-attn-gate-embedding":
        if not args.dynamic:
            raise SystemExit("checkpoint fixtures require --dynamic to verify their token-axis contract")
        if args.dim != 8:
            raise SystemExit("--dim applies only to the projection fixture")
        token_count = int(args.tokens)
        if token_count <= 0 or token_count > args.max_tokens:
            raise SystemExit("--tokens must be positive and not exceed --max-tokens")
        model, checkpoint_sha256 = load_checkpoint_video_ada_attention_gate(args.checkpoint)
        inputs = (
            torch.randn(token_count, 2560),
            torch.randn(token_count, 2560),
            torch.randn(1, 2560 * 6),
        )
        metadata = {
            "model": "dit_block0_video_ada_attn_gate_embedding",
            "dynamic": True,
            "input_shapes": [list(item.shape) for item in inputs],
            "output_shapes": [list(item.shape) for item in _outputs(model(*inputs))],
            "dim": 2560,
            "tokens": token_count,
            "max_tokens": args.max_tokens,
            "checkpoint_file": Path(args.checkpoint).name,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_parameter_prefix": "blocks.0.ada.vid.attn_gate",
            "checkpoint_tensor_count": len(dict(model.named_parameters())),
            "checkpoint_parameter_elements": sum(parameter.numel() for parameter in model.parameters()),
            "modulation_boundary": "packed [1,15360] embedding gate with attention residual",
        }
        token_dim = torch.export.Dim("tokens", min=1, max=args.max_tokens)
        dynamic_shapes = ({0: token_dim}, {0: token_dim}, {})
    elif args.model in ("attention-core", "checkpoint-attention-core", "checkpoint-ada-attention"):
        token_count = int(args.tokens)
        if token_count <= 0:
            raise SystemExit("--tokens must be positive")
        if args.dynamic:
            raise SystemExit("attention core requires static token shapes; dynamic attention is unverified")
        if args.model == "checkpoint-ada-attention":
            model, checkpoint_sha256 = load_checkpoint_video_ada_attention(args.checkpoint, token_count)
        elif args.model == "checkpoint-attention-core":
            model, checkpoint_sha256 = load_checkpoint_video_attention_core(args.checkpoint, token_count)
        else:
            model = SeedVR2AttentionCoreProbe(tokens=token_count).eval()
        inputs = (torch.randn(token_count, model.dim),)
        if args.model == "checkpoint-ada-attention":
            inputs += (torch.randn(1, model.dim * 6),)
        core = model.attention if args.model == "checkpoint-ada-attention" else model
        metadata = {
            "model": args.model,
            "dynamic": False,
            "input_shapes": [list(item.shape) for item in inputs],
            "output_shapes": [list(item.shape) for item in _outputs(model(*inputs))],
            "dim": model.dim,
            "heads": core.heads,
            "head_dim": core.head_dim,
            "tokens": token_count,
            "rope": ("fixed MMRoPE3D, single video window (1,1,tokens), 58 text-position offset,"
                     " first 126 dimensions rotated and last 2 unchanged"
                     if args.model != "attention-core" else "fixed standard 1D interleaved RoPE"),
            "attention": "video-only non-causal self-attention without mask/dropout; no text K/V or AWA packing",
        }
        if args.model != "attention-core":
            metadata.update(checkpoint_file=Path(args.checkpoint).name,
                            checkpoint_sha256=checkpoint_sha256,
                            checkpoint_parameter_prefix=("blocks.0.attn.{proj_qkv,norm_q,norm_k,proj_out}.vid"
                                + (" + blocks.0.ada.vid.attn_*" if args.model == "checkpoint-ada-attention" else "")),
                            checkpoint_tensor_count=len(dict(model.named_parameters())),
                            checkpoint_parameter_elements=sum(p.numel() for p in model.parameters()))
        dynamic_shapes = None
    elif args.model == "checkpoint-dit-block":
        if args.dynamic:
            raise SystemExit("checkpoint-dit-block is fixed-shape until custom AWA operators have PT2 support")
        if args.block_index < 0:
            raise SystemExit("--block-index must be non-negative")
        block = load_fixed_dit_block(args.checkpoint, block_index=args.block_index)
        checkpoint_sha256 = _sha256(args.checkpoint)
        model = block.eval()
        contract = DIT_BLOCK_CONTRACT
        inputs = (
            torch.randn(contract.video_tokens, contract.vid_dim),
            torch.randn(contract.text_tokens, contract.txt_dim),
            torch.randn(1, contract.emb_dim),
        )
        metadata = {
            "model": f"dit_block{args.block_index}_full",
            "block_index": args.block_index,
            "dynamic": False,
            "input_shapes": [list(item.shape) for item in inputs],
            "output_shapes": [list(item.shape) for item in _outputs(model(*inputs))],
            "source_shape": list(contract.source_shape),
            "video_tokens": contract.video_tokens,
            "text_tokens": contract.text_tokens,
            "dim": contract.vid_dim,
            "embedding_dim": contract.emb_dim,
            "heads": contract.heads,
            "head_dim": contract.head_dim,
            "checkpoint_file": Path(args.checkpoint).name,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_parameter_prefix": f"blocks.{args.block_index}.",
            "checkpoint_tensor_count": len(dict(model.named_parameters())),
            "checkpoint_parameter_elements": sum(parameter.numel() for parameter in model.parameters()),
            "conversion_status": "exported_only; restricted PNNX frontend does not yet lower custom AWA operators",
        }
        if args.block_index == 31:
            metadata["last_block_contract"] = DIT_LAST_BLOCK_CONTRACT
        dynamic_shapes = None
    else:
        if not args.dynamic:
            raise SystemExit("checkpoint fixtures require --dynamic to verify their token-axis contract")
        if args.dim != 8:
            raise SystemExit("--dim applies only to the projection fixture")
        token_count = int(args.tokens)
        if token_count <= 0 or token_count > args.max_tokens:
            raise SystemExit("--tokens must be positive and not exceed --max-tokens")
        if args.model == "checkpoint-mlp":
            model, checkpoint_sha256 = load_checkpoint_video_mlp(args.checkpoint)
            model_name = "dit_block0_video_mlp"
        else:
            model, checkpoint_sha256 = load_checkpoint_video_mlp_residual(args.checkpoint)
            model_name = "dit_block0_video_mlp_residual"
        inputs = (torch.randn(token_count, 2560),)
        metadata = {
            "model": model_name,
            "dynamic": True,
            "input_shapes": [list(item.shape) for item in inputs],
            "output_shapes": [list(item.shape) for item in _outputs(model(*inputs))],
            "dim": 2560,
            "tokens": token_count,
            "max_tokens": args.max_tokens,
            "checkpoint_file": Path(args.checkpoint).name,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_parameter_prefix": "blocks.0.mlp.vid.",
            "checkpoint_tensor_count": len(dict(model.named_parameters())),
            "checkpoint_parameter_elements": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        }
        dynamic_shapes = {"tokens": {0: torch.export.Dim("tokens", min=1, max=args.max_tokens)}}
    return model, inputs, metadata, {"dynamic_shapes": dynamic_shapes}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("awa", "projection", "checkpoint-mlp", "checkpoint-mlp-residual", "checkpoint-ada", "checkpoint-ada-embedding", "checkpoint-ada-mlp-embedding", "checkpoint-ada-attn-gate-embedding", "attention-core", "checkpoint-attention-core", "checkpoint-ada-attention", "checkpoint-dit-block"), default="awa")
    parser.add_argument("--format", choices=("pt2", "torchscript", "both"), default="pt2")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--size", default="2,19,23", metavar="T,H,W")
    parser.add_argument("--windows", default="4,3,3", metavar="T,H,W")
    parser.add_argument("--text-tokens", type=int, default=5)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=3)
    parser.add_argument("--shifted", action="store_true")
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=7)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--block-index", type=int, default=0,
                        help="checkpoint DiT block index for --model checkpoint-dit-block")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("ckpts/seedvr2_ema_3b.pth"),
        help="SeedVR2 checkpoint used by checkpoint fixtures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive")
    torch.manual_seed(args.seed)
    model, inputs, metadata, export_options = _make_export(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = {
        "awa": "awa_attention",
        "projection": "dit_mlp_probe",
        "checkpoint-mlp": "dit_block0_video_mlp",
        "checkpoint-mlp-residual": "dit_block0_video_mlp_residual",
        "checkpoint-ada": "dit_block0_video_ada_attn",
        "checkpoint-ada-embedding": "dit_block0_video_ada_attn_embedding",
        "checkpoint-ada-mlp-embedding": "dit_block0_video_ada_mlp_embedding",
        "checkpoint-ada-attn-gate-embedding": "dit_block0_video_ada_attn_gate_embedding",
        "attention-core": "seedvr2_fixed_attention_core",
        "checkpoint-attention-core": "dit_block0_video_attention_core",
        "checkpoint-ada-attention": "dit_block0_video_ada_attention",
        "checkpoint-dit-block": f"dit_block{args.block_index}_full",
    }[args.model]

    with torch.no_grad():
        expected = model(*inputs)
    reference = {"inputs": inputs, "outputs": expected}
    if args.model not in ("checkpoint-mlp", "checkpoint-mlp-residual", "checkpoint-ada-mlp-embedding"):
        reference["state_dict"] = model.state_dict()
    torch.save(reference, args.output_dir / f"{stem}_reference.pt")

    exported = None
    if args.format in ("pt2", "both"):
        export_kwargs = export_options["dynamic_shapes"]
        if export_kwargs is None:
            exported = torch.export.export(model, inputs)
        else:
            exported = torch.export.export(model, inputs, dynamic_shapes=export_kwargs)
        torch.export.save(exported, args.output_dir / f"{stem}.pt2")

    if args.format in ("torchscript", "both"):
        traced = torch.jit.trace(model, inputs, check_trace=True)
        traced.save(str(args.output_dir / f"{stem}.pt"))

    metadata.update(
        {
            "exporter": "seedvr2-pt2-fixtures-v2",
            "pt2_sha256": hashlib.sha256((args.output_dir / f"{stem}.pt2").read_bytes()).hexdigest()
            if args.format in ("pt2", "both") else None,
            "torch_version": torch.__version__,
            "seed": args.seed,
            "format": args.format,
            "pt2_file": f"{stem}.pt2" if args.format in ("pt2", "both") else None,
            "torchscript_file": f"{stem}.pt" if args.format in ("torchscript", "both") else None,
            "reference_file": f"{stem}_reference.pt",
        }
    )
    if exported is not None:
        def signature_spec(spec: object) -> dict[str, object]:
            argument = getattr(spec, "arg")
            return {
                "name": getattr(argument, "name", None),
                "kind": getattr(getattr(spec, "kind", None), "name", str(getattr(spec, "kind", None))),
                "target": str(getattr(spec, "target", None)) if getattr(spec, "target", None) is not None else None,
            }

        metadata["graph_signature"] = {
            "inputs": [signature_spec(spec) for spec in exported.graph_signature.input_specs],
            "outputs": [signature_spec(spec) for spec in exported.graph_signature.output_specs],
        }
        metadata["state_dict"] = [
            {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in exported.state_dict.items()
        ]
        metadata["range_constraints"] = {
            str(symbol): {"min": str(constraint.lower), "max": str(constraint.upper)}
            for symbol, constraint in exported.range_constraints.items()
        }
        metadata["graph_nodes"] = [
            {"name": node.name, "op": str(node.op), "target": str(node.target)}
            for node in exported.graph.nodes
        ]
    (args.output_dir / f"{stem}.manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
