"""Export the fixed-shape AWA attention boundary for PNNX inspection."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from tools.reference.awa_attention import (
    AwaAttentionPack,
    AwaAttentionUnpack,
    AwaWindowAttention,
)


class AwaAttentionGraph(nn.Module):
    """Pack QKV, attend independently per window, and restore source layout."""

    def __init__(
        self,
        size: Sequence[int],
        num_windows: Sequence[int],
        text_tokens: int,
        heads: int,
        head_dim: int,
        shifted: bool,
    ) -> None:
        super().__init__()
        pack = AwaAttentionPack(size, num_windows, text_tokens, shifted=shifted)
        self.pack = pack
        self.attention = AwaWindowAttention(pack.cu_seqlens, head_dim=head_dim)
        self.unpack = AwaAttentionUnpack(size, num_windows, text_tokens, shifted=shifted)
        self.heads = int(heads)
        self.head_dim = int(head_dim)

    def forward(
        self, vid_qkv: torch.Tensor, txt_qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        packed, _ = self.pack(vid_qkv, txt_qkv)
        attended = self.attention(packed)
        return self.unpack(attended)


def _shape(value: str) -> tuple[int, int, int]:
    parts = tuple(int(part) for part in value.split(","))
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError("expected three positive comma-separated integers")
    return parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=_shape, default=(2, 19, 23), metavar="T,H,W")
    parser.add_argument("--windows", type=_shape, default=(4, 3, 3), metavar="T,H,W")
    parser.add_argument("--text-tokens", type=int, default=5)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=3)
    parser.add_argument("--shifted", action="store_true")
    parser.add_argument(
        "--pnnx",
        type=Path,
        help="optional pnnx executable; when set, convert the generated TorchScript",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.text_tokens <= 0 or args.heads <= 0 or args.head_dim <= 0:
        raise SystemExit("text-tokens, heads, and head-dim must be positive")

    torch.manual_seed(20260823)
    model = AwaAttentionGraph(
        args.size,
        args.windows,
        args.text_tokens,
        args.heads,
        args.head_dim,
        args.shifted,
    ).eval()
    video = torch.randn(*args.size, 3, args.heads, args.head_dim)
    text = torch.randn(args.text_tokens, 3, args.heads, args.head_dim)
    traced = torch.jit.trace(model, (video, text), check_trace=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = (args.output_dir / "awa_attention.pt").resolve()
    traced.save(str(model_path))
    print(f"saved {model_path}")

    if args.pnnx is None:
        return
    if not args.pnnx.is_file():
        raise SystemExit(f"pnnx executable not found: {args.pnnx}")
    input_shape = (
        f"[{args.size[0]},{args.size[1]},{args.size[2]},3,{args.heads},{args.head_dim}],"
        f"[{args.text_tokens},3,{args.heads},{args.head_dim}]"
    )
    subprocess.run(
        [str(args.pnnx), str(model_path), f"inputshape={input_shape}"],
        cwd=args.output_dir.resolve(),
        check=True,
    )


if __name__ == "__main__":
    main()
