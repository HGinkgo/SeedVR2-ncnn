"""Export the fixed-shape AWA attention boundary for PNNX inspection."""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import nn

from tools.reference.awa_attention import (
    AwaAttentionPack,
    AwaAttentionUnpack,
    AwaWindowAttention,
)


def rewrite_ncnn_param(
    param_path: Path,
    *,
    size: Sequence[int],
    windows: Sequence[int],
    text_tokens: int,
    shifted: bool,
) -> None:
    """Replace the fixed AWA index-select boundary with ncnn custom layers.

    The generated binary is kept unchanged: disconnected ``MemoryData`` lines
    remain in the parameter file so its constant-read order stays valid.
    """

    lines = param_path.read_text().splitlines()
    if len(lines) < 2 or not lines[0].startswith("7767517"):
        raise ValueError(f"unexpected ncnn param header: {param_path}")

    def layer_is(line: str, layer_type: str, layer_name: Optional[str] = None) -> bool:
        fields = line.split()
        return len(fields) >= 2 and fields[0] == layer_type and (layer_name is None or fields[1] == layer_name)

    first_index = next(
        (index for index, line in enumerate(lines) if layer_is(line, "torch.index_select")),
        None,
    )
    first_split = next(
        (index for index, line in enumerate(lines) if layer_is(line, "Split", "splitncnn_0")),
        None,
    )
    second_split = next(
        (index for index, line in enumerate(lines) if layer_is(line, "Split", "splitncnn_1")),
        None,
    )
    if first_index is None or first_split is None or second_split is None or first_index > first_split:
        raise ValueError("cannot locate AWA pack boundary in ncnn param")

    source_t, source_h, source_w = (int(item) for item in size)
    windows_t, windows_h, windows_w = (int(item) for item in windows)
    pack_line = (
        "SeedVR2AWAPack awa_pack 2 2 in0 in1 4 awa_cu_seqlens "
        f"0={source_t} 1={source_h} 2={source_w} "
        f"3={windows_t} 4={windows_h} 5={windows_w} "
        f"6={int(text_tokens)} 7={1 if shifted else 0}"
    )

    prefix = lines[:first_index] + [pack_line] + lines[first_index + 1 : second_split + 1]
    constants = [line for line in lines[second_split + 1 :] if layer_is(line, "MemoryData")]
    second_split_fields = lines[second_split].split()
    if len(second_split_fields) < 6:
        raise ValueError("invalid AWA unpack split line in ncnn param")
    attended_blob = second_split_fields[4]
    unpack_line = (
        f"SeedVR2AWAUnpack awa_unpack 1 2 {attended_blob} out0 out1 "
        f"0={source_t} 1={source_h} 2={source_w} "
        f"3={windows_t} 4={windows_h} 5={windows_w} "
        f"6={int(text_tokens)} 7={1 if shifted else 0}"
    )
    rewritten = prefix + constants + [unpack_line]
    rewritten[1] = f"{len(rewritten) - 2} {lines[1].split()[1]}"
    param_path.write_text("\n".join(rewritten) + "\n")


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
    video = torch.randn(math.prod(args.size), 3, args.heads, args.head_dim)
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
        f"[{math.prod(args.size)},3,{args.heads},{args.head_dim}],"
        f"[{args.text_tokens},3,{args.heads},{args.head_dim}]"
    )
    subprocess.run(
        [str(args.pnnx), str(model_path), f"inputshape={input_shape}"],
        cwd=args.output_dir.resolve(),
        check=True,
    )
    rewrite_ncnn_param(
        args.output_dir / "awa_attention.ncnn.param",
        size=args.size,
        windows=args.windows,
        text_tokens=args.text_tokens,
        shifted=args.shifted,
    )


if __name__ == "__main__":
    main()
