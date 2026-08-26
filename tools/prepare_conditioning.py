"""Prepare fixed-shape SeedVR2 conditioning tensors for the C++ runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import torch


DEFAULT_SHAPE = (58, 5120)


def prepare_conditioning(
    source: Path,
    output: Path,
    *,
    expected_shape: Sequence[int] = DEFAULT_SHAPE,
) -> dict[str, object]:
    """Convert one checked-out ``.pt`` condition tensor to little-endian f32."""

    source = Path(source)
    output = Path(output)
    expected = tuple(int(dimension) for dimension in expected_shape)
    value = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"expected a tensor in {source}, got {type(value).__name__}")
    if tuple(value.shape) != expected:
        raise ValueError(f"expected shape {expected}, got {tuple(value.shape)}")

    raw = value.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()
    raw = raw.astype("<f4", copy=False).tobytes(order="C")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw)
    return {
        "source": str(source),
        "output": str(output),
        "shape": list(expected),
        "dtype": "float32",
        "count": int(value.numel()),
        "byte_length": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="source PyTorch tensor")
    parser.add_argument("output", type=Path, help="little-endian float32 output")
    parser.add_argument("--manifest", type=Path, help="optional JSON manifest path")
    parser.add_argument("--tokens", type=int, default=DEFAULT_SHAPE[0], help="expected token count")
    parser.add_argument("--width", type=int, default=DEFAULT_SHAPE[1], help="expected embedding width")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = prepare_conditioning(args.source, args.output, expected_shape=(args.tokens, args.width))
    manifest_path = args.manifest or Path(f"{args.output}.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
