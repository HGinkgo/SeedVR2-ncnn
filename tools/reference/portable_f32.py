"""Portable named float32 tensor records shared by native golden tests."""

from __future__ import annotations

import hashlib
from pathlib import Path
import struct

import torch


GOLDEN_MAGIC = b"SVR2F32\0"
GOLDEN_VERSION = 1
_GOLDEN_FILE_HEADER = struct.Struct("<8sII")
_GOLDEN_RECORD_HEADER = struct.Struct("<HBBQ4Q")


def write_portable_golden(path: Path, records: list[tuple[str, torch.Tensor]]) -> dict[str, object]:
    """Write named contiguous little-endian float32 tensors for native tests."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    manifest_records: list[dict[str, object]] = []
    with path.open("wb") as destination:
        destination.write(_GOLDEN_FILE_HEADER.pack(GOLDEN_MAGIC, GOLDEN_VERSION, len(records)))
        for name, tensor in records:
            if not name or name in seen:
                raise ValueError(f"golden record name must be unique and non-empty: {name!r}")
            seen.add(name)
            value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
            shape = tuple(int(dimension) for dimension in value.shape)
            if len(shape) > 4:
                raise ValueError(f"golden record rank must be <= 4: {name} has {shape}")
            raw = value.numpy().astype("<f4", copy=False).tobytes(order="C")
            shape_header = shape + (0,) * (4 - len(shape))
            encoded_name = name.encode("utf-8")
            if len(encoded_name) > 0xFFFF:
                raise ValueError(f"golden record name is too long: {name}")
            destination.write(
                _GOLDEN_RECORD_HEADER.pack(
                    len(encoded_name),
                    len(shape),
                    0,
                    int(value.numel()),
                    *shape_header,
                )
            )
            destination.write(encoded_name)
            offset = destination.tell()
            destination.write(raw)
            manifest_records.append(
                {
                    "name": name,
                    "shape": list(shape),
                    "count": int(value.numel()),
                    "offset": int(offset),
                    "byte_length": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
    return {
        "format": "seedvr2_f32_records",
        "version": GOLDEN_VERSION,
        "record_count": len(manifest_records),
        "records": manifest_records,
    }
