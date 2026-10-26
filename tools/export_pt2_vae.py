# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 SeedVR2-ncnn contributors

"""Export the fixed SeedVR2 VAE decoder as a PyTorch ExportedProgram.

This is an experimental conversion artifact.  It deliberately exports only
the one-frame 256x256 decoder contract used by the current native package:
``[1, 16, 32, 32] -> [1, 3, 256, 256]``.  The production runtime continues to
use the validated TorchScript/ncnn graph until the PT2 graph has complete PNNX
and ncnn CPU/Vulkan evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.reference.seedvr2_baseline import _load_reference_vae, configure_vae_precision


_TORCHVISION_SCHEMA = None


class FixedVaeDecode(nn.Module):
    """Expose the upstream decoder with the package's latent scaling contract."""

    def __init__(self, vae: nn.Module, scaling_factor: float, shifting_factor: float) -> None:
        super().__init__()
        self.vae = vae
        self.scaling_factor = float(scaling_factor)
        self.shifting_factor = float(shifting_factor)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # The wrapper accepts NCHW. The upstream decoder adds its one-frame
        # temporal axis internally, matching the native ncnn graph boundary.
        return self.vae.decode(latent / self.scaling_factor + self.shifting_factor).sample


def _ensure_torchvision_schema() -> None:
    """Make diffusers importable in environments without torchvision ops."""

    global _TORCHVISION_SCHEMA
    try:
        _TORCHVISION_SCHEMA = torch.library.Library("torchvision", "DEF")
        _TORCHVISION_SCHEMA.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
    except RuntimeError:
        # A real torchvision build may already have registered the schema.
        pass


def _init_process_group() -> bool:
    if torch.distributed.is_initialized():
        return False
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29613")
    torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--latent", type=Path, required=True,
                        help="portable or regular torch file containing [16,1,32,32] CTHW latent")
    parser.add_argument("--output", type=Path, required=True,
                        help="destination .pt2 file")
    parser.add_argument("--manifest", type=Path,
                        help="optional JSON sidecar; defaults to <output>.manifest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    upstream_root = args.upstream_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    latent_path = args.latent.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not upstream_root.is_dir() or not checkpoint.is_file() or not latent_path.is_file():
        raise SystemExit("upstream root, checkpoint, and latent must exist")
    if not torch.cuda.is_available():
        raise SystemExit("PT2 VAE export requires CUDA-enabled PyTorch")

    _ensure_torchvision_schema()
    owns_pg = _init_process_group()
    try:
        device = torch.device("cuda")
        vae, config, loading_info = _load_reference_vae(upstream_root, checkpoint, device)
        configure_vae_precision(vae, "export-fp32")
        if hasattr(vae, "set_causal_slicing"):
            vae.set_causal_slicing(split_size=None, memory_device="same")
        if hasattr(vae, "set_memory_limit"):
            vae.set_memory_limit(conv_max_mem=None, norm_max_mem=None)

        latent = torch.load(latent_path, map_location="cpu", weights_only=True)
        if tuple(latent.shape) == (1, 32, 32, 16):
            latent = latent.permute(3, 0, 1, 2).contiguous()
        if tuple(latent.shape) != (16, 1, 32, 32):
            raise ValueError(f"expected CTHW [16,1,32,32] or THWC [1,32,32,16], got {tuple(latent.shape)}")
        latent = latent[:, 0].unsqueeze(0).to(device=device, dtype=torch.float32)
        scale = float(config.vae.scaling_factor)
        shift = float(config.vae.get("shifting_factor", 0.0))
        wrapper = FixedVaeDecode(vae, scale, shift).eval()
        with torch.inference_mode():
            eager = wrapper(latent)
            if tuple(eager.shape) != (1, 3, 256, 256):
                raise ValueError(f"unexpected decoder output shape: {tuple(eager.shape)}")
            exported = torch.export.export(wrapper, (latent,))

        output.parent.mkdir(parents=True, exist_ok=True)
        torch.export.save(exported, output)
        manifest_path = (args.manifest or output.with_suffix(".manifest.json")).resolve()
        manifest = {
            "exporter": "seedvr2-pt2-vae-v1",
            "model": "vae_decode_256_fixed",
            "pt2": str(output),
            "pt2_sha256": _sha256(output),
            "upstream_root": str(upstream_root),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "latent_source": str(latent_path),
            "latent_shape": list(latent.shape),
            "output_shape": list(eager.shape),
            "scaling_factor": scale,
            "shifting_factor": shift,
            "torch_version": torch.__version__,
            "loading_info": str(loading_info),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"saved={output}")
        print(f"input_shape={tuple(latent.shape)} output_shape={tuple(eager.shape)}")
        print(f"pt2_sha256={manifest['pt2_sha256']}")
        print(f"manifest={manifest_path}")
    finally:
        if owns_pg:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
