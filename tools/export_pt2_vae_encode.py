# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 SeedVR2-ncnn contributors

"""Export the SeedVR2 VAE encoder as a PT2 program.

The default is the validated one-frame ncnn package contract
``[1, 3, 1, 256, 256] -> [1, 16, 32, 32]``.  ``--dynamic-spatial`` is an
explicit probe for a finite dynamic spatial PT2 contract.  It uses the
deterministic encoder core (rather than posterior sampling), records the
exported range constraints, and deliberately leaves conversion to a separate
step: the current independent PNNX frontend rejects dynamic rank-3+ values
with a precise diagnostic instead of producing a partially dynamic graph.
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

from tools.export_pt2_vae import _ensure_torchvision_schema, _init_process_group
from tools.reference.seedvr2_baseline import _load_reference_vae, configure_vae_precision


class FixedVaeEncode(nn.Module):
    """Expose the upstream encoder at the native one-frame tensor boundary."""

    def __init__(self, vae: nn.Module, scaling_factor: float, shifting_factor: float) -> None:
        super().__init__()
        self.vae = vae
        self.scaling_factor = float(scaling_factor)
        self.shifting_factor = float(shifting_factor)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        encoded = self.vae.encode(self.vae.preprocess(sample))
        latent = encoded.posterior.mode().squeeze(2)
        return (latent - self.shifting_factor) * self.scaling_factor


class DynamicSpatialVaeEncode(FixedVaeEncode):
    """Use the deterministic encoder core for dynamic spatial export.

    ``AutoencoderKLTemporalDecoder.encode`` contains posterior sampling
    helpers whose random shape construction is not exportable with symbolic
    height/width.  The selected ``mode`` is deterministic, so call the
    upstream private encoder directly and construct the distribution around
    its moments.  This changes only the export boundary; it does not widen the
    ncnn frontend's supported operator or dynamic-shape surface.
    """

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

        moments = self.vae._encode(self.vae.preprocess(sample))
        latent = DiagonalGaussianDistribution(moments).mode().squeeze(2)
        return (latent - self.shifting_factor) * self.scaling_factor


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
    parser.add_argument("--input", type=Path, required=True,
                        help="portable CTHW input tensor with shape [3,1,256,256]")
    parser.add_argument("--output", type=Path, required=True,
                        help="destination .pt2 file")
    parser.add_argument("--manifest", type=Path,
                        help="optional JSON sidecar; defaults to <output>.manifest.json")
    parser.add_argument(
        "--dynamic-spatial", action="store_true",
        help="export a finite symbolic height/width probe (conversion is currently rejected)",
    )
    parser.add_argument(
        "--spatial-max", type=int, default=512,
        help="upper bound for dynamic height/width (default: 512)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    upstream_root = args.upstream_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    input_path = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not upstream_root.is_dir() or not checkpoint.is_file() or not input_path.is_file():
        raise SystemExit("upstream root, checkpoint, and input must exist")
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

        sample = torch.load(input_path, map_location="cpu", weights_only=True)
        if tuple(sample.shape) == (3, 1, 256, 256):
            sample = sample.unsqueeze(0)
        if tuple(sample.shape) != (1, 3, 1, 256, 256):
            raise ValueError(f"expected CTHW [3,1,256,256] or BCHW5 [1,3,1,256,256], got {tuple(sample.shape)}")
        sample = sample.to(device=device, dtype=torch.float32)
        scale = float(config.vae.scaling_factor)
        shift = float(config.vae.get("shifting_factor", 0.0))
        if args.spatial_max < 256:
            raise ValueError("--spatial-max must be at least the example size 256")
        wrapper = (DynamicSpatialVaeEncode if args.dynamic_spatial else FixedVaeEncode)(
            vae, scale, shift
        ).eval()
        with torch.inference_mode():
            eager = wrapper(sample)
            if not args.dynamic_spatial and tuple(eager.shape) != (1, 16, 32, 32):
                raise ValueError(f"unexpected encoder output shape: {tuple(eager.shape)}")
            export_kwargs = {}
            if args.dynamic_spatial:
                height = torch.export.Dim.AUTO(max=args.spatial_max)
                width = torch.export.Dim.AUTO(max=args.spatial_max)
                export_kwargs["dynamic_shapes"] = ({2: height, 3: width},)
            exported = torch.export.export(wrapper, (sample,), **export_kwargs)

        output.parent.mkdir(parents=True, exist_ok=True)
        torch.export.save(exported, output)
        manifest_path = (args.manifest or output.with_suffix(".manifest.json")).resolve()
        manifest = {
            "exporter": "seedvr2-vae-encode-v2",
            "model": "vae_encode_dynamic_spatial_probe" if args.dynamic_spatial else "vae_encode_256_fixed",
            "pt2": str(output),
            "pt2_sha256": _sha256(output),
            "upstream_root": str(upstream_root),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "input_source": str(input_path),
            "input_shape": list(sample.shape),
            "output_shape": list(eager.shape),
            "dynamic_spatial": bool(args.dynamic_spatial),
            "spatial_max": args.spatial_max if args.dynamic_spatial else None,
            "range_constraints": {str(symbol): str(bounds) for symbol, bounds in exported.range_constraints.items()},
            "scaling_factor": scale,
            "shifting_factor": shift,
            "torch_version": torch.__version__,
            "loading_info": str(loading_info),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"saved={output}")
        print(f"input_shape={tuple(sample.shape)} output_shape={tuple(eager.shape)}")
        print(f"pt2_sha256={manifest['pt2_sha256']}")
        print(f"manifest={manifest_path}")
    finally:
        if owns_pg:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
