"""Export fixed-shape SeedVR2 VAE encode/decode graphs for PNNX inspection."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn


def rewrite_ncnn_param(param_path: Path) -> None:
    """Replace fixed-shape causal tile/concat triples with one custom layer.

    For the initial one-frame VAE graph, every matching triple prepends two
    copies of the first temporal frame. It contains no weight layer, so the
    generated ncnn binary remains valid without rewriting its data order.
    """

    lines = param_path.read_text().splitlines()
    if len(lines) < 2 or lines[0] != "7767517":
        raise ValueError(f"unexpected ncnn param header: {param_path}")

    def fields(line: str) -> list[str]:
        return line.split()

    rewritten: list[str] = []
    replacements = 0
    depth_to_space_replacements = 0
    index = 2
    while index < len(lines):
        split = fields(lines[index])
        if (
            len(split) >= 9
            and split[0] == "Crop"
            and "-23310=1,-233" in split
            and "-23311=1,1" in split
            and "-23309=1,2" in split
        ):
            rewritten.append(" ".join("-23309=1,1" if token == "-23309=1,2" else token for token in split))
            replacements += 1
            index += 1
            continue
        tile = fields(lines[index + 1]) if index + 1 < len(lines) else []
        concat = fields(lines[index + 2]) if index + 2 < len(lines) else []
        reshape = split
        permute = tile
        output_reshape = concat

        depth_shape = None
        if len(reshape) >= 7 and reshape[0] == "Reshape":
            for token in reshape[6:]:
                if token.startswith("6="):
                    raw_shape = token[2:].strip('"')
                    try:
                        values = tuple(int(value) for value in raw_shape.split(","))
                    except ValueError:
                        values = ()
                    if len(values) == 7 and all(value > 0 for value in values):
                        depth_shape = values
                    break
        is_depth_to_space = (
            depth_shape is not None
            and len(permute) == 7
            and permute[:4] == ["Permute", permute[1], "1", "1"]
            and permute[-1] == "0=0"
            and len(output_reshape) >= 9
            and output_reshape[:4] == ["Reshape", output_reshape[1], "1", "1"]
            and output_reshape[4] == permute[5]
            and output_reshape[5] != ""
        )
        is_causal_tile = (
            len(split) == 7
            and split[:4] == ["Split", split[1], "1", "2"]
            and len(tile) == 6
            and tile[:4] == ["torch.tile", tile[1], "1", "1"]
            and len(concat) >= 8
            and concat[:4] == ["Concat", concat[1], "2", "1"]
            and tile[4] == split[6]
            and concat[4] == tile[5]
            and concat[5] == split[5]
            and concat[-1] == "0=1"
        )
        if is_depth_to_space:
            _, _, _, _, x, y, z = depth_shape
            rewritten.append(
                f"SeedVR2DepthToSpace depth_to_space_{depth_to_space_replacements} 1 1 "
                f"{reshape[4]} {output_reshape[5]} 0={x} 1={y} 2={z}"
            )
            depth_to_space_replacements += 1
            replacements += 1
            index += 3
            continue
        if is_causal_tile:
            rewritten.append(
                f"SeedVR2TemporalPad temporal_pad_{replacements} 1 1 {split[4]} {concat[6]} 0=2"
            )
            replacements += 1
            index += 3
            continue
        rewritten.append(lines[index])
        index += 1

    if replacements == 0:
        if any(line.startswith(("SeedVR2TemporalPad ", "SeedVR2DepthToSpace ")) for line in lines[2:]):
            return
        raise ValueError(f"cannot locate causal temporal tile boundary in {param_path}")
    header = lines[1].split()
    if len(header) != 2:
        raise ValueError(f"invalid ncnn layer/blob count in {param_path}")
    layer_count = len(rewritten)
    rewritten.insert(0, lines[0])
    rewritten.insert(1, f"{layer_count} {header[1]}")
    param_path.write_text("\n".join(rewritten) + "\n")


class VaeEncodeWrapper(nn.Module):
    def __init__(self, vae: nn.Module, scaling_factor: float) -> None:
        super().__init__()
        self.vae = vae
        self.scaling_factor = float(scaling_factor)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        encoded = self.vae.encode(self.vae.preprocess(sample))
        latent = encoded.posterior.mode().squeeze(2)
        return latent * self.scaling_factor


class VaeDecodeWrapper(nn.Module):
    def __init__(self, vae: nn.Module, scaling_factor: float) -> None:
        super().__init__()
        self.vae = vae
        self.scaling_factor = float(scaling_factor)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        decoded = self.vae.decode(latent / self.scaling_factor).sample
        return self.vae.postprocess(decoded) if hasattr(self.vae, "postprocess") else decoded


def _shape(value: str) -> tuple[int, int, int, int, int]:
    parts = tuple(int(part) for part in value.split(","))
    if len(parts) != 5 or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError("expected B,C,T,H,W as five positive integers")
    return parts


def _load_vae(upstream_root: Path, checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(upstream_root))
    previous_cwd = Path.cwd()
    os.chdir(upstream_root)
    try:
        from common.config import create_object, load_config

        config = load_config("configs_3b/main.yaml")
        vae = create_object(config.vae.model)
        vae.requires_grad_(False).eval().to(device=device, dtype=torch.bfloat16)
        state = torch.load(checkpoint, map_location=device, mmap=True, weights_only=False)
        vae.load_state_dict(state, strict=True)
        if hasattr(vae, "set_causal_slicing"):
            vae.set_causal_slicing(**config.vae.slicing)
        if hasattr(vae, "set_memory_limit"):
            vae.set_memory_limit(**config.vae.memory_limit)
        return vae, config
    finally:
        os.chdir(previous_cwd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-shape", type=_shape, default=(1, 3, 1, 128, 128), metavar="B,C,T,H,W")
    parser.add_argument("--upstream-root", type=Path, default=Path(os.environ.get("SEEDVR2_UPSTREAM_ROOT", "")))
    parser.add_argument("--checkpoint", type=Path, default=Path(os.environ.get("SEEDVR2_CKPT_DIR", "ckpts")) / "ema_vae.pth")
    parser.add_argument("--pnnx", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.upstream_root.is_dir():
        raise SystemExit("--upstream-root or SEEDVR2_UPSTREAM_ROOT must point to the frozen SeedVR source")
    if not args.checkpoint.is_file():
        raise SystemExit(f"missing VAE checkpoint: {args.checkpoint}")
    if not torch.cuda.is_available():
        raise SystemExit("VAE export requires CUDA-enabled PyTorch")

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29612")
        torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)

    vae, config = _load_vae(args.upstream_root, args.checkpoint, torch.device("cuda"))
    # PNNX's CPU interpreter cannot execute the BF16 attention path. Export
    # the first ncnn graph in FP32; later Vulkan precision changes are separate.
    vae = vae.float()
    dtype = torch.float32
    sample = torch.zeros(args.input_shape, device="cuda", dtype=dtype)
    scale = float(config.vae.scaling_factor)
    encode = VaeEncodeWrapper(vae, scale).eval()
    decode = VaeDecodeWrapper(vae, scale).eval()

    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        latent = encode(sample)
        reconstruction = decode(latent)
        encode_trace = torch.jit.trace(encode, sample, check_trace=False)
        decode_trace = torch.jit.trace(decode, latent, check_trace=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    encode_path = (args.output_dir / "vae_encode.pt").resolve()
    decode_path = (args.output_dir / "vae_decode.pt").resolve()
    encode_trace.save(str(encode_path))
    decode_trace.save(str(decode_path))
    print(f"encode_input={tuple(sample.shape)} encode_output={tuple(latent.shape)}")
    print(f"decode_input={tuple(latent.shape)} decode_output={tuple(reconstruction.shape)}")
    print(f"saved={encode_path} {decode_path}")

    if args.pnnx is not None:
        for model_path, input_tensor in ((encode_path, sample), (decode_path, latent)):
            shape = ",".join(str(value) for value in input_tensor.shape)
            subprocess.run(
                [str(args.pnnx), str(model_path), f"inputshape=[{shape}]"],
                cwd=args.output_dir.resolve(),
                check=True,
            )
            rewrite_ncnn_param(args.output_dir / f"{model_path.stem}.ncnn.param")


if __name__ == "__main__":
    main()
