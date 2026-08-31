"""Export shape-parameterized SeedVR2 VAE encode/decode graphs for PNNX."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn


def migrate_legacy_depth_to_space_params(param_path: Path) -> bool:
    """Correct the asymmetric upsample axis order emitted by older exports."""

    param_path = Path(param_path)
    lines = param_path.read_text().splitlines()
    changed = False
    rewritten: list[str] = []
    for line in lines:
        fields = line.split()
        if len(fields) >= 9 and fields[0] == "SeedVR2DepthToSpace":
            parameters = {field.partition("=")[0]: field.partition("=")[2] for field in fields[6:] if "=" in field}
            if parameters.get("0") == "1" and parameters.get("1") == "2" and parameters.get("2") == "2":
                fields = ["0=2" if field == "0=1" else "2=1" if field == "2=2" else field for field in fields]
                changed = True
        rewritten.append(" ".join(fields) if fields else line)
    if changed:
        param_path.write_text("\n".join(rewritten) + "\n")
    return changed


def rewrite_ncnn_param(param_path: Path) -> None:
    """Replace fixed-shape causal tile/concat triples with one custom layer.

    For the initial one-frame VAE graph, every matching triple prepends two
    copies of the first temporal frame. It contains no weight layer, so the
    generated ncnn binary remains valid without rewriting its data order.
    """

    migrate_legacy_depth_to_space_params(param_path)
    lines = param_path.read_text().splitlines()
    if len(lines) < 2 or lines[0] != "7767517":
        raise ValueError(f"unexpected ncnn param header: {param_path}")

    def fields(line: str) -> list[str]:
        return line.split()

    def param_value(layer: list[str], key: int, default: str) -> str:
        prefix = f"{key}="
        for token in layer[6:]:
            if token.startswith(prefix):
                return token[len(prefix):]
        return default

    def set_param_value(layer: list[str], key: int, value: str) -> None:
        prefix = f"{key}="
        for index, token in enumerate(layer[6:], start=6):
            if token.startswith(prefix):
                layer[index] = f"{prefix}{value}"
                return
        layer.append(f"{prefix}{value}")

    def has_zero_convolution_padding(layer: list[str]) -> bool:
        try:
            left = int(param_value(layer, 4, "0"))
            right = int(param_value(layer, 15, str(left)))
            top = int(param_value(layer, 14, str(left)))
            bottom = int(param_value(layer, 16, str(top)))
            front = int(param_value(layer, 24, str(left)))
            behind = int(param_value(layer, 17, str(front)))
        except ValueError:
            return False
        return left == right == top == bottom == front == behind == 0

    def is_fusable_zero_padding(layer: list[str]) -> bool:
        if len(layer) < 6 or layer[0] != "Padding" or layer[2:4] != ["1", "1"]:
            return False
        try:
            return (
                int(param_value(layer, 4, "0")) == 0
                and float(param_value(layer, 5, "0")) == 0.0
                and int(param_value(layer, 6, "0")) == 0
                and int(param_value(layer, 7, "0")) == 0
                and int(param_value(layer, 8, "0")) == 0
            )
        except ValueError:
            return False

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
        is_standalone_causal_tile = (
            len(split) == 7
            and split[:4] == ["Split", split[1], "1", "2"]
            and len(tile) == 6
            and tile[:4] == ["torch.tile", tile[1], "1", "1"]
            and tile[4] == split[6]
        )
        if is_depth_to_space:
            # PNNX stores the trailing rearrange factors as temporal, width,
            # height. SeedVR2DepthToSpace consumes height, width, temporal.
            _, _, _, _, z, y, x = depth_shape
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
        if is_standalone_causal_tile:
            # The first Split output is still consumed by the following branch.
            # Keep the Split and replace only its tile branch.
            rewritten.append(lines[index])
            rewritten.append(
                f"SeedVR2TemporalPad temporal_pad_{replacements} 1 1 {tile[4]} {tile[5]} 0=2"
            )
            replacements += 1
            index += 2
            continue
        rewritten.append(lines[index])
        index += 1

    fused: list[str] = []
    index = 0
    while index < len(rewritten):
        temporal_pad = fields(rewritten[index])
        next_layer = fields(rewritten[index + 1]) if index + 1 < len(rewritten) else []
        convolution = next_layer
        is_fusable_causal_convolution = (
            len(temporal_pad) == 7
            and temporal_pad[:4] == ["SeedVR2TemporalPad", temporal_pad[1], "1", "1"]
            and len(convolution) >= 7
            and convolution[0] == "Convolution3D"
            and convolution[2:4] == ["1", "1"]
            and convolution[4] == temporal_pad[5]
        )
        if is_fusable_causal_convolution:
            convolution[0] = "SeedVR2CausalConv3D"
            convolution[4] = temporal_pad[4]
            convolution.append(f"31={temporal_pad[6].split('=', 1)[1]}")
            fused.append(" ".join(convolution))
            index += 2
            continue

        padding = next_layer
        convolution = fields(rewritten[index + 2]) if index + 2 < len(rewritten) else []
        is_fusable_padding_causal_convolution = (
            len(temporal_pad) == 7
            and temporal_pad[:4] == ["SeedVR2TemporalPad", temporal_pad[1], "1", "1"]
            and is_fusable_zero_padding(padding)
            and padding[4] == temporal_pad[5]
            and len(convolution) >= 7
            and convolution[0] == "Convolution3D"
            and convolution[2:4] == ["1", "1"]
            and convolution[4] == padding[5]
            and has_zero_convolution_padding(convolution)
        )
        if is_fusable_padding_causal_convolution:
            convolution[0] = "SeedVR2CausalConv3D"
            convolution[4] = temporal_pad[4]
            set_param_value(convolution, 4, param_value(padding, 2, "0"))
            set_param_value(convolution, 15, param_value(padding, 3, "0"))
            set_param_value(convolution, 14, param_value(padding, 0, "0"))
            set_param_value(convolution, 16, param_value(padding, 1, "0"))
            convolution.append(f"31={temporal_pad[6].split('=', 1)[1]}")
            fused.append(" ".join(convolution))
            index += 3
            continue

        fused.append(rewritten[index])
        index += 1

    if replacements == 0:
        has_custom_layer = any(
            line.startswith(("SeedVR2TemporalPad ", "SeedVR2DepthToSpace ", "SeedVR2CausalConv3D "))
            for line in lines[2:]
        )
        if not has_custom_layer:
            raise ValueError(f"cannot locate causal temporal tile boundary in {param_path}")
        if fused == lines[2:]:
            return
    header = lines[1].split()
    if len(header) != 2:
        raise ValueError(f"invalid ncnn layer/blob count in {param_path}")
    layer_count = len(fused)
    fused.insert(0, lines[0])
    fused.insert(1, f"{layer_count} {header[1]}")
    param_path.write_text("\n".join(fused) + "\n")


def normalize_dynamic_vae_template(param_path: Path) -> None:
    """Replace VAE spatial reshapes with expressions evaluated by ncnn at runtime."""

    param_path = Path(param_path)
    lines = param_path.read_text().splitlines()
    if len(lines) < 2 or lines[0] != "7767517":
        raise ValueError(f"unexpected ncnn param header: {param_path}")

    def parse_layer(line: str) -> dict[str, object]:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"invalid ncnn layer line: {line}")
        try:
            bottom_count = int(fields[2])
            top_count = int(fields[3])
        except ValueError as exc:
            raise ValueError(f"invalid ncnn blob counts: {line}") from exc
        parameter_start = 4 + bottom_count + top_count
        if bottom_count < 0 or top_count < 0 or len(fields) < parameter_start:
            raise ValueError(f"invalid ncnn blob declaration: {line}")
        return {
            "fields": fields,
            "bottom_count": bottom_count,
            "top_count": top_count,
            "bottoms": tuple(fields[4 : 4 + bottom_count]),
            "tops": tuple(fields[4 + bottom_count : parameter_start]),
            "parameter_start": parameter_start,
        }

    def parameters(layer: dict[str, object]) -> dict[str, str]:
        fields = layer["fields"]
        parameter_start = layer["parameter_start"]
        assert isinstance(fields, list) and isinstance(parameter_start, int)
        values: dict[str, str] = {}
        for field in fields[parameter_start:]:
            key, separator, value = field.partition("=")
            if separator:
                values[key] = value
        return values

    def set_shape_expr(layer: dict[str, object], expression: str, shape_reference: str | None = None) -> None:
        fields = layer["fields"]
        bottom_count = layer["bottom_count"]
        top_count = layer["top_count"]
        bottoms = layer["bottoms"]
        tops = layer["tops"]
        parameter_start = layer["parameter_start"]
        assert isinstance(fields, list) and isinstance(bottom_count, int) and isinstance(top_count, int)
        assert isinstance(bottoms, tuple) and isinstance(tops, tuple) and isinstance(parameter_start, int)
        parameter_fields = [field for field in fields[parameter_start:] if not field.startswith("6=")]
        if shape_reference is not None:
            bottoms = (*bottoms, shape_reference)
            bottom_count += 1
            layer["bottom_count"] = bottom_count
            layer["bottoms"] = bottoms
        layer["fields"] = [
            fields[0],
            fields[1],
            str(bottom_count),
            str(top_count),
            *bottoms,
            *tops,
            *parameter_fields,
            f'6="{expression}"',
        ]
        layer["parameter_start"] = 4 + bottom_count + top_count

    layers = [parse_layer(line) for line in lines[2:]]
    producers: dict[str, int] = {}
    consumers: dict[str, list[int]] = {}
    for layer_index, layer in enumerate(layers):
        tops = layer["tops"]
        bottoms = layer["bottoms"]
        assert isinstance(tops, tuple) and isinstance(bottoms, tuple)
        for top in tops:
            if top in producers:
                raise ValueError(f"duplicate ncnn blob producer for {top}")
            producers[top] = layer_index
        for bottom in bottoms:
            consumers.setdefault(bottom, []).append(layer_index)

    reshape_count = 0
    dynamic_reshape_count = 0
    replacements = 0
    for layer_index, layer in enumerate(layers):
        fields = layer["fields"]
        bottoms = layer["bottoms"]
        tops = layer["tops"]
        assert isinstance(fields, list) and isinstance(bottoms, tuple) and isinstance(tops, tuple)
        if fields[0] != "Reshape":
            continue
        reshape_count += 1
        if "6" in parameters(layer):
            dynamic_reshape_count += 1
            continue
        if len(bottoms) != 1 or len(tops) != 1:
            raise ValueError(f"unsupported VAE reshape boundary: {fields[1]}")

        shape = parameters(layer)
        rank3 = {"0", "1", "2"}.issubset(shape) and "11" not in shape
        rank4 = {"0", "1", "2", "11"}.issubset(shape)
        rank2 = {"0", "1"}.issubset(shape) and "2" not in shape and "11" not in shape
        producer_index = producers.get(bottoms[0])
        producer = layers[producer_index] if producer_index is not None else None
        producer_params = parameters(producer) if producer is not None else {}

        if rank4:
            set_shape_expr(layer, "0w,0h,0c,1")
        elif rank2:
            set_shape_expr(layer, "*(0w,0h),0c")
        elif rank3 and producer is not None and producer["fields"][0] == "Permute" and producer_params.get("0") == "6":
            set_shape_expr(layer, "0w,0h,0d")
        elif rank3 and producer is not None and producer["fields"][0] == "Permute" and producer_params.get("0") == "1":
            next_layers = consumers.get(tops[0], [])
            if len(next_layers) != 1:
                raise ValueError(f"ambiguous VAE attention reshape consumer: {fields[1]}")
            residual_add = layers[next_layers[0]]
            residual_fields = residual_add["fields"]
            residual_bottoms = residual_add["bottoms"]
            assert isinstance(residual_fields, list) and isinstance(residual_bottoms, tuple)
            if residual_fields[0] != "BinaryOp" or len(residual_bottoms) != 2 or tops[0] not in residual_bottoms:
                raise ValueError(f"unsupported VAE attention reshape consumer: {fields[1]}")
            residual_blob = residual_bottoms[0] if residual_bottoms[1] == tops[0] else residual_bottoms[1]
            set_shape_expr(layer, "1w,1h,1c", residual_blob)
        else:
            raise ValueError(f"unsupported static VAE reshape: {fields[1]}")
        replacements += 1

    if replacements == 0:
        if reshape_count > 0 and dynamic_reshape_count == reshape_count:
            return
        raise ValueError(f"no static VAE reshapes found in {param_path}")
    param_path.write_text("\n".join((lines[0], lines[1], *(" ".join(layer["fields"]) for layer in layers))) + "\n")


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


def _pnnx_command(pnnx: Path, model_path: Path, input_shape: tuple[int, ...]) -> list[str]:
    shape = ",".join(str(value) for value in input_shape)
    return [str(pnnx.resolve()), str(model_path.resolve()), f"inputshape=[{shape}]"]


def disable_memory_slicing_for_trace(vae: nn.Module) -> None:
    """Trace the direct causal-convolution path PNNX can map to ncnn."""

    if hasattr(vae, "set_memory_limit"):
        vae.set_memory_limit(conv_max_mem=None, norm_max_mem=None)


def _resolve_vae_paths(upstream_root: Path, checkpoint: Path) -> tuple[Path, Path]:
    """Resolve paths before changing cwd for imports from the upstream tree."""

    return upstream_root.expanduser().resolve(), checkpoint.expanduser().resolve()


def _load_vae(upstream_root: Path, checkpoint: Path, device: torch.device):
    upstream_root, checkpoint = _resolve_vae_paths(upstream_root, checkpoint)
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
    parser.add_argument(
        "--alternate-input-shape",
        type=_shape,
        default=(1, 3, 1, 720, 1280),
        metavar="B,C,T,H,W",
    )
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
    alternate_sample = torch.zeros(args.alternate_input_shape, device="cuda", dtype=dtype)
    scale = float(config.vae.scaling_factor)
    encode = VaeEncodeWrapper(vae, scale).eval()
    decode = VaeDecodeWrapper(vae, scale).eval()

    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        latent = encode(sample)
        alternate_latent = encode(alternate_sample)
        reconstruction = decode(latent)
        alternate_reconstruction = decode(alternate_latent)
        disable_memory_slicing_for_trace(vae)
        encode_trace = torch.jit.trace(encode, sample, check_trace=False)
        decode_trace = torch.jit.trace(decode, latent, check_trace=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    encode_path = (args.output_dir / "vae_encode.pt").resolve()
    decode_path = (args.output_dir / "vae_decode.pt").resolve()
    encode_trace.save(str(encode_path))
    decode_trace.save(str(decode_path))
    print(f"encode_input={tuple(sample.shape)} encode_output={tuple(latent.shape)}")
    print(f"decode_input={tuple(latent.shape)} decode_output={tuple(reconstruction.shape)}")
    print(f"alternate_encode_input={tuple(alternate_sample.shape)} alternate_encode_output={tuple(alternate_latent.shape)}")
    print(
        f"alternate_decode_input={tuple(alternate_latent.shape)} "
        f"alternate_decode_output={tuple(alternate_reconstruction.shape)}"
    )
    print(f"saved={encode_path} {decode_path}")

    if args.pnnx is not None:
        for model_path, input_tensor in ((encode_path, sample), (decode_path, latent)):
            subprocess.run(
                _pnnx_command(args.pnnx, model_path, tuple(input_tensor.shape)),
                cwd=args.output_dir.resolve(),
                check=True,
            )
            rewrite_ncnn_param(args.output_dir / f"{model_path.stem}.ncnn.param")
            normalize_dynamic_vae_template(args.output_dir / f"{model_path.stem}.ncnn.param")


if __name__ == "__main__":
    main()
