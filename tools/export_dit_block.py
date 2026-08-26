"""Export and rewrite the fixed-shape SeedVR2 DiT block probe."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, Sequence

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.reference.dit_block import DIT_BLOCK_CONTRACT, load_fixed_dit_block


def _shape(value: str) -> tuple[int, int, int]:
    """Parse the only source shape currently supported by this probe."""

    try:
        parts = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise ValueError("expected three positive comma-separated integers") from exc
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise ValueError("expected three positive comma-separated integers")
    if parts != DIT_BLOCK_CONTRACT.source_shape:
        raise ValueError(
            "only the supported fixed shape "
            f"{','.join(str(item) for item in DIT_BLOCK_CONTRACT.source_shape)} is available"
        )
    return parts


def rewrite_ncnn_param(
    param_path: Path,
    *,
    size: Sequence[int] = DIT_BLOCK_CONTRACT.source_shape,
    windows: Sequence[int] = DIT_BLOCK_CONTRACT.window_shape,
    text_tokens: int = DIT_BLOCK_CONTRACT.text_tokens,
    shifted: bool = False,
) -> None:
    """Replace the traced static AWA and MMRoPE boundaries with custom layers.

    The generated model binary is deliberately not changed.  In particular,
    disconnected ``MemoryData`` layers remain in the graph so ncnn consumes
    the serialized weight stream in PNNX's original order.
    """

    lines = param_path.read_text().splitlines()
    if len(lines) < 2 or not lines[0].startswith("7767517"):
        raise ValueError(f"unexpected ncnn param header: {param_path}")

    @dataclass(frozen=True)
    class Layer:
        line: str
        layer_type: str
        name: str
        bottoms: tuple[str, ...]
        tops: tuple[str, ...]

    def parse_layer(line: str) -> Layer:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"invalid ncnn layer line: {line}")
        try:
            bottom_count = int(fields[2])
            top_count = int(fields[3])
        except ValueError as exc:
            raise ValueError(f"invalid ncnn blob counts: {line}") from exc
        begin = 4
        end = begin + bottom_count + top_count
        if bottom_count < 0 or top_count < 0 or len(fields) < end:
            raise ValueError(f"invalid ncnn blob declaration: {line}")
        return Layer(
            line=line,
            layer_type=fields[0],
            name=fields[1],
            bottoms=tuple(fields[begin : begin + bottom_count]),
            tops=tuple(fields[begin + bottom_count : end]),
        )

    def fix_text_batch_axis(line: str) -> str:
        fields = line.split()
        if len(fields) < 4 or fields[0] != "Reshape":
            return line
        if (
            f"0={DIT_BLOCK_CONTRACT.txt_dim}" in fields
            and f"1={int(text_tokens)}" in fields
            and "12=233" in fields
            and "13=1" in fields
        ):
            return " ".join("13=0" if field == "13=1" else field for field in fields)
        return line

    layers = [parse_layer(line) for line in lines[2:]]

    def fix_existing_video_batch_reshape(current_lines: list[str]) -> list[str]:
        current_layers = [parse_layer(line) for line in current_lines[2:]]
        unpack = next((layer for layer in current_layers if layer.layer_type == "SeedVR2AWAUnpack"), None)
        if unpack is None or not unpack.tops:
            return current_lines
        video_blob = unpack.tops[0]
        custom_fields = unpack.line.split()
        custom_params = {
            int(field.split("=", 1)[0]): int(field.split("=", 1)[1])
            for field in custom_fields[4 + len(unpack.bottoms) + len(unpack.tops):]
            if "=" in field and field.split("=", 1)[0].isdigit()
        }
        video_tokens = custom_params.get(0, int(size[0])) * custom_params.get(1, int(size[1])) * custom_params.get(2, int(size[2]))
        existing_index = next((index for index, layer in enumerate(current_layers) if layer.name == "reshape_awa_video_batch"), None)
        if existing_index is not None:
            fields = current_layers[existing_index].line.split()
            for index, field in enumerate(fields):
                if field.startswith("1="):
                    fields[index] = f"1={video_tokens}"
            updated = list(current_lines)
            updated[existing_index + 2] = " ".join(fields)
            return updated
        consumer_index = next(
            (index for index, layer in enumerate(current_layers)
             if layer.layer_type == "InnerProduct" and video_blob in layer.bottoms),
            None,
        )
        if consumer_index is None:
            return current_lines
        input_line = (
            f"Reshape reshape_awa_video_batch 1 1 {video_blob} awa_video_batch "
            f"0={DIT_BLOCK_CONTRACT.vid_dim} 1={video_tokens} 12=233 13=0"
        )
        rewritten = [current_lines[0], "0 0"]
        for index, layer in enumerate(current_layers):
            if index == consumer_index:
                rewritten.append(input_line)
            fields = layer.line.split()
            if index == consumer_index:
                begin = 4
                for bottom_index in range(int(fields[2])):
                    if fields[begin + bottom_index] == video_blob:
                        fields[begin + bottom_index] = "awa_video_batch"
                rewritten.append(" ".join(fields))
            else:
                rewritten.append(layer.line)
        rewritten_nodes = [parse_layer(line) for line in rewritten[2:]]
        blob_count = len({top for layer in rewritten_nodes for top in layer.tops})
        rewritten[1] = f"{len(rewritten_nodes)} {blob_count}"
        return rewritten
    custom_layer_types = {
        "SeedVR2AWAPack",
        "SeedVR2AWAUnpack",
        "SeedVR2MMRoPE",
        "SeedVR2WindowAttention",
    }
    existing_custom = [layer for layer in layers if layer.layer_type in custom_layer_types]
    existing_types = {layer.layer_type for layer in existing_custom}
    if len(existing_custom) == 2 and existing_types == {
        "SeedVR2AWAPack",
        "SeedVR2AWAUnpack",
    }:
        fixed_lines = [fix_text_batch_axis(line) for line in fix_existing_video_batch_reshape(lines)]
        if fixed_lines != lines:
            param_path.write_text("\n".join(fixed_lines) + "\n")
        return
    if len(existing_custom) == 4 and existing_types == custom_layer_types:
        fixed_lines = [fix_text_batch_axis(line) for line in fix_existing_video_batch_reshape(lines)]
        if fixed_lines != lines:
            param_path.write_text("\n".join(fixed_lines) + "\n")
        return
    if existing_custom:
        raise ValueError(f"incomplete existing SeedVR2 rewrite in {param_path}")

    producers: Dict[str, int] = {}
    consumers: dict[str, list[int]] = defaultdict(list)
    for index, layer in enumerate(layers):
        for top in layer.tops:
            if top in producers:
                raise ValueError(f"duplicate ncnn blob producer for {top}")
            producers[top] = index
        for bottom in layer.bottoms:
            consumers[bottom].append(index)

    pack_candidates: list[tuple[int, int]] = []
    for index, layer in enumerate(layers):
        if layer.layer_type != "torch.index_select" or not layer.bottoms:
            continue
        concat_index = producers.get(layer.bottoms[0])
        if concat_index is None:
            continue
        concat = layers[concat_index]
        if concat.layer_type == "Concat" and len(concat.bottoms) == 2:
            pack_candidates.append((index, concat_index))
    if len(pack_candidates) != 1:
        raise ValueError(f"expected one DiT AWA pack boundary, found {len(pack_candidates)}")
    pack_index, concat_index = pack_candidates[0]
    pack_concat_index = concat_index
    pack = layers[pack_index]
    concat = layers[concat_index]
    if len(pack.tops) != 1:
        raise ValueError("DiT AWA pack index-select must have one output")

    rope_start_candidates = [
        index
        for index, layer in enumerate(layers)
        if index > pack_index
        and layer.layer_type == "Slice"
        and layer.bottoms == (pack.tops[0],)
        and len(layer.tops) == 3
    ]
    if len(rope_start_candidates) > 1:
        raise ValueError(f"expected at most one DiT MMRoPE boundary, found {len(rope_start_candidates)}")

    rope_start_index: int | None = None
    rope_end_index: int | None = None
    if rope_start_candidates:
        rope_start_index = rope_start_candidates[0]
        rope_end_candidates: list[int] = []
        for index, layer in enumerate(layers):
            if index <= rope_start_index or layer.layer_type != "Reshape" or len(layer.bottoms) != 1:
                continue
            concat_index = producers.get(layer.bottoms[0])
            if concat_index is None:
                continue
            concat_layer = layers[concat_index]
            followers = consumers.get(layer.tops[0], [])
            if (
                concat_layer.layer_type == "Concat"
                and len(concat_layer.bottoms) == 3
                and len(followers) == 1
                and layers[followers[0]].layer_type in {"Split", "Slice"}
                and len(layers[followers[0]].tops) >= 3
            ):
                rope_end_candidates.append(index)
        if len(rope_end_candidates) == 1:
            rope_end_index = rope_end_candidates[0]
        elif not rope_end_candidates:
            # PNNX folds the small-grid rotary path into a final three-way
            # Concat instead of emitting the Split/Reshape pair used by the
            # larger 45x80 export.  The Concat output is still the packed
            # Q/K/V boundary consumed by the custom attention layer.
            folded_candidates = [
                index
                for index, layer in enumerate(layers)
                if index > rope_start_index
                and layer.layer_type == "Concat"
                and len(layer.bottoms) == 3
                and len(consumers.get(layer.tops[0], [])) == 1
                and layers[consumers[layer.tops[0]][0]].layer_type == "Reshape"
                and len(consumers.get(layers[consumers[layer.tops[0]][0]].tops[0], [])) == 1
                and layers[consumers[layers[consumers[layer.tops[0]][0]].tops[0]][0]].layer_type
                in {"Split", "Slice"}
            ]
            if len(folded_candidates) != 1:
                raise ValueError(
                    "expected one folded DiT MMRoPE output boundary, "
                    f"found {len(folded_candidates)}"
                )
            rope_end_index = folded_candidates[0]
        else:
            raise ValueError(
                "expected one DiT MMRoPE output reshape, "
                f"found {len(rope_end_candidates)}"
            )

    def resolve_split_source(blob: str) -> str:
        seen = set()
        while blob not in seen:
            seen.add(blob)
            producer_index = producers.get(blob)
            if producer_index is None:
                return blob
            producer = layers[producer_index]
            if producer.layer_type != "Split" or len(producer.bottoms) != 1:
                return blob
            blob = producer.bottoms[0]
        raise ValueError("cyclic Split aliases in ncnn param")

    def trace_unpack_path(selector_index: int) -> tuple[str, bool, set[int]]:
        selector = layers[selector_index]
        if len(selector.tops) != 1:
            raise ValueError("DiT AWA unpack index-select must have one output")
        current = selector.tops[0]
        path = {selector_index}
        reduced = False
        reshape_types = {"torch.index_select", "Reshape", "Tensor.reshape"}
        while True:
            next_layers = consumers.get(current, [])
            if len(next_layers) != 1:
                return current, reduced, path
            next_index = next_layers[0]
            next_layer = layers[next_index]
            if next_layer.layer_type in reshape_types and len(next_layer.tops) == 1:
                path.add(next_index)
                current = next_layer.tops[0]
                continue
            if next_layer.layer_type in {"Reduction", "torch.mean"} and len(next_layer.tops) == 1:
                path.add(next_index)
                return next_layer.tops[0], True, path
            return current, reduced, path

    selector_groups: dict[str, list[int]] = defaultdict(list)
    for index, layer in enumerate(layers):
        if index <= pack_index or layer.layer_type != "torch.index_select" or not layer.bottoms:
            continue
        selector_groups[resolve_split_source(layer.bottoms[0])].append(index)

    unpack_candidates = []
    for attended_blob, selectors in selector_groups.items():
        if len(selectors) != 2:
            continue
        paths = [trace_unpack_path(index) for index in selectors]
        reduced_paths = [path for path in paths if path[1]]
        plain_paths = [path for path in paths if not path[1]]
        if len(reduced_paths) == 1 and len(plain_paths) == 1:
            unpack_candidates.append((attended_blob, plain_paths[0], reduced_paths[0]))
    if len(unpack_candidates) != 1:
        raise ValueError(f"expected one DiT AWA unpack boundary, found {len(unpack_candidates)}")
    attended_blob, video_path, text_path = unpack_candidates[0]

    source_t, source_h, source_w = (int(item) for item in size)
    windows_t, windows_h, windows_w = (int(item) for item in windows)
    pack_line = (
        f"SeedVR2AWAPack awa_pack 2 2 {concat.bottoms[0]} {concat.bottoms[1]} "
        f"{pack.tops[0]} awa_cu_seqlens "
        f"0={source_t} 1={source_h} 2={source_w} "
        f"3={windows_t} 4={windows_h} 5={windows_w} "
        f"6={int(text_tokens)} 7={1 if shifted else 0}"
    )
    unpack_line = (
        f"SeedVR2AWAUnpack awa_unpack 1 2 {attended_blob} {video_path[0]} {text_path[0]} "
        f"0={source_t} 1={source_h} 2={source_w} "
        f"3={windows_t} 4={windows_h} 5={windows_w} "
        f"6={int(text_tokens)} 7={1 if shifted else 0}"
    )

    rope_line: str | None = None
    rope_removed: set[int] = set()
    attention_line: str | None = None
    attention_removed: set[int] = set()
    attention_start_index: int | None = None
    if rope_start_index is not None and rope_end_index is not None:
        rope_output = layers[rope_end_index].tops[0]
        rope_line = (
            f"SeedVR2MMRoPE mmrope 1 1 {pack.tops[0]} {rope_output} "
            f"0={source_t} 1={source_h} 2={source_w} "
            f"3={windows_t} 4={windows_h} 5={windows_w} "
            f"6={int(text_tokens)} 7={1 if shifted else 0} 8=126"
        )
        rope_removed = {
            index
            for index in range(rope_start_index, rope_end_index + 1)
            if layers[index].layer_type != "MemoryData"
        }

        attention_end_index = producers.get(attended_blob)
        attention_start_candidates = [
            index
            for index, layer in enumerate(layers)
            if index > rope_end_index
            and layer.layer_type == "Split"
            and layer.bottoms == (rope_output,)
            and len(layer.tops) >= 3
        ]
        if attention_end_index is not None and layers[attention_end_index].layer_type == "Concat":
            attention_layer = layers[attention_end_index]
            if len(attention_layer.bottoms) == 9:
                if len(attention_start_candidates) != 1:
                    raise ValueError(
                        "expected one DiT static window-attention Split, "
                        f"found {len(attention_start_candidates)}"
                    )
                attention_start_index = attention_start_candidates[0]
                if attention_start_index >= attention_end_index:
                    raise ValueError("DiT static window-attention boundaries are out of order")
                attention_line = (
                    f"SeedVR2WindowAttention awa_attention 1 1 {rope_output} {attended_blob} "
                    f"0={source_t} 1={source_h} 2={source_w} "
                    f"3={windows_t} 4={windows_h} 5={windows_w} "
                    f"6={int(text_tokens)} 7={1 if shifted else 0}"
                )
                attention_removed = {
                    index
                    for index in range(attention_start_index, attention_end_index + 1)
                    if layers[index].layer_type != "MemoryData"
                }
        elif tuple(size) != DIT_BLOCK_CONTRACT.source_shape:
            # On small fixed grids PNNX keeps the dense attention as a direct
            # MatMul chain. Its output is the Reshape immediately before the
            # two-way video/text Split; static index-select layers after that
            # split remain the validated unpack path.
            split_candidates = []
            for index in range(rope_end_index + 1, len(layers)):
                layer = layers[index]
                if layer.layer_type != "Split" or len(layer.tops) != 2:
                    continue
                producer_index = producers.get(layer.bottoms[0])
                if producer_index is None or producer_index <= rope_end_index:
                    continue
                if layers[producer_index].layer_type != "Reshape":
                    continue
                if any(
                    later.layer_type == "torch.index_select"
                    and later.bottoms
                    and later.bottoms[0] in layer.tops
                    for later in layers[index + 1 :]
                ):
                    split_candidates.append((producer_index, layer.bottoms[0]))
            if len(split_candidates) != 1:
                raise ValueError(
                    "expected one folded DiT attention output boundary, "
                    f"found {len(split_candidates)}"
                )
            attention_end_index, attended_blob = split_candidates[0]
            attention_start_index = rope_end_index + 1
            attention_line = (
                f"SeedVR2WindowAttention awa_attention 1 1 {rope_output} {attended_blob} "
                f"0={source_t} 1={source_h} 2={source_w} "
                f"3={windows_t} 4={windows_h} 5={windows_w} "
                f"6={int(text_tokens)} 7={1 if shifted else 0}"
            )
            attention_removed = {
                index
                for index in range(attention_start_index, attention_end_index + 1)
                if layers[index].layer_type != "MemoryData"
            }

    # The custom Pack consumes the video/text QKV tensors separately.  Remove
    # PNNX's pre-pack Concat as well; keeping it makes ncnn treat the two token
    # batches as one broadcast batch and rejects the fixed 64/58 lengths.
    removed = {pack_index} | video_path[2] | text_path[2] | rope_removed | attention_removed
    if tuple(size) != DIT_BLOCK_CONTRACT.source_shape:
        removed.add(pack_concat_index)
    unpack_insert_index = min(video_path[2] | text_path[2])
    video_batch_consumer_index: int | None = None
    for index, layer in enumerate(layers):
        if index in removed or video_path[0] not in layer.bottoms:
            continue
        if layer.layer_type == "InnerProduct":
            video_batch_consumer_index = index
            break
    video_batch_blob = "awa_video_batch"
    rewritten_layers: list[str] = []
    for index, layer in enumerate(layers):
        if index == pack_index:
            rewritten_layers.append(pack_line)
        if rope_line is not None and index == rope_start_index:
            rewritten_layers.append(rope_line)
        if attention_line is not None and index == attention_start_index:
            rewritten_layers.append(attention_line)
        if index == unpack_insert_index:
            rewritten_layers.append(unpack_line)
        if index == video_batch_consumer_index:
            rewritten_layers.append(
                f"Reshape reshape_awa_video_batch 1 1 {video_path[0]} {video_batch_blob} "
                f"0={int(DIT_BLOCK_CONTRACT.vid_dim)} 1={source_t * source_h * source_w} 12=233 13=0"
            )
        if index not in removed:
            line = fix_text_batch_axis(layer.line)
            if index == video_batch_consumer_index:
                fields = line.split()
                begin = 4
                for bottom_index in range(int(fields[2])):
                    if fields[begin + bottom_index] == video_path[0]:
                        fields[begin + bottom_index] = video_batch_blob
                line = " ".join(fields)
            rewritten_layers.append(line)

    rewritten_nodes = [parse_layer(line) for line in rewritten_layers]
    blob_count = len({top for layer in rewritten_nodes for top in layer.tops})
    param_path.write_text(
        "\n".join((lines[0], f"{len(rewritten_layers)} {blob_count}", *rewritten_layers)) + "\n"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def export_torchscript(checkpoint: Path, output_dir: Path, *, seed: int) -> tuple[Path, Path]:
    """Export block 0 and deterministic FP32 golden tensors on the CPU."""

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SeedVR2 DiT checkpoint not found: {checkpoint}")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(seed))
    block = load_fixed_dit_block(checkpoint).float().eval()
    vid = torch.randn(DIT_BLOCK_CONTRACT.video_tokens, DIT_BLOCK_CONTRACT.vid_dim)
    txt = torch.randn(DIT_BLOCK_CONTRACT.text_tokens, DIT_BLOCK_CONTRACT.txt_dim)
    emb = torch.randn(1, DIT_BLOCK_CONTRACT.emb_dim)
    with torch.inference_mode():
        vid_out, txt_out = block(vid, txt, emb)
        traced = torch.jit.trace(block, (vid, txt, emb), check_trace=True)

    model_path = output_dir / "dit_block_0.pt"
    golden_path = output_dir / "dit_block_0_golden.pt"
    manifest_path = output_dir / "dit_block_0_manifest.json"
    traced.save(str(model_path))
    torch.save(
        {
            "vid": vid,
            "txt": txt,
            "emb": emb,
            "vid_out": vid_out,
            "txt_out": txt_out,
        },
        golden_path,
    )
    manifest_path.write_text(
        json.dumps(
            {
                "block_index": 0,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": _sha256(checkpoint),
                "seed": int(seed),
                "dtype": "float32",
                "inputs": {
                    "vid": list(vid.shape),
                    "txt": list(txt.shape),
                    "emb": list(emb.shape),
                },
                "outputs": {
                    "vid_out": list(vid_out.shape),
                    "txt_out": list(txt_out.shape),
                },
                "tolerance": {
                    "atol": DIT_BLOCK_CONTRACT.fp32_atol,
                    "rtol": DIT_BLOCK_CONTRACT.fp32_rtol,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return model_path, manifest_path


def run_pnnx(pnnx: Path, model_path: Path, output_dir: Path) -> None:
    """Convert an exported block with one explicit fixed-shape invocation."""

    pnnx = Path(pnnx)
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    if not pnnx.is_file():
        raise FileNotFoundError(f"pnnx executable not found: {pnnx}")
    if not model_path.is_file():
        raise FileNotFoundError(f"TorchScript model not found: {model_path}")
    input_shape = (
        f"[{DIT_BLOCK_CONTRACT.video_tokens},{DIT_BLOCK_CONTRACT.vid_dim}],"
        f"[{DIT_BLOCK_CONTRACT.text_tokens},{DIT_BLOCK_CONTRACT.txt_dim}],"
        f"[1,{DIT_BLOCK_CONTRACT.emb_dim}]"
    )
    subprocess.run(
        [str(pnnx.resolve()), str(model_path.resolve()), f"inputshape={input_shape}"],
        cwd=output_dir.resolve(),
        check=True,
    )


def _update_manifest(manifest_path: Path, **values: object) -> None:
    manifest = json.loads(manifest_path.read_text())
    manifest.update(values)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("SEEDVR2_CKPT_DIR", "ckpts")) / "seedvr2_ema_3b.pth",
    )
    parser.add_argument("--pnnx", type=Path)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--size", type=_shape, default=DIT_BLOCK_CONTRACT.source_shape)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path, manifest_path = export_torchscript(args.checkpoint, args.output_dir, seed=args.seed)
    print(f"saved {model_path}")
    if args.pnnx is None:
        return

    run_pnnx(args.pnnx, model_path, args.output_dir)
    param_path = Path(args.output_dir).resolve() / "dit_block_0.ncnn.param"
    rewrite_ncnn_param(param_path)
    custom_layers = sum(
        line.startswith(
            (
                "SeedVR2AWAPack ",
                "SeedVR2AWAUnpack ",
                "SeedVR2MMRoPE ",
                "SeedVR2WindowAttention ",
            )
        )
        for line in param_path.read_text().splitlines()[2:]
    )
    _update_manifest(
        manifest_path,
        ncnn_param=str(param_path),
        ncnn_layers=int(param_path.read_text().splitlines()[1].split()[0]),
        awa_custom_layers=custom_layers,
    )
    print(f"rewritten {param_path}")


if __name__ == "__main__":
    main()
