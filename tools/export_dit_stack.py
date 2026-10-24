"""Export a fixed SeedVR2 DiT stack through explicit TorchScript or PT2 frontends."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.export_dit_block import (
    normalize_dynamic_awa_template,
    normalize_dynamic_dit_gemm_rows,
    rewrite_ncnn_param,
)
from tools.reference.dit_stack import (
    DIT_STACK_CONTRACT,
    DIT_OUTPUT_MODULATION,
    FixedDitEmbedding,
    FixedDitInput,
    FixedDitOutput,
    iter_loaded_blocks,
    load_dit_state,
    make_dit_stack_contract,
)
from tools.reference.portable_f32 import GOLDEN_MAGIC, GOLDEN_VERSION, write_portable_golden
from tools.reference.dit_block import DIT_LAST_BLOCK_CONTRACT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _trace_save(model: torch.nn.Module, inputs: tuple[torch.Tensor, ...], path: Path) -> None:
    with torch.inference_mode():
        traced = torch.jit.trace(model, inputs, check_trace=True)
    traced.save(str(path))


def _run_pnnx(pnnx: Path, model_path: Path, output_dir: Path, input_shape: str) -> None:
    subprocess.run(
        [
            str(Path(pnnx).resolve()),
            str(Path(model_path).resolve()),
            f"inputshape={input_shape}",
            "fp16=0",
        ],
        cwd=Path(output_dir).resolve(),
        check=True,
    )


def _custom_layer_count(param_path: Path) -> int:
    prefixes = (
        "SeedVR2AWAPack ",
        "SeedVR2AWAUnpack ",
        "SeedVR2MMRoPE ",
        "SeedVR2WindowAttention ",
    )
    return sum(line.startswith(prefixes) for line in param_path.read_text().splitlines()[2:])


def _attach_ncnn_artifacts(
    manifest: dict[str, object], output_dir: Path, graph_records: list[dict[str, object]], contract
) -> None:
    artifacts: list[dict[str, object]] = []
    for name in contract.graph_names:
        param_path = output_dir / f"{name}.ncnn.param"
        bin_path = output_dir / f"{name}.ncnn.bin"
        if not param_path.is_file() or not bin_path.is_file():
            continue
        record = next((graph for graph in graph_records if graph["name"] == name), {"name": name})
        record["ncnn_param"] = str(param_path.resolve())
        record["ncnn_bin"] = str(bin_path.resolve())
        record["ncnn_layers"] = int(param_path.read_text().splitlines()[1].split()[0])
        record["awa_custom_layers"] = _custom_layer_count(param_path)
        artifacts.append(record)
    if artifacts:
        manifest["graphs"] = artifacts


def _shape3(value: str) -> tuple[int, int, int]:
    try:
        parts = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected T,H,W positive integers") from exc
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError("expected T,H,W positive integers")
    return parts


def export_stack(
    checkpoint: Path,
    output_dir: Path,
    pnnx: Path | None,
    seed: int,
    text_tokens: int = DIT_STACK_CONTRACT.text_tokens,
    source_shape: tuple[int, int, int] = DIT_STACK_CONTRACT.source_shape,
    format: str = "torchscript",
    resume: bool = False,
) -> Path:
    if format not in ("torchscript", "pt2"):
        raise ValueError("format must be torchscript or pt2")
    if resume and format != "pt2":
        raise ValueError("--resume requires --format pt2")
    checkpoint = Path(checkpoint)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = make_dit_stack_contract(text_tokens, source_shape=source_shape)
    if not resume and any((output_dir / f"{name}{suffix}").exists()
                          for name in contract.graph_names
                          for suffix in (".pt", ".pt2", ".ncnn.param", ".ncnn.bin")):
        raise FileExistsError("stack artifacts already exist; choose a new directory or use PT2 --resume")
    if resume and (output_dir / "dit_output.pt2").is_file():
        metadata = output_dir / "dit_output.manifest.json"
        if (not metadata.is_file()
                or json.loads(metadata.read_text()).get("output_modulation") != DIT_OUTPUT_MODULATION):
            raise ValueError("cannot resume obsolete output Ada; use a new directory and re-export dit_output")
    last_name = f"dit_block_{contract.block_count - 1:02d}"
    if resume and (output_dir / f"{last_name}.pt2").is_file():
        metadata = output_dir / f"{last_name}.manifest.json"
        if (not metadata.is_file() or json.loads(metadata.read_text()).get("last_block_contract")
                != DIT_LAST_BLOCK_CONTRACT):
            raise ValueError("cannot resume obsolete last block; re-export the video-only Ada/MLP boundary")
    checkpoint_sha256 = _sha256(checkpoint)
    export_records: dict[str, dict[str, object]] = {}

    def save_graph(model, inputs, name, block_index=None):
        if format == "torchscript":
            path = output_dir / f"{name}.pt"
            _trace_save(model, inputs, path)
            export_records[name] = {"frontend": "torchscript", "torchscript": str(path),
                                    "source_sha256": _sha256(path)}
            return
        from tools.convert_pt2 import convert

        path = output_dir / f"{name}.pt2"
        metadata_path = path.with_suffix(".manifest.json")
        reused = resume and path.is_file()
        if reused:
            if not metadata_path.is_file():
                raise ValueError(f"cannot resume PT2 source without its manifest: {path}")
            previous = json.loads(metadata_path.read_text())
            expected = {"exporter": "seedvr2-pt2-fixtures-v2",
                        "model": f"dit_block{block_index}_full" if block_index is not None else name,
                        "block_index": block_index, "source_shape": list(contract.source_shape),
                        "text_tokens": contract.text_tokens, "checkpoint_sha256": checkpoint_sha256,
                        "torch_version": torch.__version__, "pt2_sha256": _sha256(path)}
            if name == "dit_output":
                expected["output_modulation"] = DIT_OUTPUT_MODULATION
            if name == last_name:
                expected["last_block_contract"] = DIT_LAST_BLOCK_CONTRACT
            if any(previous.get(key) != value for key, value in expected.items()):
                raise ValueError(f"cannot resume incompatible PT2 source: {path}")
        else:
            exported = torch.export.export(model, inputs)
            torch.export.save(exported, path)
        # Reuse the converter's reviewed fixed-block AWA contract. Each graph
        # retains its own source hash and conversion record, preventing a
        # TorchScript artifact from being labelled as PT2 by directory name.
        metadata = {
            "exporter": "seedvr2-pt2-fixtures-v2",
            "model": f"dit_block{block_index}_full" if block_index is not None else name,
            "block_index": block_index,
            "pt2_sha256": _sha256(path),
            "source_shape": list(contract.source_shape),
            "text_tokens": contract.text_tokens,
            "checkpoint_sha256": checkpoint_sha256,
            "torch_version": torch.__version__,
        }
        if name == "dit_output":
            metadata["output_modulation"] = DIT_OUTPUT_MODULATION
        if name == last_name:
            metadata["last_block_contract"] = DIT_LAST_BLOCK_CONTRACT
        if not reused:
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        record = {"frontend": "pt2", "pt2": str(path), "source_sha256": metadata["pt2_sha256"]}
        if pnnx is not None:
            conversion_path = output_dir / f"{name}.conversion.json"
            conversion = json.loads(conversion_path.read_text()) if reused and conversion_path.is_file() else {}
            complete = (conversion.get("status") == "success"
                        and conversion.get("frontend") == "independent"
                        and conversion.get("frontend_contract", {}).get("pt2_sha256") == metadata["pt2_sha256"]
                        and all((output_dir / f"{name}{suffix}").is_file()
                                for suffix in (".ncnn.param", ".ncnn.bin")))
            if not complete:
                if conversion_path.is_file() and resume:
                    # Retain the failed diagnostic when retrying a conversion.
                    shutil.copy2(conversion_path, conversion_path.with_suffix(".previous.json"))
                conversion = convert(path, Path(pnnx), output_dir,
                                     overwrite=resume,
                                     rewrite_dit_block=block_index is not None,
                                     size=contract.source_shape, windows=contract.window_shape,
                                     text_tokens=contract.text_tokens,
                                     shifted=contract.block_shifted(block_index) if block_index is not None else False)
            record["conversion"] = str(output_dir / f"{name}.conversion.json")
            record["conversion_status"] = conversion["status"]
        export_records[name] = record
    torch.manual_seed(int(seed))
    state = load_dit_state(checkpoint)

    video_patches = torch.randn(contract.video_patch_shape)
    text_input = torch.randn(contract.text_input_shape)
    timestep = torch.tensor([0.5], dtype=torch.float32)

    input_model = FixedDitInput(state, contract=contract).float().eval()
    with torch.inference_mode():
        video_hidden, text_hidden = input_model(video_patches, text_input)
    save_graph(input_model, (video_patches, text_input), "dit_input")

    embedding_model = FixedDitEmbedding(state, contract=contract).float().eval()
    with torch.inference_mode():
        embedding = embedding_model(timestep)
    save_graph(embedding_model, (timestep,), "dit_embedding")

    graph_records: list[dict[str, object]] = [
        {
            "name": "dit_input",
            **export_records["dit_input"],
            "inputs": [list(video_patches.shape), list(text_input.shape)],
            "outputs": [list(video_hidden.shape), list(text_hidden.shape)],
        },
        {
            "name": "dit_embedding",
            **export_records["dit_embedding"],
            "inputs": [list(timestep.shape)],
            "outputs": [list(embedding.shape)],
        },
    ]

    golden_blocks: list[dict[str, object]] = []
    golden_records: list[tuple[str, torch.Tensor]] = [
        ("input_video_patches", video_patches),
        ("input_text", text_input),
        ("timestep", timestep),
        ("input_video_hidden", video_hidden),
        ("input_text_hidden", text_hidden),
        ("embedding", embedding),
    ]
    for block_index, block in iter_loaded_blocks(state, contract=contract):
        block_name = f"dit_block_{block_index:02d}"
        block_input = (video_hidden.detach().clone(), text_hidden.detach().clone(), embedding.detach().clone())
        with torch.inference_mode():
            video_hidden, text_hidden = block(*block_input)
        save_graph(block, block_input, block_name, block_index)
        graph_records.append({"name": block_name, **export_records[block_name],
                              "inputs": [list(value.shape) for value in block_input],
                              "outputs": [list(video_hidden.shape), list(text_hidden.shape)]})
        golden_blocks.append(
            {
                "index": block_index,
                "shifted": contract.block_shifted(block_index),
                "video_output_shape": list(video_hidden.shape),
                "text_output_shape": list(text_hidden.shape),
            }
        )
        golden_records.extend(
            [
                (f"block_{block_index:02d}_video", video_hidden),
                (f"block_{block_index:02d}_text", text_hidden),
            ]
        )
        del block

    output_model = FixedDitOutput(state, contract=contract).float().eval()
    with torch.inference_mode():
        output_tokens = output_model(video_hidden, embedding)
    save_graph(output_model, (video_hidden.detach().clone(), embedding.detach().clone()), "dit_output")
    golden_records.append(("output_video", output_tokens))
    golden_f32_path = output_dir / "dit_stack_golden.f32"
    golden_f32_layout = write_portable_golden(golden_f32_path, golden_records)
    torch.save(
        {
            "video_patches": video_patches,
            "text_input": text_input,
            "timestep": timestep,
            "embedding": embedding,
            "video_output": output_tokens,
            "video_hidden": video_hidden,
            "text_hidden": text_hidden,
            "block_count": len(golden_blocks),
        },
        output_dir / "dit_stack_golden.pt",
    )
    graph_records.append(
        {
            "name": "dit_output",
            **export_records["dit_output"],
            "inputs": [list(video_hidden.shape), list(embedding.shape)],
            "outputs": [list(output_tokens.shape)],
        }
    )

    manifest = {
        "kind": "seedvr2_dit_stack",
        "frontend": format,
        "output_modulation": DIT_OUTPUT_MODULATION,
        "last_block_contract": DIT_LAST_BLOCK_CONTRACT,
        "torch_version": torch.__version__,
        "source_shape": list(contract.source_shape),
        "window_shape": list(contract.window_shape),
        "text_tokens": contract.text_tokens,
        "block_count": contract.block_count,
        "seed": int(seed),
        "timestep": float(timestep.item()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "graphs": graph_records,
        "blocks": golden_blocks,
        "golden": str((output_dir / "dit_stack_golden.pt").resolve()),
        "golden_f32": str(golden_f32_path.resolve()),
        "golden_f32_sha256": _sha256(golden_f32_path),
        "golden_f32_layout": golden_f32_layout,
        "dtype": "float32",
    }
    manifest_path = output_dir / "dit_stack_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    if pnnx is None or format == "pt2":
        _attach_ncnn_artifacts(manifest, output_dir, graph_records, contract)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest_path

    _run_pnnx(
        pnnx,
        output_dir / "dit_input.pt",
        output_dir,
        f"[{contract.video_tokens},{contract.patch_width}],[{contract.text_tokens},5120]",
    )
    normalize_dynamic_dit_gemm_rows(output_dir / "dit_input.ncnn.param")
    _run_pnnx(pnnx, output_dir / "dit_embedding.pt", output_dir, "[1]")
    for block_index in range(contract.block_count):
        block_name = f"dit_block_{block_index:02d}"
        _run_pnnx(
            pnnx,
            output_dir / f"{block_name}.pt",
            output_dir,
            f"[{contract.video_tokens},2560],[{contract.text_tokens},2560],[1,15360]",
        )
        param_path = output_dir / f"{block_name}.ncnn.param"
        rewrite_ncnn_param(
            param_path,
            size=contract.source_shape,
            windows=contract.window_shape,
            text_tokens=contract.text_tokens,
            shifted=contract.block_shifted(block_index),
        )
        normalize_dynamic_awa_template(param_path)
    _run_pnnx(
        pnnx,
        output_dir / "dit_output.pt",
        output_dir,
        f"[{contract.video_tokens},2560],[1,15360]",
    )
    normalize_dynamic_dit_gemm_rows(output_dir / "dit_output.ncnn.param")

    _attach_ncnn_artifacts(manifest, output_dir, graph_records, contract)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("SEEDVR2_CKPT_DIR", "ckpts")) / "seedvr2_ema_3b.pth",
    )
    parser.add_argument("--pnnx", type=Path)
    parser.add_argument("--format", choices=("torchscript", "pt2"), default="torchscript",
                        help="source format; PT2 uses the independent frontend, TorchScript remains the default")
    parser.add_argument("--resume", action="store_true",
                        help="reuse hash/metadata-checked PT2 sources and successful conversions; recompute eager golden")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--text-tokens", type=int, choices=(58, 64), default=58)
    parser.add_argument("--source-shape", type=_shape3, default=DIT_STACK_CONTRACT.source_shape, metavar="T,H,W")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = export_stack(
        args.checkpoint,
        args.output_dir,
        args.pnnx,
        args.seed,
        args.text_tokens,
        source_shape=args.source_shape,
        format=args.format,
        resume=args.resume,
    )
    print(f"saved {manifest}")


if __name__ == "__main__":
    main()
