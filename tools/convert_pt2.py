# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 SeedVR2-ncnn contributors

"""Convert a restricted PT2 graph with the independent PNNX frontend."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.export_dit_block import rewrite_ncnn_param as rewrite_dit_block_param
from tools.export_vae import rewrite_ncnn_param as rewrite_vae_param
from tools.reference.dit_block import DIT_BLOCK_CONTRACT


def validate_input_contract(pt2: Path) -> None:
    """Reject unvalidated calling conventions before producing native artifacts."""
    import torch

    program = torch.export.load(pt2)
    for spec in program.graph_signature.input_specs:
        if spec.kind.name == "USER_INPUT" and not isinstance(spec.arg, torch.export.graph_signature.TensorArgument):
            raise ValueError("PT2 input contract: every user input must be a tensor")
    for spec in program.graph_signature.output_specs:
        if spec.kind.name != "USER_OUTPUT":
            raise ValueError(f"PT2 output contract: mutation/state output {spec.kind.name} is unsupported")
        if not isinstance(spec.arg, torch.export.graph_signature.TensorArgument):
            raise ValueError("PT2 output contract: every user output must be a tensor")
    inputs = program.call_spec.in_spec
    if inputs is None or inputs.type is not tuple or inputs.num_children != 2:
        raise ValueError("PT2 input structure must be a flat positional tensor tuple")
    args, kwargs = inputs.children()
    if (args.type is not tuple or kwargs.type is not dict or kwargs.num_children
            or any(not child.is_leaf() for child in args.children())):
        raise ValueError("PT2 input structure: nested or keyword inputs are unsupported; use positional tensors")
    outputs = program.call_spec.out_spec
    if outputs is None or (not outputs.is_leaf() and (
            outputs.type not in (tuple, list) or any(not child.is_leaf() for child in outputs.children()))):
        raise ValueError("PT2 output structure: expected one tensor or a flat tensor tuple/list")
    for node in program.graph.nodes:
        if node.op == "call_function" and isinstance(node.target, torch._ops.HigherOrderOperator):
            raise ValueError(f"PT2 control flow/higher-order operator is unsupported: {node.target}")


def validate_native_layers(param: Path) -> None:
    """Do not publish graphs containing an unlowered PyTorch/custom operator."""
    registry = Path(__file__).resolve().parents[1] / "third_party/ncnn/src/CMakeLists.txt"
    if not registry.is_file():
        raise ValueError("ncnn layer registry is missing; initialize the existing ncnn submodule")
    supported = set(re.findall(r"ncnn_add_layer\((\w+)", registry.read_text()))
    supported.update((
        "SeedVR2AWAPack", "SeedVR2MMRoPE", "SeedVR2WindowAttention", "SeedVR2AWAUnpack",
        "SeedVR2TemporalPad", "SeedVR2CausalConv3D", "SeedVR2DepthToSpace",
        "SeedVR2PointwiseConv3D",
    ))
    lines = [line for line in param.read_text().splitlines() if line.strip()]
    if len(lines) < 2 or lines[0].strip() != "7767517":
        raise ValueError("invalid ncnn param header")
    counts = lines[1].split()
    if len(counts) != 2 or not all(value.isdigit() for value in counts) or int(counts[0]) != len(lines) - 2:
        raise ValueError("invalid ncnn param layer count")
    unsupported = [line.split()[0] for line in lines[2:] if line.split()[0] not in supported]
    if unsupported:
        raise ValueError("unsupported/unlowered ncnn layer types: " + ", ".join(sorted(set(unsupported))))


def _existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file not found: {path}")
    return path


def _positive_shape(value: str) -> tuple[int, int, int]:
    try:
        parts = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected T,H,W") from exc
    if len(parts) != 3 or any(item <= 0 for item in parts):
        raise argparse.ArgumentTypeError("expected three positive comma-separated integers")
    return parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pt2", type=_existing_file)
    parser.add_argument("--pnnx", type=_existing_file, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true", help="replace existing generated artifacts; never reuse them as conversion evidence")
    parser.add_argument("--rewrite-dit-block", action="store_true",
                        help="replace the fixed checkpoint DiT AWA boundary with validated custom layers")
    parser.add_argument("--rewrite-vae", action="store_true",
                        help="lower the fixed one-frame VAE PT2 decoder boundary to SeedVR2 custom layers")
    parser.add_argument("--fp16-weights", action="store_true", help="opt into lossy ncnn weight conversion (default: float32)")
    parser.add_argument(
        "--allow-diagnostics",
        action="store_true",
        help="allow PNNX unsupported-operation diagnostics after manual review",
    )
    parser.add_argument("--size", type=_positive_shape, default=(2, 19, 23))
    parser.add_argument("--windows", type=_positive_shape, default=(4, 3, 3))
    parser.add_argument("--text-tokens", type=int, default=5)
    parser.add_argument("--shifted", action="store_true")
    return parser.parse_args()


def convert(
    pt2: Path,
    pnnx: Path,
    output_dir: Path | None = None,
    *,
    rewrite_dit_block: bool = False,
    rewrite_vae: bool = False,
    fp16_weights: bool = False,
    allow_diagnostics: bool = False,
    overwrite: bool = False,
    size: tuple[int, int, int] = (2, 19, 23),
    windows: tuple[int, int, int] = (4, 3, 3),
    text_tokens: int = 5,
    shifted: bool = False,
) -> dict[str, object]:
    pt2 = pt2.resolve()
    pnnx = pnnx.resolve()
    validate_input_contract(pt2)
    if rewrite_dit_block and rewrite_vae:
        raise ValueError("choose one rewrite mode")
    if rewrite_dit_block and size == (2, 19, 23) and text_tokens == 5:
        # Default generic graph dimensions to the reviewed fixed block contract.
        size = DIT_BLOCK_CONTRACT.source_shape
        windows = DIT_BLOCK_CONTRACT.window_shape
        text_tokens = DIT_BLOCK_CONTRACT.text_tokens
    if rewrite_dit_block:
        manifest_path = pt2.with_suffix(".manifest.json")
        if not manifest_path.is_file():
            raise ValueError("DiT block rewrite requires the manifest from export_pt2_seedvr2.py")
        manifest = json.loads(manifest_path.read_text())
        model_name = str(manifest.get("model", ""))
        if (manifest.get("exporter") not in ("seedvr2-pt2-fixtures-v1", "seedvr2-pt2-fixtures-v2")
                or re.fullmatch(r"dit_block\d+_full", model_name) is None
                or int(manifest.get("block_index", -1)) < 0
                or manifest.get("pt2_sha256") != hashlib.sha256(pt2.read_bytes()).hexdigest()):
            raise ValueError("DiT block rewrite manifest does not identify this exported fixture")
        if (manifest.get("source_shape") != list(size)
                or manifest.get("text_tokens") != text_tokens):
            raise ValueError("DiT block rewrite shape metadata does not match the exporter manifest")
    destination = (output_dir or pt2.parent).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    local_pt2 = destination / pt2.name
    stem = local_pt2.with_suffix("")
    def artifact(suffix: str) -> Path:
        return stem.with_name(stem.name + suffix)

    expected = [
        artifact(".pnnx.param"),
        artifact(".pnnx.bin"),
        artifact("_pnnx.py"),
        artifact(".ncnn.param"),
        artifact(".ncnn.bin"),
        artifact("_ncnn.py"),
    ]
    generated = expected + [artifact(".raw.ncnn.param"), artifact(".raw.ncnn.bin"),
                            artifact(".frontend.pnnx.param"), artifact(".frontend.pnnx.bin"),
                            artifact(".frontend.pnnx.json")]
    collisions = [str(path) for path in generated if path.exists()]
    if collisions and not overwrite:
        raise ValueError("generated artifacts already exist; choose a new output directory or --overwrite: "
                         + ", ".join(collisions))
    if local_pt2 != pt2 and local_pt2.exists() and not overwrite:
        if hashlib.sha256(local_pt2.read_bytes()).digest() != hashlib.sha256(pt2.read_bytes()).digest():
            raise ValueError("destination PT2 differs from the source; choose a new directory or --overwrite")
    if any(path.exists() and not path.is_file() for path in generated):
        raise ValueError("an artifact path is not a regular file; output directory was left unchanged")
    command = [str(pnnx), str(local_pt2), f"fp16={int(fp16_weights)}"]
    source = next((parent for parent in pnnx.parents
                   if (parent / "tools/pnnx/python/pt2_to_pnnx.py").is_file()), None)
    if source is None:
        raise ValueError("independent PT2 frontend missing beside PNNX; run prepare_pnnx_pt2.sh")
    frontend_path = source / "tools/pnnx/python/pt2_to_pnnx.py"
    command[1] = str(artifact(".frontend.pnnx.param"))
    for key, path in zip(("pnnxparam", "pnnxbin", "pnnxpy", "ncnnparam", "ncnnbin", "ncnnpy"), expected):
        command.append(f"{key}={path}")
    record_path = destination / f"{stem.name}.conversion.json"
    log_path = destination / f"{stem.name}.conversion.log"
    record = {
        "pt2": str(local_pt2),
        "frontend": "independent",
        "frontend_source": str(frontend_path),
        "pnnx": str(pnnx),
        "command": command,
        "status": "failed",
        "returncode": None,
        "artifacts": [],
        "log": str(log_path),
        "rewrite_dit_block": rewrite_dit_block,
        "rewrite_vae": rewrite_vae,
        "fp16_weights": fp16_weights,
        "allow_diagnostics": allow_diagnostics,
        "diagnostics": [],
    }
    for path in generated:
        path.unlink(missing_ok=True)
    log_path.write_text("")
    try:
        if local_pt2 != pt2:
            shutil.copy2(pt2, local_pt2)
        spec = importlib.util.spec_from_file_location("pnnx_pt2_frontend", frontend_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        record["frontend_contract"] = module.lower(local_pt2, artifact(".frontend.pnnx.param"),
                                                   artifact(".frontend.pnnx.bin"))
        completed = subprocess.run(
            command, cwd=destination, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        )
        log = completed.stdout
        log_path.write_text(log)
        print(log, end="")
        record["returncode"] = completed.returncode
        diagnostics = [line for line in log.splitlines() if any(
            token in line.lower() for token in ("warning", "not supported", "ignore torch"))]
        record["diagnostics"] = diagnostics
        if completed.returncode != 0:
            raise RuntimeError(f"pnnx failed with exit code {completed.returncode}; see {log_path}")
        missing = [str(path) for path in expected if not path.is_file()]
        if missing:
            raise RuntimeError("pnnx completed without expected artifacts: " + ", ".join(missing))
        if rewrite_dit_block:
            accepted = re.compile(r"^ignore torch\.index_select index_select(?:_\d+)? param dim=0$")
            unexpected = [line for line in diagnostics if not accepted.fullmatch(line.strip())]
            if unexpected:
                raise RuntimeError("PNNX reported unreviewed diagnostics for the DiT block: "
                                   + " | ".join(unexpected))
        elif diagnostics and not allow_diagnostics:
            raise RuntimeError(
                "PNNX reported diagnostics; inspect the conversion JSON or rerun "
                "with --allow-diagnostics after reviewing them: " + " | ".join(diagnostics))
        if rewrite_dit_block or rewrite_vae:
            shutil.copy2(artifact(".ncnn.param"), artifact(".raw.ncnn.param"))
            shutil.copy2(artifact(".ncnn.bin"), artifact(".raw.ncnn.bin"))
            if rewrite_dit_block:
                rewrite_dit_block_param(artifact(".ncnn.param"), size=size, windows=windows,
                                        text_tokens=text_tokens, shifted=shifted)
            else:
                rewrite_vae_param(artifact(".ncnn.param"))
        validate_native_layers(artifact(".ncnn.param"))
        record["status"] = "success"
        record["artifacts"] = [str(path) for path in generated if path.is_file()]
    except (OSError, RuntimeError, ValueError) as exc:
        record["error"] = str(exc)
        for path in generated:
            path.unlink(missing_ok=True)
        raise
    finally:
        record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main() -> None:
    args = parse_args()
    if args.text_tokens <= 0:
        raise SystemExit("--text-tokens must be positive")
    try:
        record = convert(
            args.pt2,
            args.pnnx,
            args.output_dir,
            rewrite_dit_block=args.rewrite_dit_block,
            rewrite_vae=args.rewrite_vae,
            fp16_weights=args.fp16_weights,
            allow_diagnostics=args.allow_diagnostics,
            overwrite=args.overwrite,
            size=args.size,
            windows=args.windows,
            text_tokens=args.text_tokens,
            shifted=args.shifted,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
