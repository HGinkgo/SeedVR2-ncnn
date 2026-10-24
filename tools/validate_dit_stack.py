#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 SeedVR2-ncnn contributors
"""Validate exported SeedVR2 DiT stack graphs with the ncnn smoke runner.

The exporter writes one portable golden file containing the inputs and the
eager outputs for the input, embedding, all DiT blocks, and output graphs.
This tool exercises those same graphs through ncnn and records per-graph
maximum absolute error and process wall time for CPU and/or Vulkan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import struct
import subprocess
import tempfile
import time

import numpy as np


GOLDEN_MAGIC = b"SVR2F32\0"
GOLDEN_HEADER = struct.Struct("<8sII")
GOLDEN_RECORD = struct.Struct("<HBBQ4Q")


def _read_golden(path: Path) -> dict[str, np.ndarray]:
    records: dict[str, np.ndarray] = {}
    with path.open("rb") as source:
        magic, version, count = GOLDEN_HEADER.unpack(source.read(GOLDEN_HEADER.size))
        if magic != GOLDEN_MAGIC or version != 1:
            raise ValueError(f"unsupported golden header in {path}")
        for _ in range(count):
            name_bytes, rank, _reserved, element_count, *shape = GOLDEN_RECORD.unpack(
                source.read(GOLDEN_RECORD.size)
            )
            name = source.read(name_bytes).decode("utf-8")
            array = np.fromfile(source, dtype="<f4", count=element_count)
            if array.size != element_count:
                raise ValueError(f"truncated golden record {name!r} in {path}")
            records[name] = array.reshape(tuple(shape[:rank])).copy()
    return records


def _cases(records: dict[str, np.ndarray]) -> list[tuple[str, list[tuple[str, np.ndarray]], list[tuple[str, np.ndarray]]]]:
    cases = [
        (
            "dit_input",
            [("in0", records["input_video_patches"]), ("in1", records["input_text"])],
            [("out0", records["input_video_hidden"]), ("out1", records["input_text_hidden"])],
        ),
        ("dit_embedding", [("in0", records["timestep"])], [("out0", records["embedding"])]),
    ]
    for index in range(32):
        if index == 0:
            video, text = records["input_video_hidden"], records["input_text_hidden"]
        else:
            video = records[f"block_{index - 1:02d}_video"]
            text = records[f"block_{index - 1:02d}_text"]
        cases.append(
            (
                f"dit_block_{index:02d}",
                [("in0", video), ("in1", text), ("in2", records["embedding"])],
                [
                    ("out0", records[f"block_{index:02d}_video"]),
                    ("out1", records[f"block_{index:02d}_text"]),
                ],
            )
        )
    cases.append(
        (
            "dit_output",
            [("in0", records["block_31_video"]), ("in1", records["embedding"])],
            [("out0", records["output_video"])],
        )
    )
    return cases


def _run_graph(
    *,
    root: Path,
    runner: Path,
    mode: str,
    name: str,
    inputs: list[tuple[str, np.ndarray]],
    outputs: list[tuple[str, np.ndarray]],
    temporary: Path,
    verbose: bool,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    input_paths: list[Path] = []
    for index, (_blob, value) in enumerate(inputs):
        path = temporary / f"{name}.{mode}.input{index}.f32"
        value.astype("<f4", copy=False).tofile(path)
        input_paths.append(path)

    command: list[str] = []
    if mode == "vulkan":
        command.append(str(root / "tools" / "with-nvidia-vulkan-runtime.sh"))
    command.extend(
        [
            str(runner),
            mode,
            str(temporary / "graph.ncnn.param"),
            str(temporary / "graph.ncnn.bin"),
            "--multi",
            str(len(inputs)),
        ]
    )
    for (blob, value), path in zip(inputs, input_paths):
        shape = value.shape
        width = int(shape[-1]) if shape else 1
        height = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
        command.extend([blob, str(path), str(width), str(height)])

    errors: list[tuple[float, float, float]] = []
    process_seconds = 0.0
    for output_index, (blob, expected) in enumerate(outputs):
        output_path = temporary / f"{name}.{mode}.output{output_index}.f32"
        run_command = command + [blob, str(output_path)]
        started = time.perf_counter()
        completed = subprocess.run(
            run_command,
            stdout=subprocess.PIPE if not verbose else None,
            stderr=subprocess.STDOUT if not verbose else None,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(f"{name} {mode} {blob}: native runner exited {completed.returncode}\n"
                               f"{completed.stdout or 'see verbose runner output'}")
        process_seconds += time.perf_counter() - started
        if verbose and completed.stdout:
            print(completed.stdout, end="")
        actual = np.fromfile(output_path, dtype="<f4")
        if actual.size != expected.size:
            raise RuntimeError(
                f"{name} {mode} {blob}: expected {expected.size} values, got {actual.size}"
            )
        difference = np.abs(actual.reshape(expected.shape) - expected)
        expected_abs = np.abs(expected)
        limit = atol + rtol * float(np.max(expected_abs))
        if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)) or float(np.max(difference)) > limit:
            raise RuntimeError(f"{name} {mode} {blob}: max_abs={np.max(difference):.9g} exceeds {limit:.9g} "
                               "or contains nonfinite values")
        errors.append(
            (
                float(np.max(difference)),
                float(np.max(difference) / max(float(np.max(expected_abs)), 1.0e-12)),
                float(np.sqrt(np.mean(np.square(difference)))),
            )
        )

    result = {
        "name": name,
        "max_abs": max(error[0] for error in errors),
        "max_rel": max(error[1] for error in errors),
        "rmse": max(error[2] for error in errors),
        "wall_seconds": process_seconds,
    }
    print(
        f"{mode} {name} max_abs={result['max_abs']:.9g} "
        f"max_rel={result['max_rel']:.9g} rmse={result['rmse']:.9g} "
        f"wall_seconds={process_seconds:.3f}",
        flush=True,
    )
    return result


def validate(output_dir: Path, runner: Path, modes: tuple[str, ...], result_json: Path | None, verbose: bool,
             baseline_dir: Path | None = None, atol: float = 5e-4, rtol: float = 5e-5,
             graph_names: list[str] | None = None) -> dict[str, object]:
    output_dir = output_dir.resolve()
    runner = runner.resolve()
    if not runner.is_file():
        raise FileNotFoundError(f"ncnn runner not found: {runner}")
    records = _read_golden(output_dir / "dit_stack_golden.f32")
    cases = _cases(records)
    if graph_names is not None:
        available = {name for name, _, _ in cases}
        if not graph_names or len(set(graph_names)) != len(graph_names) or set(graph_names) - available:
            raise ValueError("--graphs must contain distinct names from the exported stack")
        cases = [case for case in cases if case[0] in graph_names]
    if not np.isfinite(atol) or not np.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("atol/rtol must be finite and nonnegative")
    result: dict[str, object] = {"graphs": len(cases), "modes": {}, "baseline": {},
                                 "bound": "max_abs <= atol + rtol * max(abs(reference))",
                                 "atol": atol, "rtol": rtol}
    root = Path(__file__).resolve().parents[1]
    manifest_path = output_dir / "dit_stack_manifest.json"
    if manifest_path.is_file():
        result["stack_manifest"] = json.loads(manifest_path.read_text())
        result["stack_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    result["repository_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    result["repository_dirty"] = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True).strip())
    result["system"] = platform.platform()
    with runner.open("rb") as source:
        result["runner_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
    result["torchscript_baseline"] = str(baseline_dir.resolve()) if baseline_dir else None
    with tempfile.TemporaryDirectory(prefix="seedvr2-dit-stack-") as temporary_name:
        temporary = Path(temporary_name)
        for mode in modes:
            rows: list[dict[str, object]] = []
            baseline_rows: list[dict[str, object]] = []
            for name, inputs, outputs in cases:
                source_param = output_dir / f"{name}.ncnn.param"
                source_bin = output_dir / f"{name}.ncnn.bin"
                if not source_param.is_file() or not source_bin.is_file():
                    raise FileNotFoundError(f"missing ncnn graph for {name} in {output_dir}")
                # The runner only reads graphs. Link them into its temporary
                # directory instead of duplicating tens of GB of weights for
                # every backend, keeping the validated source files intact.
                for source in (source_param, source_bin):
                    target = temporary / ("graph.ncnn.param" if source == source_param else "graph.ncnn.bin")
                    target.unlink(missing_ok=True)
                    target.symlink_to(source)
                rows.append(
                    _run_graph(
                        root=root,
                        runner=runner,
                        mode=mode,
                        name=name,
                        inputs=inputs,
                        outputs=outputs,
                        temporary=temporary,
                        verbose=verbose,
                        atol=atol,
                        rtol=rtol,
                    )
                )
                if baseline_dir is not None:
                    # Feed both frontends the identical eager intermediates,
                    # then compare native outputs, independent of either golden.
                    if not (baseline_dir / f"{name}.pt").is_file():
                        raise ValueError(f"TorchScript baseline source missing: {baseline_dir / name}")
                    baseline_temporary = temporary / "baseline"
                    baseline_temporary.mkdir(exist_ok=True)
                    for suffix in (".ncnn.param", ".ncnn.bin"):
                        target = baseline_temporary / f"graph{suffix}"
                        target.unlink(missing_ok=True)
                        target.symlink_to((baseline_dir / f"{name}{suffix}").resolve())
                    baseline_row = _run_graph(root=root, runner=runner, mode=mode, name=name,
                                              inputs=inputs, outputs=outputs, temporary=baseline_temporary,
                                              verbose=verbose, atol=atol, rtol=rtol)
                    paired_errors = []
                    for index, (_blob, expected) in enumerate(outputs):
                        actual = np.fromfile(temporary / f"{name}.{mode}.output{index}.f32", dtype="<f4")
                        baseline = np.fromfile(baseline_temporary / f"{name}.{mode}.output{index}.f32", dtype="<f4")
                        delta = np.abs(actual - baseline)
                        error = float(np.max(delta))
                        if error > atol + rtol * float(np.max(np.abs(baseline))):
                            raise RuntimeError(f"{name} {mode}: PT2/TorchScript output{index} max_abs={error}")
                        paired_errors.append(error)
                    baseline_row["paired_max_abs"] = max(paired_errors)
                    baseline_rows.append(baseline_row)
            result["modes"][mode] = rows
            if baseline_dir is not None:
                result["baseline"][mode] = baseline_rows
    result["status"] = "pass"
    if result_json is not None:
        result_json.parent.mkdir(parents=True, exist_ok=True)
        result_json.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="export_dit_stack output directory")
    parser.add_argument("--runner", type=Path, required=True, help="built seedvr2-pt2-smoke executable")
    parser.add_argument("--mode", choices=("cpu", "vulkan", "both"), default="both")
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--verbose", action="store_true", help="show each ncnn subprocess output")
    parser.add_argument("--baseline-dir", type=Path, help="TorchScript graphs to run on the identical inputs")
    parser.add_argument("--graphs", nargs="+", help="validate only named graphs; default validates all 35")
    parser.add_argument("--atol", type=float, default=5e-4)
    parser.add_argument("--rtol", type=float, default=5e-5,
                        help="relative to each output's maximum absolute reference value")
    args = parser.parse_args()
    modes = ("cpu", "vulkan") if args.mode == "both" else (args.mode,)
    validate(args.output_dir, args.runner, modes, args.result_json, args.verbose,
             args.baseline_dir, args.atol, args.rtol, args.graphs)


if __name__ == "__main__":
    main()
