#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 SeedVR2-ncnn contributors
"""Run fixed-256 Vulkan image/CFG cases sequentially and retain measured evidence.

Conversion/testing helper only; the published C++ runtime needs no Python.
GPU memory is sampled for the whole selected device, not an exact process peak.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def capture(command: list[str]) -> str:
    return subprocess.check_output(command, cwd=ROOT, text=True).strip()


def gpu_memory(gpu_id: int) -> int | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        return int(capture(["nvidia-smi", "-i", str(gpu_id), "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"]))
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def run_case(binary: Path, package: Path, image: Path, output_dir: Path,
             gpu_id: int, name: str, steps: int, scale: float, seed: int | None = None,
             precision: str = "bf16") -> dict:
    if seed is not None and not 0 <= seed <= 0xffffffff:
        raise ValueError("seed must be in [0, 4294967295]")
    if precision not in {"bf16", "fp32"}:
        raise ValueError("precision must be bf16 or fp32")
    output = output_dir / f"{name}.png"
    log = output_dir / f"{name}.log"
    command = [str(ROOT / "tools/with-nvidia-vulkan-runtime.sh"), str(binary),
               "--model-dir", str(package), "--input", str(image), "--output", str(output),
               "--width", "256", "--height", "256", "--gpu-id", str(gpu_id),
               "--steps", str(steps), "--cfg-scale", str(scale), "--profile"]
    if seed is not None:
        command += ["--seed", str(seed)]
    if precision == "fp32":
        command += ["--precision", precision]
    initial_memory = gpu_memory(gpu_id)
    samples = [] if initial_memory is None else [initial_memory]
    started = time.perf_counter()
    with log.open("w") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        while True:
            try:
                process.wait(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                sample = gpu_memory(gpu_id)
                if sample is not None:
                    samples.append(sample)
    row = {"name": name, "steps": steps, "cfg_scale": scale, "precision": precision, "command": command,
           "exit_code": process.returncode, "process_wall_seconds": time.perf_counter() - started,
           "package_manifest_sha256": sha256(package / "manifest.sha256"),
           "device_memory_initial_mib": initial_memory,
           "device_memory_sampled_peak_mib": max(samples) if samples else None,
           "device_memory_sample_count": len(samples), "log": str(log)}
    if seed is not None:
        row.update(seed=seed, noise_algorithm="mt19937-box-muller-v1")
    text = log.read_text()
    fingerprints = re.findall(
        r"profile name=tensor-fingerprint phase=(\S+) frame=(\d+) fnv1a64=([0-9a-f]{16}) finite=([01])", text)
    row["tensor_fingerprints"] = {f"{phase}:{frame}": digest for phase, frame, digest, _ in fingerprints}
    profile = re.search(r"profile name=total ms=([0-9.]+) peak-rss-mib=([0-9]+)", text)
    if profile:
        row.update(profile_total_ms=float(profile[1]), process_peak_rss_mib=int(profile[2]))
    if process.returncode != 0 or "seedvr2-image-inference: ok" not in text or not output.is_file():
        row["status"] = "failed"
        return row
    phases = ["condition-latent", "denoised-latent", "reconstruction"]
    if seed is not None:
        phases.append("initial-noise")
    if (set(row["tensor_fingerprints"]) != {f"{phase}:0" for phase in phases}
            or len(fingerprints) != len(phases) or any(finite != "1" for _, _, _, finite in fingerprints)):
        row["status"] = "failed-tensor-contract"
        return row
    if seed is not None and (f"stage=noise-create distribution=normal seed={seed} frame=0 "
                             "algorithm=mt19937-box-muller-v1") not in text:
        row["status"] = "failed-noise-contract"
        return row
    # The negative graph must only run when guidance is enabled. Every
    # multi-step case must reach the last scheduled DiT step.
    if precision == "fp32":
        events = re.findall(r"stage=(load|release)-dit-branch step=(\d+) branch=(\d+)", text)
        expected = [(event, str(step), str(branch)) for step in range(steps)
                    for branch in range(2 if scale > 1 else 1) for event in ("load", "release")]
        decode = text.find("stage=load-decode")
        row["sequential_cfg_events"] = events
        last_release = text.rfind("stage=release-dit-branch")
        row["cfg_stacks_released_before_decode"] = last_release >= 0 and decode > last_release
        if events != expected or not row["cfg_stacks_released_before_decode"]:
            row["status"] = "failed-runtime-contract"
            return row
    else:
        negative_ran = "stage=load-dit-stack-negative" in text
        released = text.find("stage=release-dit-stacks-for-decode")
        decode = text.find("stage=load-decode")
        row["cfg_stacks_released_before_decode"] = released >= 0 and decode > released
        if (negative_ran != (scale > 1)
                or (scale > 1 and not row["cfg_stacks_released_before_decode"])
                or (steps > 1 and f"stage=dit-stack step={steps - 1}" not in text)):
            row["status"] = "failed-runtime-contract"
            return row
    row.update(status="pass", output=str(output), output_sha256=sha256(output))
    print(f"{name}: pass, total={row.get('profile_total_ms')} ms, "
          f"sampled device memory={row['device_memory_sampled_peak_mib']} MiB", flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/seedvr2-ncnn")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--baseline-package", type=Path, help="optional TorchScript positive-only baseline")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="new evidence directory")
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()
    if args.gpu_id < 0:
        parser.error("--gpu-id must select one explicit GPU")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = {"date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "repository_commit": capture(["git", "rev-parse", "HEAD"]),
              "repository_dirty": bool(capture(["git", "status", "--porcelain"])),
              "system": platform.platform(), "precision": "runtime BF16 storage/FP32 arithmetic",
              "input_size": [256, 256], "gpu_id": args.gpu_id,
              "binary_sha256": sha256(args.binary), "input_sha256": sha256(args.input),
              "gpu_memory_method": "whole-device memory.used sampled with nvidia-smi at approximately 0.5 s; includes other processes",
              "runs": []}
    if shutil.which("nvidia-smi"):
        record["gpu"] = capture(["nvidia-smi", "-i", str(args.gpu_id),
                                  "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"])
    result = output / "result.json"
    try:
        for package in [args.package] + ([args.baseline_package] if args.baseline_package else []):
            subprocess.run([str(args.binary.resolve()), "--check-model", "--model-dir", str(package.resolve())],
                           check=True, cwd=ROOT)
        cases = []
        if args.baseline_package:
            cases.extend((args.baseline_package, f"torchscript-s{steps}", steps, 1.0) for steps in (1, 2))
        cases.extend((args.package, f"pt2-cfg{scale:g}-s{steps}", steps, scale)
                     for scale in (1.0, 1.5) for steps in (1, 2))
        for package, name, steps, scale in cases:
            row = run_case(args.binary.resolve(), package.resolve(), args.input.resolve(), output,
                           args.gpu_id, name, steps, scale)
            record["runs"].append(row)
            result.write_text(json.dumps(record, indent=2) + "\n")
            if row["status"] != "pass":
                raise RuntimeError(f"{name}: {row['status']}; inspect {row['log']}")
        record["status"] = "pass"
    except Exception as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        result.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
