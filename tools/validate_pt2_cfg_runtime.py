#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 SeedVR2-ncnn contributors
"""Compare matched DiT frontends and validate a two-frame fixed-256 CFG AVI.

Run after run_pt2_cfg_experiment.py. The same-binary repeat must reproduce
all three CPU tensor handoffs and the PNG exactly. Frontend PNG comparisons
are measurements, not quality thresholds. Graph numerical checks are separate.
Requires the existing conversion environment's NumPy and Pillow, not runtime
dependencies. GPU commands execute sequentially on one explicitly selected GPU.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.run_pt2_cfg_experiment import ROOT, gpu_memory, run_case, sha256


def compare(first: Path, second: Path) -> dict:
    with Image.open(first) as image:
        a = np.asarray(image.convert("RGB"), dtype=np.float64)
    with Image.open(second) as image:
        b = np.asarray(image.convert("RGB"), dtype=np.float64)
    if a.shape != b.shape or a.shape != (256, 256, 3):
        raise ValueError(f"expected two RGB 256x256 images: {a.shape}, {b.shape}")
    delta = np.abs(a - b)
    return {"first": str(first), "second": str(second), "shape": list(a.shape),
            "first_sha256": sha256(first), "second_sha256": sha256(second),
            "max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
            "rmse": float(np.sqrt(np.mean(delta ** 2))),
            "channel_fraction_over_1": float(np.mean(delta > 1)),
            "channel_fraction_over_5": float(np.mean(delta > 5))}


def run_video(binary: Path, package: Path, video: Path, output: Path, gpu_id: int,
              *, seed: int | None = None, frames: int = 2, start_frame: int = 0) -> dict:
    if frames <= 0 or start_frame < 0 or (seed is not None and not 0 <= seed <= 0xffffffff):
        raise ValueError("expected positive frames, nonnegative start and uint32 seed")
    output.mkdir(exist_ok=False)
    stem = f"pt2-cfg1.5-s2-f{frames}"
    avi, log = output / f"{stem}.avi", output / f"{stem}.log"
    command = [str(ROOT / "tools/with-nvidia-vulkan-runtime.sh"), str(binary),
               "--model-dir", str(package), "--input", str(video), "--output", str(avi),
               "--width", "256", "--height", "256", "--gpu-id", str(gpu_id),
               "--frames", str(frames), "--start-frame", str(start_frame),
               "--steps", "2", "--cfg-scale", "1.5", "--profile"]
    if seed is not None:
        command += ["--seed", str(seed)]
    initial = gpu_memory(gpu_id)
    samples = [] if initial is None else [initial]
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
    row = {"command": command, "exit_code": process.returncode,
           "process_wall_seconds": time.perf_counter() - started,
           "binary_sha256": sha256(binary), "input_sha256": sha256(video),
           "package_manifest_sha256": sha256(package / "manifest.sha256"),
           "input_frames": frames, "start_frame": start_frame,
           "steps_per_frame": 2, "cfg_scale": 1.5, "gpu_id": gpu_id,
           "precision": "runtime BF16 storage/FP32 arithmetic",
           "temporal_contract": "frame-wise CLI; source_t=1; no temporal VAE or multi-frame attention",
           "device_memory_initial_mib": initial,
           "device_memory_sampled_peak_mib": max(samples) if samples else None,
           "device_memory_sample_count": len(samples),
           "gpu_memory_method": "whole-device nvidia-smi memory.used sampled approximately every 0.5 s; includes other processes",
           "log": str(log), "status": "failed"}
    text = log.read_text()
    profile = re.search(r"profile name=total ms=([0-9.]+) peak-rss-mib=([0-9]+)", text)
    if profile:
        row.update(profile_total_ms=float(profile[1]), process_peak_rss_mib=int(profile[2]))
    if process.returncode == 0 and "seedvr2-video-inference: ok" in text and avi.is_file():
        # Validate the RGB24 AVI container written by this project's native CLI.
        with avi.open("rb") as source:
            header = source.read(4096)
        if header[:4] != b"RIFF" or header[8:12] != b"AVI ":
            raise ValueError("output is not a RIFF AVI")
        start = header.index(b"avih") + 8
        actual_frames = struct.unpack_from("<I", header, start + 16)[0]
        width, height = struct.unpack_from("<II", header, start + 32)
        released = text.find("stage=release-dit-stacks-for-decode")
        row.update(output=str(avi), output_sha256=sha256(avi), output_frames=actual_frames,
                   output_size=[width, height],
                   cfg_stacks_released_before_decode=0 <= released < text.find("stage=load-decode"))
        if (actual_frames == frames and (width, height) == (256, 256)
                and row["cfg_stacks_released_before_decode"]
                and "stage=load-dit-stack-negative" in text
                and "stage=dit-stack step=1" in text):
            row["status"] = "pass"
    if seed is not None:
        row.update(seed=seed, noise_algorithm="mt19937-box-muller-v1")
        fingerprints = re.findall(
            r"profile name=tensor-fingerprint phase=(\S+) frame=(\d+) fnv1a64=([0-9a-f]{16}) finite=([01])", text)
        row["tensor_fingerprints"] = {f"{phase}:{frame}": digest for phase, frame, digest, _ in fingerprints}
        expected = {f"{phase}:{frame}" for frame in range(start_frame, start_frame + frames)
                    for phase in ("initial-noise", "condition-latent", "denoised-latent", "reconstruction")}
        if (set(row["tensor_fingerprints"]) != expected or len(fingerprints) != len(expected)
                or any(finite != "1" for _, _, _, finite in fingerprints)
                or any((f"stage=noise-create distribution=normal seed={seed} frame={frame} "
                        "algorithm=mt19937-box-muller-v1") not in text
                       for frame in range(start_frame, start_frame + frames))):
            row["status"] = "failed-noise-contract"
    (output / "result.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-results", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--control-package", type=Path, required=True,
                        help="TorchScript DiT with identical PT2 VAE and conditioning")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--video-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="new evidence directory")
    parser.add_argument("--binary", type=Path, default=ROOT / "build/seedvr2-ncnn")
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()
    primary = json.loads(args.image_results.read_text())
    binary = args.binary.resolve()
    package, control = args.package.resolve(), args.control_package.resolve()
    if (args.gpu_id < 0 or primary.get("status") != "pass"
            or primary.get("binary_sha256") != sha256(binary)
            or primary.get("input_sha256") != sha256(args.input)
            or primary.get("gpu_id") != args.gpu_id):
        parser.error("image evidence must pass with the same binary, input and GPU")
    rows = {row["name"]: row for row in primary["runs"]}
    expected = ["torchscript-s1", "torchscript-s2", "pt2-cfg1-s1", "pt2-cfg1-s2",
                "pt2-cfg1.5-s1", "pt2-cfg1.5-s2"]
    if any(name not in rows or rows[name]["status"] != "pass" for name in expected):
        parser.error("image evidence must contain the complete six-case matrix")
    for name in expected:
        fingerprints = rows[name].get("tensor_fingerprints", {})
        if (set(fingerprints) != {f"{phase}:0" for phase in
                                 ("condition-latent", "denoised-latent", "reconstruction")}
                or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{16}", value)
                       for value in fingerprints.values())):
            parser.error(f"image evidence lacks valid tensor handoffs: {name}")
        if sha256(Path(rows[name]["output"])) != rows[name]["output_sha256"]:
            parser.error(f"image artifact differs from recorded evidence: {name}")
    for name in expected[2:]:
        if rows[name]["package_manifest_sha256"] != sha256(package / "manifest.sha256"):
            parser.error("candidate manifest differs from image evidence")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = {"date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "primary_result": str(args.image_results.resolve()),
              "primary_result_sha256": sha256(args.image_results),
              "repository_commit": primary["repository_commit"],
              "repository_dirty": primary["repository_dirty"],
              "validator_sha256": sha256(Path(__file__)),
              "binary_sha256": sha256(binary), "input_sha256": sha256(args.input),
              "gpu_id": args.gpu_id, "gpu_memory_method": primary["gpu_memory_method"],
              "control_frontend": "TorchScript DiT; same PT2 VAE as candidate",
              "png_acceptance": "difference measurements only; no quality/equivalence claim",
              "runs": []}
    try:
        subprocess.run([str(binary), "--check-model", "--model-dir", str(control)], check=True, cwd=ROOT)
        shared = [f"256x256/{name}.ncnn.{suffix}" for name in ("vae_encode", "vae_decode")
                  for suffix in ("param", "bin")]
        shared += [f"conditioning/{sign}_emb.f32" for sign in ("pos", "neg")]
        record["matched_vae_and_conditioning_hashes"] = {
            name: sha256(package / name) for name in shared}
        for name, digest in record["matched_vae_and_conditioning_hashes"].items():
            if sha256(control / name) != digest:
                raise ValueError(f"control and candidate differ at {name}")
        for scale in (1.0, 1.5):
            row = run_case(binary, control, args.input.resolve(), output, args.gpu_id,
                           f"torchscript-cfg{scale:g}-s2", 2, scale)
            record["runs"].append(row)
            if row["status"] != "pass":
                raise RuntimeError(row)
        pairs = {
            "matched_dit_frontends_cfg1_s2": (output / "torchscript-cfg1-s2.png", Path(rows["pt2-cfg1-s2"]["output"])),
            "matched_dit_frontends_cfg1.5_s2": (output / "torchscript-cfg1.5-s2.png", Path(rows["pt2-cfg1.5-s2"]["output"])),
            "production_vs_pt2_s1": (Path(rows["torchscript-s1"]["output"]), Path(rows["pt2-cfg1-s1"]["output"])),
            "production_vs_pt2_s2": (Path(rows["torchscript-s2"]["output"]), Path(rows["pt2-cfg1-s2"]["output"])),
            "pt2_cfg_effect_s2": (Path(rows["pt2-cfg1-s2"]["output"]), Path(rows["pt2-cfg1.5-s2"]["output"]))}
        record["png_comparisons"] = {name: compare(*paths) for name, paths in pairs.items()}
        # Scope exact repeatability to this fixed fixture, binary and GPU.
        # A frontend comparison cannot explain variation within one frontend.
        repeated = run_case(binary, package, args.input.resolve(), output, args.gpu_id,
                            "pt2-cfg1.5-s2-repeat", 2, 1.5)
        record["runs"].append(repeated)
        if repeated["status"] != "pass":
            raise RuntimeError(repeated)
        record["png_comparisons"]["pt2_same_binary_repeat_cfg1.5_s2"] = compare(
            Path(rows["pt2-cfg1.5-s2"]["output"]), Path(repeated["output"]))
        repeat_metrics = record["png_comparisons"]["pt2_same_binary_repeat_cfg1.5_s2"]
        first_fingerprints = rows["pt2-cfg1.5-s2"].get("tensor_fingerprints", {})
        if (repeat_metrics["max_abs"] != 0 or repeat_metrics["first_sha256"] != repeat_metrics["second_sha256"]
                or len(first_fingerprints) != 3 or first_fingerprints != repeated["tensor_fingerprints"]):
            raise RuntimeError("same-binary fixed-fixture tensor/PNG repeatability failed")
        record["same_binary_repeatability"] = {
            "status": "pass", "scope": "fixed 256 fixture, identical binary/input/manifest/GPU",
            "tensor_fingerprints": first_fingerprints, "png_sha256": repeated["output_sha256"]}
        invalid = output / "invalid-128.png"
        command = [str(ROOT / "tools/with-nvidia-vulkan-runtime.sh"), str(binary),
                   "--model-dir", str(package), "--input", str(args.input.resolve()),
                   "--output", str(invalid), "--width", "128", "--height", "128", "--gpu-id", str(args.gpu_id)]
        result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (output / "invalid-128.log").write_text(result.stdout)
        if (result.returncode == 0 or invalid.exists() or "stage=vae" in result.stdout
                or "model variant is missing for target 128x128" not in result.stdout):
            raise RuntimeError("fixed-shape rejection contract failed")
        record["fixed_shape_rejection"] = {"command": command, "exit_code": result.returncode,
                                            "diagnostic": result.stdout.strip()}
        record["video"] = run_video(binary, package, args.video_input.resolve(), output / "video", args.gpu_id)
        if record["video"]["status"] != "pass":
            raise RuntimeError(record["video"])
        record["status"] = "pass"
    except Exception as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        (output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
