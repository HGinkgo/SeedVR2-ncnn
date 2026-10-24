#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 SeedVR2-ncnn contributors
"""Compare fixed-256 closed-loop Vulkan denoising with imported SeedVR CPU code.

Uses identical saved Gaussian noise, synthetic or saved latent conditioning, and the
package's actual positive/negative text embeddings. Imports the original
NaDiT, uniform-trailing schedule, CFG and Euler sampler with the documented
CPU/FP32 architecture adaptations. This does not validate VAE, image
preprocessing, FlashAttention/BF16, or image quality.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import math
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_reference_dit import _AttentionPrecision
from tools.export_dit_stack import _sha256
from tools.prepare_pt2_cfg_package import DIT_GRAPHS, LAST_BLOCK_CONTRACT, OUTPUT_MODULATION
from tools.reference.imported_dit import imported_source_hashes, load_reference_dit
from tools.reference.portable_f32 import write_portable_golden
from tools.run_pt2_cfg_experiment import gpu_memory


def patch_output(value):
    if value.shape != (16, 1, 32, 32):
        raise ValueError("expected a fixed [16,1,32,32] latent")
    return value[:, 0].permute(1, 2, 0).reshape(16, 2, 16, 2, 16).permute(0, 2, 1, 3, 4).reshape(256, 64)


def model_video(sample, condition):
    if sample.shape != (16, 1, 32, 32) or condition.shape != sample.shape:
        raise ValueError("expected matching fixed latent shapes")
    # Match the original inference input: current noise, latent SR condition,
    # and the spatial SR mask, in channel-last unpatched token order.
    channels = torch.cat((sample, condition, torch.ones_like(sample[:1])), dim=0)
    return channels.permute(1, 2, 3, 0).reshape(1024, 33)


def load_condition(path, generator):
    if path is None:
        return torch.randn((16, 1, 32, 32), generator=generator) * 0.1, {
            "kind": "synthetic FP32 Gaussian scaled by 0.1"}
    path = path.resolve()
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(value, torch.Tensor) or value.shape != (16, 1, 32, 32)
            or value.dtype != torch.float32 or not torch.isfinite(value).all()):
        raise ValueError("saved condition must be a finite float32 [16,1,32,32] tensor")
    return value.contiguous(), dict(kind="saved latent; identical condition bytes for both denoisers",
                                     path=str(path), sha256=_sha256(path))


def load_initial_noise(path, generator):
    if path is None:
        return torch.randn((16, 1, 32, 32), generator=generator), {
            "algorithm": "PyTorch CPU Gaussian saved once; native runner reads identical bytes"}
    path = path.resolve()
    if path.stat().st_size != 16 * 32 * 32 * 4:
        raise ValueError("initial noise must contain exactly 16384 finite little-endian float32 values")
    value = np.fromfile(path, dtype="<f4")
    if not np.isfinite(value).all():
        raise ValueError("initial noise must contain exactly 16384 finite little-endian float32 values")
    return torch.from_numpy(value.reshape(16, 1, 32, 32)), {
        "algorithm": "supplied FP32 bytes; native runner reads identical bytes",
        "path": str(path), "sha256": _sha256(path)}


def verify_package(package, checkpoint_hash):
    provenance = json.loads(package.with_name(package.name + ".provenance.json").read_text())
    if provenance.get("kind") != "fixed-256-pt2-dit-package":
        raise ValueError("expected a genuine PT2 package with external provenance")
    stacks = {m["text_tokens"]: m for m in provenance["stack_manifests"].values()}
    if set(stacks) != {58, 64}:
        raise ValueError("expected positive 58-token and negative 64-token stacks")
    for m in stacks.values():
        if (m.get("frontend") != "pt2" or m.get("source_shape") != [1, 16, 16]
                or m.get("block_count") != 32 or m.get("checkpoint_sha256") != checkpoint_hash
                or m.get("last_block_contract") != LAST_BLOCK_CONTRACT
                or m.get("output_modulation") != OUTPUT_MODULATION):
            raise ValueError("package provenance lacks the corrected fixed-256 PT2 contracts")
        graphs = {g["name"]: g for g in m["graphs"]}
        if len(m["graphs"]) != 35 or set(graphs) != set(DIT_GRAPHS) or any(
                g.get("frontend") != "pt2" or g.get("conversion_status") != "success"
                or len(g.get("source_sha256", "")) != 64 for g in graphs.values()):
            raise ValueError("incomplete PT2 source/conversion metadata")
    hashes = provenance["runtime_sha256"]
    if len(hashes) != 146:
        raise ValueError("expected exactly 146 runtime artifacts")
    entries = {}
    for line in (package / "manifest.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if name in entries:
            raise ValueError("duplicate manifest entry")
        entries[name] = digest
    if entries != hashes:
        raise ValueError("runtime manifest differs from external provenance")
    for name, digest in hashes.items():
        path = package / name
        if not path.resolve().is_relative_to(package) or path.is_symlink() or _sha256(path) != digest:
            raise ValueError(f"unsafe or changed runtime artifact: {name}")
    return dict(manifest_sha256=_sha256(package / "manifest.sha256"),
                provenance_sha256=_sha256(package.with_name(package.name + ".provenance.json")),
                verified_runtime_artifacts=len(hashes),
                source_provenance="recorded PT2 sources; runtime bytes rehashed, source PT2 archives not rehashed here")


def build_reference(args, output, result):
    checkpoint_hash = _sha256(args.checkpoint)
    result.update(checkpoint_sha256=checkpoint_hash, package=verify_package(args.package, checkpoint_hash))
    text = {}
    for sign, tokens in (("pos", 58), ("neg", 64)):
        path = args.package / f"conditioning/{sign}_emb.f32"
        raw = np.fromfile(path, dtype="<f4")
        if raw.size != tokens * 5120 or not np.isfinite(raw).all():
            raise ValueError("invalid package text embeddings")
        text[sign] = torch.from_numpy(raw.reshape(tokens, 5120))
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    noise, noise_metadata = load_initial_noise(getattr(args, "initial_noise", None), generator)
    condition, condition_metadata = load_condition(args.condition, generator)
    records = [("initial_noise", noise), ("condition", condition),
               ("input_text_positive", text["pos"]), ("input_text_negative", text["neg"])]
    model, config, metadata = load_reference_dit(args.reference_root, args.checkpoint, 64)
    from common.diffusion.config import (create_schedule_from_config, create_sampler_from_config,
                                         create_sampling_timesteps_from_config)
    from common.diffusion.utils import classifier_free_guidance
    # The product's fixed-256 timetable is the official float-T, shift=1
    # timetable; steps/CFG are explicit bounded test overrides of 50/7.5.
    if (config.diffusion.schedule.type != "lerp" or config.diffusion.schedule.T != 1000.0
            or config.diffusion.sampler.type != "euler" or config.diffusion.sampler.prediction_type != "v_lerp"
            or config.diffusion.cfg.rescale != 0):
        raise ValueError("unreviewed original diffusion configuration")
    config.diffusion.timesteps.sampling.steps = args.steps
    schedule = create_schedule_from_config(config.diffusion.schedule, torch.device("cpu"))
    timesteps = create_sampling_timesteps_from_config(config.diffusion.timesteps.sampling, schedule, torch.device("cpu"))
    sampler = create_sampler_from_config(config.diffusion.sampler, schedule, timesteps)
    records.extend([("timesteps", timesteps.timesteps), ("cfg_scale", torch.tensor([args.cfg_scale]))])
    result.update(reference=metadata, timesteps=timesteps.timesteps.tolist(),
                  shape=[16, 1, 32, 32], seed=args.seed,
                  noise_algorithm=noise_metadata["algorithm"], initial_noise=noise_metadata,
                  conditioning=condition_metadata, text_conditioning="actual package text embeddings",
                  configuration_overrides=dict(steps=args.steps, cfg_scale=args.cfg_scale),
                  comparison_limit="max_abs <= 5e-4 + 5e-5 * reference_peak at every prediction and updated latent")
    previous_step = -1

    def predict(arguments):
        nonlocal previous_step
        i = arguments.i
        if i != previous_step + 1 or not 0 <= i < args.steps:
            raise ValueError("unexpected imported sampler model call order")
        if i:
            records.append((f"step_{i-1}_latent", arguments.x_t.clone()))
        previous_step = i
        video = model_video(arguments.x_t, condition)
        predictions = []
        for sign, tokens in (("pos", 58), ("neg", 64)):
            raw = model(video, text[sign], torch.tensor([[1, 32, 32]]), torch.tensor([[tokens]]),
                        arguments.t.reshape(1)).vid_sample
            latent = raw.reshape(1, 32, 32, 16).permute(3, 0, 1, 2).contiguous()
            predictions.append(latent)
            records.append((f"step_{i}_{'positive' if sign == 'pos' else 'negative'}", patch_output(latent)))
        return classifier_free_guidance(*predictions, scale=args.cfg_scale, rescale=0.0)

    precision = _AttentionPrecision(bypass=True)
    started = time.perf_counter()
    with torch.inference_mode(), precision:
        endpoint = sampler.sample(noise.clone(), predict)
    records.append((f"step_{args.steps-1}_latent", endpoint))
    if previous_step != args.steps - 1 or precision.casts != args.steps * 2 * 96:
        raise ValueError("incomplete imported model/sampler execution")
    if any(not torch.isfinite(value).all() for _, value in records):
        raise ValueError("non-finite reference tensors")
    fixture = output / "sampling.f32"
    result.update(reference_wall_seconds=time.perf_counter() - started,
                  reference_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  attention_casts_observed=precision.casts, cpu_threads=torch.get_num_threads(),
                  fixture_layout=write_portable_golden(fixture, records), fixture_sha256=_sha256(fixture),
                  reference_sources=imported_source_hashes(args.reference_root))
    del model, records, sampler, endpoint
    gc.collect()
    return fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-root", "checkpoint", "package", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--runner", type=Path, default=ROOT / "build/seedvr2-reference-sampling-smoke")
    parser.add_argument("--steps", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--condition", type=Path,
                        help="optional saved finite float32 [16,1,32,32] latent, e.g. from compare_reference_vae.py")
    parser.add_argument("--initial-noise", type=Path, help="optional raw little-endian FP32 [16,1,32,32] noise")
    args = parser.parse_args()
    if (not math.isfinite(args.cfg_scale) or not 1 < args.cfg_scale <= 20
            or not 0 <= args.seed <= 0xffffffff or args.gpu_id < 0):
        parser.error("expected CFG in (1,20], uint32 seed, and a nonnegative GPU id")
    args.package = args.package.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_names = ["tools/compare_reference_sampling.py", "tools/reference/imported_dit.py",
                    "tools/compare_reference_dit.py", "tools/reference_sampling_smoke.cpp",
                    "CMakeLists.txt", "tools/with-nvidia-vulkan-runtime.sh",
                    "src/inference/performance_profile.cpp", "src/inference/performance_profile.h",
                    "src/inference/vulkan_benchmark.h"]
    for directory in ("src/dit", "src/awa", "src/sampler", "src/resolution"):
        source_names.extend(str(path.relative_to(ROOT)) for path in sorted((ROOT / directory).rglob("*"))
                            if path.suffix in {".cpp", ".h", ".comp"})
    result = dict(status="running", scope="adapted FP32 closed-loop denoiser, identical supplied latent conditioning",
                  unmodified_official=False, full_image_quality_validated=False,
                  command=[sys.executable, *sys.argv], date_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  torch_version=torch.__version__, system=platform.platform(), peak_gpu_memory_mib=None,
                  repository_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  repository_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
                  sources={name: _sha256(ROOT / name) for name in source_names},
                  runner_sha256=_sha256(args.runner))
    record = output / "result.json"
    try:
        fixture = build_reference(args, output, result)
        record.write_text(json.dumps(result, indent=2) + "\n")
        command = [str(ROOT / "tools/with-nvidia-vulkan-runtime.sh"), str(args.runner.resolve()),
                   str(args.package / "256x256"), str(args.package / "256x256/cfg_64"), str(fixture), str(args.gpu_id)]
        samples = [gpu_memory(args.gpu_id)]
        rss_samples = []
        result["gpu"] = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu_id),
            "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()
        log = output / "native.log"
        started = time.perf_counter()
        with log.open("w") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
            while True:
                try:
                    status = Path(f"/proc/{process.pid}/status").read_text()
                    rss_samples.extend(int(line.split()[1]) / 1024 for line in status.splitlines()
                                       if line.startswith("VmHWM:"))
                except (FileNotFoundError, ProcessLookupError):
                    pass
                try:
                    process.wait(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    samples.append(gpu_memory(args.gpu_id))
        rows = [json.loads(line.removeprefix("sampling_stage ")) for line in log.read_text().splitlines()
                if line.startswith("sampling_stage ")]
        runtime = [json.loads(line.removeprefix("sampling_runtime ")) for line in log.read_text().splitlines()
                   if line.startswith("sampling_runtime ")]
        result.update(native_command=command, native_wall_seconds=time.perf_counter() - started,
                      native_exit_code=process.returncode, stages=rows, native_log_sha256=_sha256(log),
                      device_memory_initial_mib=samples[0], device_memory_sampled_peak_mib=max(
                          (s for s in samples if s is not None), default=None),
                      device_memory_sample_count=sum(s is not None for s in samples),
                      process_sampled_vmhwm_mib=max(rss_samples, default=None),
                      rss_method="native /proc/PID/status VmHWM sampled about every 0.5s; final peak may be missed",
                      gpu_memory_method="whole-device memory.used sampled about every 0.5s, not an exact process peak")
        result["native_runtime"] = runtime
        result["sources_unchanged_at_end"] = all(_sha256(ROOT / name) == value for name, value in result["sources"].items())
        result["runner_unchanged_at_end"] = _sha256(args.runner) == result["runner_sha256"]
        result["fixture_unchanged_at_end"] = _sha256(fixture) == result["fixture_sha256"]
        result["condition_unchanged_at_end"] = (not args.condition or
            _sha256(args.condition) == result["conditioning"]["sha256"])
        if not all(result[name] for name in ("sources_unchanged_at_end", "runner_unchanged_at_end",
                                            "fixture_unchanged_at_end", "condition_unchanged_at_end")):
            raise RuntimeError("validation source, binary, fixture or saved condition changed during measurement")
        expected = {f"step_{i}_{stage}" for i in range(args.steps) for stage in ("positive", "negative", "latent")}
        if (process.returncode or len(runtime) != 1 or len(rows) != len(expected) or {r["stage"] for r in rows} != expected
                or any(not r["pass"] for r in rows)):
            raise RuntimeError("closed-loop native comparison failed; inspect native.log and retained stage errors")
        result["status"] = "pass"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record.write_text(json.dumps(result, indent=2) + "\n")
    print(record)


if __name__ == "__main__":
    main()
