#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 SeedVR2-ncnn contributors
"""Audit fixed-256 PT2 VAE CPU/Vulkan against separately supplied SeedVR code.

Measures both pristine checkpoint FP32 and the existing exporter's
BF16-roundtrip weights with FP32 arithmetic. Uses posterior.mode(), which
differs from the original inference entry's default posterior sampling.
No external source, weights, or runtime Python dependency is bundled.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_reference_dit import PINNED_REFERENCE
from tools.export_dit_stack import _sha256
from tools.reference.portable_f32 import write_portable_golden
from tools.run_pt2_cfg_experiment import gpu_memory


def image_tensor(path):
    with Image.open(path) as image:
        if image.mode != "RGB" or image.size != (256, 256):
            raise ValueError("expected an RGB 256x256 image; resizing/cropping is outside this audit")
        value = np.array(image, dtype=np.uint8, copy=True)
    return torch.from_numpy(value).permute(2, 0, 1).unsqueeze(1).float().div(255).mul(2).sub(1)


def errors(actual, expected):
    if actual.shape != expected.shape or not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("non-finite or mismatched reference tensors")
    difference = (actual.double() - expected.double())
    return dict(max_abs=difference.abs().max().item(), rmse=difference.square().mean().sqrt().item())


def verify_vae(package):
    provenance_path = package.with_name(package.name + ".provenance.json")
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("kind") != "fixed-256-pt2-dit-package":
        raise ValueError("expected the separately provenance-recorded PT2 package")
    entries = {}
    for line in (package / "manifest.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if name in entries:
            raise ValueError("duplicate package manifest entry")
        entries[name] = digest
    if entries != provenance["runtime_sha256"]:
        raise ValueError("package manifest differs from external provenance")
    checked = {}
    for stem in ("vae_encode", "vae_decode"):
        for suffix in ("param", "bin"):
            name = f"256x256/{stem}.ncnn.{suffix}"
            path = package / name
            if path.is_symlink() or not path.resolve().is_relative_to(package) or _sha256(path) != entries[name]:
                raise ValueError(f"unsafe or changed VAE artifact: {name}")
            checked[name] = entries[name]
    return dict(manifest_sha256=_sha256(package / "manifest.sha256"),
                provenance_sha256=_sha256(provenance_path), verified_vae_runtime_files=checked,
                scope="four VAE artifacts rehashed; DiT and source PT2 archives not rehashed here")


def build_references(args, output, result):
    reference = args.reference_root.resolve()
    commit_file = reference / ".source-commit"
    commit = (commit_file.read_text().strip() if commit_file.is_file() else
              subprocess.check_output(["git", "-C", str(reference), "rev-parse", "HEAD"], text=True).strip())
    if commit != PINNED_REFERENCE:
        raise ValueError("expected the reviewed pinned original SeedVR source")
    result.update(reference_commit=commit, checkpoint_sha256=_sha256(args.checkpoint),
                  package=verify_vae(args.package), image_sha256=_sha256(args.image))
    sample = image_tensor(args.image)
    sys.path.insert(0, str(reference))
    import diffusers.utils.import_utils as integration
    integration._transformers_available = False
    from common.config import create_object, load_config
    previous = Path.cwd()
    try:
        os.chdir(reference)
        config = load_config("configs_3b/main.yaml")
        model = create_object(config.vae.model).float().cpu().requires_grad_(False).eval()
    finally:
        os.chdir(previous)
    imported = Path(sys.modules[model.__class__.__module__].__file__).resolve()
    if imported != reference / "models/video_vae_v3/modules/attn_video_vae.py":
        raise ValueError("a different reference VAE is already imported")
    state = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if len(state) != 250 or any(value.dtype != torch.float32 for value in state.values()):
        raise ValueError("expected the reviewed 250-entry FP32 VAE checkpoint")
    loaded = model.load_state_dict(state, strict=True)
    del state
    model.set_causal_slicing(split_size=None, memory_device="same")
    model.set_memory_limit(conv_max_mem=None, norm_max_mem=None)
    scale, shift = float(config.vae.scaling_factor), float(config.vae.get("shifting_factor", 0.0))
    if scale != 0.9152 or shift != 0:
        raise ValueError("unreviewed VAE latent scaling")
    result.update(checkpoint_entries=250, missing_keys=loaded.missing_keys, unexpected_keys=loaded.unexpected_keys,
                  checkpoint_dtype="float32", arithmetic="CPU float32", posterior="mode, not sample",
                  scaling_factor=scale, shifting_factor=shift, cpu_threads=torch.get_num_threads(),
                  preprocessing="RGB uint8 / 255 * 2 - 1, no resize; original VAE preprocess",
                  adaptations=["CPU FP32 arithmetic instead of original configured BF16",
                               "causal slicing and convolution/norm memory slicing disabled for one frame",
                               "unused diffusers transformers integration disabled in this process"],
                  dependency_versions={name: importlib.metadata.version(name) for name in
                                       ("torch", "diffusers", "omegaconf", "einops", "Pillow")})
    tensors, fixtures = {}, {}
    for precision in ("checkpoint-fp32", "bf16-roundtrip-fp32"):
        if precision == "bf16-roundtrip-fp32":
            peak, changed, count = 0.0, 0, 0
            for value in model.state_dict().values():
                if value.is_floating_point():
                    rounded = value.bfloat16().float()
                    peak = max(peak, (value - rounded).abs().max().item())
                    changed += int((value != rounded).sum())
                    count += value.numel()
            result["weight_rounding"] = dict(max_abs=peak, changed_values=changed, floating_values=count)
            model.bfloat16().float()
        print(f"reference VAE {precision}: encode/decode", flush=True)
        started = time.perf_counter()
        with torch.inference_mode():
            # The original wrapper also constructs an unused sampled latent;
            # only its independently computed posterior mode is consumed here.
            encoded = model.encode(model.preprocess(sample.unsqueeze(0)))
            latent = (encoded.posterior.mode().squeeze(2) - shift) * scale
            reconstructed = model.decode(latent / scale + shift).sample
        elapsed = time.perf_counter() - started
        values = (latent.squeeze(0).unsqueeze(1).clone(), reconstructed.squeeze(0).unsqueeze(1).clone())
        if values[0].shape != (16, 1, 32, 32) or values[1].shape != (3, 1, 256, 256):
            raise ValueError("original VAE returned unexpected shapes")
        for value in values:
            if not torch.isfinite(value).all():
                raise ValueError("non-finite original VAE outputs")
        fixture = output / f"{precision}.f32"
        layout = write_portable_golden(fixture, [("input", sample), ("latent", values[0]),
                                                ("reconstruction", values[1])])
        fixtures[precision] = fixture
        tensors[precision] = values
        result.setdefault("references", {})[precision] = dict(wall_seconds=elapsed,
            fixture_sha256=_sha256(fixture), fixture_layout=layout)
        rgb = (values[1][:, 0].permute(1, 2, 0).clamp(-1, 1).add(1).mul(127.5).round().byte().numpy())
        Image.fromarray(rgb).save(output / f"{precision}.png")
        # This also makes the real-image condition reusable by the next
        # development-only saved-noise sampler comparison.
        torch.save(values[0], output / f"{precision}-condition.pt")
    raw, rounded = tensors["checkpoint-fp32"], tensors["bf16-roundtrip-fp32"]
    result["rounding_effect"] = dict(latent=errors(rounded[0], raw[0]), reconstruction=errors(rounded[1], raw[1]))
    result["reference_peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    result["reference_sources"] = {str(Path(m.__file__).resolve().relative_to(reference)): _sha256(Path(m.__file__))
        for m in list(sys.modules.values()) if getattr(m, "__file__", None)
        and Path(m.__file__).is_file() and Path(m.__file__).resolve().is_relative_to(reference)}
    for name in ("LICENSE", "configs_3b/main.yaml", "models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml"):
        result["reference_sources"][name] = _sha256(reference / name)
    del model, tensors, encoded, latent, reconstructed
    gc.collect()
    return fixtures


def run_native(args, output, precision, fixture, mode):
    strict_fp32 = mode == "vulkan" and args.native_precision == "fp32" and precision == "bf16-roundtrip-fp32"
    encode_limit, decode_limit = (1e-3, 1e-3) if strict_fp32 else (0.1, 0.25)
    command = [str(ROOT / "tools/with-nvidia-vulkan-runtime.sh"), str(args.runner.resolve()),
               str(args.package / "256x256"), str(fixture), mode, str(args.gpu_id),
               "--max-encode-abs", str(encode_limit), "--max-decode-abs", str(decode_limit)]
    if mode == "vulkan" and args.native_precision == "fp32":
        command.append("--fp32")
    log = output / f"{precision}-{mode}.log"
    samples = [gpu_memory(args.gpu_id)] if mode == "vulkan" else []
    rss_samples = []
    started = time.perf_counter()
    with log.open("w") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        while process.poll() is None:
            try:
                status = Path(f"/proc/{process.pid}/status").read_text()
                rss_samples.extend(int(line.split()[1]) / 1024 for line in status.splitlines()
                                   if line.startswith("VmHWM:"))
            except (FileNotFoundError, ProcessLookupError):
                pass
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                if mode == "vulkan":
                    samples.append(gpu_memory(args.gpu_id))
    rows = []
    for line in log.read_text().splitlines():
        if line.startswith(("vae_encode ", "vae_decode_from_native_latent ")):
            stage, *fields = line.split()
            row = {key: (value if key == "fnv1a64" else float(value)) for key, value in
                   (field.split("=", 1) for field in fields)}
            row.update(stage=stage, limit=encode_limit if stage == "vae_encode" else decode_limit)
            row["pass"] = np.isfinite([row[k] for k in ("max_abs", "rmse", "max_rel")]).all().item() and row["max_abs"] <= row["limit"]
            rows.append(row)
    passed = process.returncode == 0 and len(rows) == 2 and {r["stage"] for r in rows} == {
        "vae_encode", "vae_decode_from_native_latent"} and all(r["pass"] for r in rows)
    result = dict(reference_weights=precision, mode=mode,
                  storage=args.native_precision if mode == "vulkan" else "fp32",
                  command=command, exit_code=process.returncode,
                  wall_seconds=time.perf_counter() - started, stages=rows, passed=passed,
                  log_sha256=_sha256(log), process_sampled_vmhwm_mib=max(rss_samples, default=None),
                  rss_method="native /proc/PID/status VmHWM sampled about every 0.5s; final peak may be missed")
    if samples:
        result.update(device_memory_initial_mib=samples[0], device_memory_sampled_peak_mib=max(
            (s for s in samples if s is not None), default=None), device_memory_sample_count=sum(s is not None for s in samples),
            gpu_memory_method="whole-device memory.used sampled about every 0.5s, not an exact process peak")
    print(f"native VAE {precision} {mode}: {'pass' if passed else 'fail'}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-root", "checkpoint", "package", "image", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--runner", type=Path, default=ROOT / "build/seedvr2-vae-smoke")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--native-precision", choices=("bf16", "fp32"), default="bf16",
                        help="Vulkan storage; FP32 validates matching rounded weights at max_abs<=1e-3")
    parser.add_argument("--native-modes", nargs="+", choices=("cpu", "vulkan"), default=["cpu", "vulkan"],
                        help="select native backends, e.g. vulkan after the separate CPU baseline")
    args = parser.parse_args()
    if args.gpu_id < 0 or len(set(args.native_modes)) != len(args.native_modes):
        parser.error("expected a nonnegative GPU id and distinct native backends")
    for name in ("reference_root", "checkpoint", "package", "image", "runner"):
        setattr(args, name, getattr(args, name).resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_names = ["tools/compare_reference_vae.py", "tools/vae_ncnn_smoke.cpp", "tools/export_vae.py",
                    "tools/reference/portable_f32.py", "src/vae/static_vae_graph.cpp"]
    source_names += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "src/vae").rglob("*"))
                     if p.is_file() and str(p.relative_to(ROOT)) not in source_names]
    result = dict(status="running", scope="single RGB256 image VAE mode encode/native-latent decode audit",
                  official_default_sampling_validated=False, full_image_quality_validated=False,
                  date_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), command=[sys.executable, *sys.argv],
                  torch_version=torch.__version__, system=platform.platform(),
                  repository_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  repository_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
                  sources={name: _sha256(ROOT / name) for name in source_names}, runner_sha256=_sha256(args.runner),
                  native_precision=args.native_precision, native_modes=args.native_modes,
                  thresholds="FP32 Vulkan matching rounded reference: encode/decode 1e-3; other cases: 0.1/0.25, fixed before measurement",
                  peak_process_gpu_memory_mib=None)
    record = output / "result.json"
    try:
        fixtures = build_references(args, output, result)
        result["gpu"] = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu_id),
            "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()
        result["native_runs"] = []
        for precision, fixture in fixtures.items():
            for mode in args.native_modes:
                record.write_text(json.dumps(result, indent=2) + "\n")
                result["native_runs"].append(run_native(args, output, precision, fixture, mode))
        result["sources_unchanged_at_end"] = all(_sha256(ROOT / name) == value for name, value in result["sources"].items())
        result["runner_unchanged_at_end"] = _sha256(args.runner) == result["runner_sha256"]
        if not result["sources_unchanged_at_end"] or not result["runner_unchanged_at_end"]:
            raise RuntimeError("measured sources or binary changed during experiment")
        if not all(run["passed"] for run in result["native_runs"]):
            raise RuntimeError("one or more native VAE comparisons failed; retained all requested diagnostics")
        result["status"] = "pass"
    except Exception as exc:
        result.update(status="fail", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record.write_text(json.dumps(result, indent=2) + "\n")
        print(f"{result['status']}: {record}", flush=True)


if __name__ == "__main__":
    main()
