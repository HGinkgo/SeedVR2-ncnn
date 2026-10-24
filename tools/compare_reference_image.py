#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 SeedVR2-ncnn contributors
"""Compare a fixed RGB256 VAE/DiT/CFG/Euler/VAE chain with imported SeedVR.

Development-only FP32 audit: each side encodes its own condition, advances
its own sampling state, and decodes its own final latent. Uses posterior
mode, matching BF16-rounded VAE weights and the documented reference DiT
adaptations. This does not validate the original default BF16 image quality.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_reference_sampling import build_reference
from tools.compare_reference_vae import build_references, image_tensor
from tools.export_dit_stack import _sha256
from tools.reference.imported_dit import imported_source_hashes
from tools.reference.portable_f32 import write_portable_golden
from tools.run_pt2_cfg_experiment import gpu_memory
from tools.validate_dit_stack import _read_golden


def decode_reference(reference, checkpoint, latent):
    # The earlier VAE audit verifies the pinned source and checkpoint dtype.
    # Construct the same original class, without any export/trace adapter.
    from common.config import create_object, load_config
    previous = Path.cwd()
    try:
        os.chdir(reference)
        config = load_config("configs_3b/main.yaml")
        model = create_object(config.vae.model).float().cpu().requires_grad_(False).eval()
    finally:
        os.chdir(previous)
    if Path(sys.modules[model.__class__.__module__].__file__).resolve() != reference / "models/video_vae_v3/modules/attn_video_vae.py":
        raise ValueError("a different reference VAE is already imported")
    state = torch.load(checkpoint, mmap=True, map_location="cpu", weights_only=True)
    if len(state) != 250 or any(value.dtype != torch.float32 for value in state.values()):
        raise ValueError("expected the reviewed FP32 VAE checkpoint")
    model.load_state_dict(state, strict=True)
    del state
    model.bfloat16().float()
    model.set_causal_slicing(split_size=None, memory_device="same")
    model.set_memory_limit(conv_max_mem=None, norm_max_mem=None)
    if float(config.vae.scaling_factor) != 0.9152 or float(config.vae.get("shifting_factor", 0)) != 0:
        raise ValueError("unreviewed VAE scaling")
    started = time.perf_counter()
    with torch.inference_mode():
        decoded = model.decode(latent[:, 0].unsqueeze(0) / 0.9152).sample.squeeze(0).unsqueeze(1).clone()
    elapsed = time.perf_counter() - started
    if decoded.shape != (3, 1, 256, 256) or not torch.isfinite(decoded).all():
        raise ValueError("invalid original decoded image")
    del model
    gc.collect()
    return decoded, elapsed


def save_rgb(value, path):
    if value.shape != (3, 1, 256, 256) or not np.isfinite(value).all():
        raise ValueError("invalid decoded image tensor")
    rgb = np.rint((np.clip(value[:, 0].transpose(1, 2, 0), -1, 1) + 1) * 127.5).astype(np.uint8)
    Image.fromarray(rgb).save(path)
    return rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-root", "checkpoint", "vae-checkpoint", "package", "image", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--runner", type=Path, default=ROOT / "build/seedvr2-reference-sampling-smoke")
    parser.add_argument("--steps", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--initial-noise", type=Path, help="optional raw little-endian FP32 [16,1,32,32] noise")
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()
    if (not math.isfinite(args.cfg_scale) or not 1 < args.cfg_scale <= 20
            or not 0 <= args.seed <= 0xffffffff or args.gpu_id < 0):
        parser.error("expected CFG in (1,20], uint32 seed, and a nonnegative GPU id")
    for name in ("reference_root", "checkpoint", "vae_checkpoint", "package", "image", "runner"):
        setattr(args, name, getattr(args, name).resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_names = ["CMakeLists.txt", "tools/with-nvidia-vulkan-runtime.sh",
                    "tools/compare_reference_image.py", "tools/compare_reference_sampling.py",
                    "tools/compare_reference_vae.py", "tools/compare_reference_dit.py",
                    "tools/export_dit_stack.py", "tools/export_vae.py", "tools/prepare_pt2_cfg_package.py",
                    "tools/reference/imported_dit.py", "tools/reference/portable_f32.py",
                    "tools/validate_dit_stack.py", "tools/run_pt2_cfg_experiment.py",
                    "tools/reference_sampling_smoke.cpp"]
    for directory in ("src/dit", "src/awa", "src/sampler", "src/vae", "src/resolution", "src/inference"):
        source_names.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT / directory).rglob("*"))
                            if p.suffix in {".cpp", ".h", ".comp"})
    result = dict(status="running", date_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  scope="RGB256 adapted FP32 own-condition, own-state VAE/DiT/CFG/Euler/VAE chain",
                  unmodified_official=False, full_image_quality_validated=False,
                  official_default_sampling_validated=False, native_condition_used=True,
                  precision="FP32 storage/arithmetic; application Vulkan Conv3D and compensated attention; ncnn CPU timestep sin/cos; other ncnn layers may use their available CPU fallback",
                  image_limit=1e-3, torch_version=torch.__version__, system=platform.platform(),
                  command=[sys.executable, *sys.argv],
                  repository_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  repository_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
                  sources={name: _sha256(ROOT / name) for name in source_names}, runner_sha256=_sha256(args.runner))
    record = output / "result.json"
    try:
        torch.set_num_threads(4)
        vae_args = argparse.Namespace(**vars(args))
        vae_args.checkpoint = args.vae_checkpoint
        result["vae_reference"] = {}
        build_references(vae_args, output, result["vae_reference"])
        args.condition = output / "bf16-roundtrip-fp32-condition.pt"
        fixture = build_reference(args, output, result)
        result["conditioning"]["kind"] = "original VAE condition in reference fixture; native VAE encodes its own condition"
        records = _read_golden(fixture)
        endpoint = torch.from_numpy(records[f"step_{args.steps-1}_latent"])
        decoded, elapsed = decode_reference(args.reference_root, args.vae_checkpoint, endpoint)
        result["reference_final_decode_wall_seconds"] = elapsed
        image_fixture = output / "image.f32"
        result["image_fixture_layout"] = write_portable_golden(image_fixture, [
            ("input", image_tensor(args.image)), ("reconstruction", decoded)])
        result["image_fixture_sha256"] = _sha256(image_fixture)
        result["reference_sources"] = imported_source_hashes(args.reference_root)
        reference_rgb = save_rgb(decoded.numpy(), output / "reference.png")
        del decoded, endpoint, records
        gc.collect()
        package = args.package / "256x256"
        command = [str(ROOT / "tools/with-nvidia-vulkan-runtime.sh"), str(args.runner), str(package),
                   str(package / "cfg_64"), str(fixture), str(args.gpu_id),
                   "--image-fixture", str(image_fixture), "--output-prefix", str(output / "native")]
        result["native_command"] = command
        result["gpu"] = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu_id),
            "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()
        record.write_text(json.dumps(result, indent=2) + "\n")
        samples, rss = [gpu_memory(args.gpu_id)], []
        started = time.perf_counter()
        log = output / "native.log"
        with log.open("w") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
            while True:
                try:
                    rss.extend(int(line.split()[1]) / 1024 for line in
                               Path(f"/proc/{process.pid}/status").read_text().splitlines() if line.startswith("VmHWM:"))
                except (FileNotFoundError, ProcessLookupError):
                    pass
                try:
                    process.wait(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    samples.append(gpu_memory(args.gpu_id))
        text = log.read_text().splitlines()
        result.update(native_wall_seconds=time.perf_counter() - started, native_exit_code=process.returncode,
                      stages=[json.loads(line.removeprefix("sampling_stage ")) for line in text if line.startswith("sampling_stage ")],
                      image_stages=[json.loads(line.removeprefix("image_stage ")) for line in text if line.startswith("image_stage ")],
                      native_log_sha256=_sha256(log), device_memory_initial_mib=samples[0],
                      device_memory_sampled_peak_mib=max((s for s in samples if s is not None), default=None),
                      device_memory_sample_count=sum(s is not None for s in samples),
                      process_sampled_vmhwm_mib=max(rss, default=None),
                      memory_method="whole-device GPU memory and native /proc VmHWM sampled about every .5s; final peaks may be missed")
        result["native_runtime"] = [json.loads(line.removeprefix("sampling_runtime ")) for line in text if line.startswith("sampling_runtime ")]
        result["image_runtime"] = [json.loads(line.removeprefix("image_runtime ")) for line in text if line.startswith("image_runtime ")]
        result["sources_unchanged_at_end"] = all(_sha256(ROOT / name) == digest for name, digest in result["sources"].items())
        result["runner_unchanged_at_end"] = _sha256(args.runner) == result["runner_sha256"]
        result["fixtures_unchanged_at_end"] = (_sha256(fixture) == result["fixture_sha256"]
            and _sha256(image_fixture) == result["image_fixture_sha256"]
            and _sha256(args.condition) == result["conditioning"]["sha256"]
            and (args.initial_noise is None or _sha256(args.initial_noise) == result["initial_noise"]["sha256"]))
        if not all(result[name] for name in ("sources_unchanged_at_end", "runner_unchanged_at_end", "fixtures_unchanged_at_end")):
            raise ValueError("measured sources, binary or fixtures changed")
        if len(result["stages"]) != 3 * args.steps or len(result["image_stages"]) != 2:
            raise ValueError("incomplete own-condition image chain")
        result["native_outputs"] = {name: _sha256(output / f"native.{name}.f32")
                                    for name in ("condition", "latent", "reconstruction")}
        actual = np.fromfile(output / "native.reconstruction.f32", dtype="<f4").reshape(3, 1, 256, 256)
        native_rgb = save_rgb(actual, output / "native.png")
        difference = np.abs(native_rgb.astype(np.int16) - reference_rgb.astype(np.int16))
        result["rgb_diagnostic"] = dict(max_abs=int(difference.max()), mean_abs=float(difference.mean()),
                                        different_channels=int(np.count_nonzero(difference)), quality_validated=False)
        if process.returncode or not all(row["pass"] for row in result["stages"] + result["image_stages"]):
            raise ValueError("own-condition image chain exceeds the predeclared numerical limits")
        result["status"] = "pass"
    except Exception as error:
        result.update(status="failed", error=str(error))
        raise
    finally:
        record.write_text(json.dumps(result, indent=2) + "\n")
    print(record)


if __name__ == "__main__":
    main()
