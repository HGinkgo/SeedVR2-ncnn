#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Generate the canonical fixed RGB256 SeedVR2 reference golden.

This is a reference-only tool. It imports the pinned upstream VAE and the
reviewed independent CPU DiT adapter; neither is part of the published
runtime. It generates the FP32 source anchor only. The final PNG is decoded
from the sampler endpoint; this is not an official CUDA quality baseline.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.compare_reference_vae import image_tensor
from tools.compare_reference_dit import _AttentionPrecision
from tools.compare_reference_sampling import model_video, patch_output
from tools.export_dit_stack import _sha256
from tools.reference.imported_dit import load_reference_dit, imported_source_hashes
from tools.reference.portable_f32 import write_portable_golden


def _write_rgb(value: torch.Tensor, path: Path) -> None:
    if tuple(value.shape) != (3, 1, 256, 256):
        raise ValueError(f"expected [3,1,256,256], got {tuple(value.shape)}")
    rgb = np.rint((value[:, 0].permute(1, 2, 0).clamp(-1, 1).numpy() + 1) * 127.5).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def _load_vae(reference_root: Path, checkpoint: Path, precision: str):
    # The upstream VAE is intentionally constructed in the frozen source tree.
    if str(reference_root) not in sys.path:
        sys.path.insert(0, str(reference_root))
    previous = Path.cwd()
    os.chdir(reference_root)
    try:
        from common.config import create_object, load_config

        config = load_config("configs_3b/main.yaml")
        model = create_object(config.vae.model).float().cpu().requires_grad_(False).eval()
    finally:
        os.chdir(previous)
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if len(state) != 250 or any(value.dtype != torch.float32 for value in state.values()):
        raise ValueError("expected the reviewed 250-entry FP32 VAE checkpoint")
    model.load_state_dict(state, strict=True)
    del state
    if precision == "bf16":
        model.bfloat16()
    model.set_causal_slicing(split_size=None, memory_device="same")
    model.set_memory_limit(conv_max_mem=None, norm_max_mem=None)
    if float(config.vae.scaling_factor) != 0.9152 or float(config.vae.get("shifting_factor", 0)) != 0:
        raise ValueError("unexpected VAE latent scaling")
    return model, config


def _run_vae(reference_root: Path, checkpoint: Path, image: Path, precision: str, output: Path):
    sample = image_tensor(image)
    model, config = _load_vae(reference_root, checkpoint, precision)
    scale = float(config.vae.scaling_factor)
    shift = float(config.vae.get("shifting_factor", 0.0))
    started = time.perf_counter()
    with torch.inference_mode():
        encoded = model.encode(model.preprocess(sample.unsqueeze(0).to(model.dtype)))
        latent = (encoded.posterior.mode().squeeze(2) - shift) * scale
    elapsed = time.perf_counter() - started
    latent = latent.squeeze(0).unsqueeze(1).contiguous().float().cpu()
    if tuple(latent.shape) != (16, 1, 32, 32):
        raise ValueError("unexpected VAE output shape")
    if not torch.isfinite(latent).all():
        raise ValueError("VAE produced non-finite values")
    output.mkdir(parents=True, exist_ok=True)
    encode_layout = write_portable_golden(output / "vae_encode.f32", [("input", sample), ("latent", latent)])
    torch.save(latent, output / "condition.pt")
    del model, encoded
    gc.collect()
    return {
        "precision": precision,
        "weight_arithmetic": "FP32 parameters and execution" if precision == "fp32" else
            "BF16 parameters, inputs and execution; upstream explicit FP32 promotions retained",
        "wall_seconds": elapsed,
        "latent_shape": list(latent.shape),
        "vae_encode": {"path": "vae_encode.f32", "sha256": _sha256(output / "vae_encode.f32"), "layout": encode_layout},
        "condition": {"path": "condition.pt", "sha256": _sha256(output / "condition.pt")},
    }


def _decode_endpoint(reference_root: Path, checkpoint: Path, precision: str, output: Path):
    endpoint = torch.load(output / "sampled_latent.pt", map_location="cpu", weights_only=True)
    if tuple(endpoint.shape) != (16, 1, 32, 32) or not torch.isfinite(endpoint).all():
        raise ValueError("sampler endpoint must be a finite [16,1,32,32] latent")
    model, config = _load_vae(reference_root, checkpoint, precision)
    with torch.inference_mode():
        latent = endpoint[:, 0].unsqueeze(0).to(model.dtype)
        latent = latent / float(config.vae.scaling_factor) + float(config.vae.get("shifting_factor", 0.0))
        reconstruction = model.decode(latent).sample
        if hasattr(model, "postprocess"):
            reconstruction = model.postprocess(reconstruction)
    reconstruction = reconstruction.squeeze(0).unsqueeze(1).float().cpu().contiguous()
    if tuple(reconstruction.shape) != (3, 1, 256, 256) or not torch.isfinite(reconstruction).all():
        raise ValueError("VAE decoder produced an invalid image")
    layout = write_portable_golden(output / "vae_decode.f32", [("latent", endpoint), ("reconstruction", reconstruction)])
    _write_rgb(reconstruction, output / "output.png")
    del model
    gc.collect()
    return {
        "input": "sampled_latent.f32 (Euler/CFG endpoint, not VAE reconstruction)",
        "vae_decode": {"path": "vae_decode.f32", "sha256": _sha256(output / "vae_decode.f32"), "layout": layout},
        "output": {"path": "output.png", "sha256": _sha256(output / "output.png")},
    }


def _run_dit(reference_root: Path, checkpoint: Path, package: Path, condition: Path,
             precision: str, args: argparse.Namespace, output: Path):
    condition_value = torch.load(condition, map_location="cpu", weights_only=True).float().contiguous()
    if tuple(condition_value.shape) != (16, 1, 32, 32):
        raise ValueError("condition must be [16,1,32,32]")
    text = {}
    for sign, tokens in (("positive", 58), ("negative", 64)):
        path = package / "conditioning" / ("pos_emb.f32" if sign == "positive" else "neg_emb.f32")
        raw = np.fromfile(path, dtype="<f4")
        if raw.size != tokens * 5120:
            raise ValueError(f"invalid {sign} text embedding size")
        text[sign] = torch.from_numpy(raw.reshape(tokens, 5120).copy())
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn((16, 1, 32, 32), generator=generator, dtype=torch.float32)
    if precision == "bf16":
        condition_value = condition_value.bfloat16()
        # Use the same saved CPU noise sample for both precisions. This is a
        # deterministic conversion check, not upstream CUDA RNG reproduction.
        noise = noise.bfloat16()
        text = {sign: value.bfloat16() for sign, value in text.items()}
    model, config, reference_metadata = load_reference_dit(reference_root, checkpoint, 64)
    if precision == "fp32":
        reference_metadata["adaptations"] = list(reference_metadata["adaptations"]) + [
            "96 DiT Q/K/V BF16 casts bypassed for the FP32 architecture anchor"
        ]
    from common.diffusion.config import create_schedule_from_config, create_sampler_from_config, create_sampling_timesteps_from_config
    from common.diffusion.utils import classifier_free_guidance

    config.diffusion.timesteps.sampling.steps = args.steps
    schedule = create_schedule_from_config(config.diffusion.schedule, torch.device("cpu"))
    timesteps = create_sampling_timesteps_from_config(config.diffusion.timesteps.sampling, schedule, torch.device("cpu"))
    sampler = create_sampler_from_config(config.diffusion.sampler, schedule, timesteps)
    records = {}

    def predict(arguments):
        video = model_video(arguments.x_t, condition_value)
        predictions = {}
        for sign, tokens in (("positive", 58), ("negative", 64)):
            raw = model(video, text[sign], torch.tensor([[1, 32, 32]]), torch.tensor([[tokens]]), arguments.t.reshape(1)).vid_sample
            latent = raw.reshape(1, 32, 32, 16).permute(3, 0, 1, 2).contiguous()
            predictions[sign] = latent
            records[f"dit_{sign}_step_{arguments.i:02d}"] = patch_output(latent)
        cfg = classifier_free_guidance(predictions["positive"], predictions["negative"], scale=args.cfg_scale, rescale=0.0)
        records[f"cfg_step_{arguments.i:02d}"] = cfg.clone()
        return cfg

    precision_context = (_AttentionPrecision(bypass=True) if precision == "fp32" else
                         torch.autocast("cpu", torch.bfloat16))
    with torch.inference_mode(), precision_context:
        endpoint = sampler.sample(noise.clone(), predict)
    if args.steps != 1:
        raise ValueError("canonical golden is intentionally one-step; use a separate audit for longer sampling")
    output.mkdir(parents=True, exist_ok=True)
    layouts = {}
    for name in ("dit_positive_step_00", "dit_negative_step_00", "cfg_step_00"):
        path = output / f"{name}.f32"
        layouts[name] = write_portable_golden(path, [(name, records[name])])
    write_portable_golden(output / "initial_noise.f32", [("initial_noise", noise)])
    write_portable_golden(output / "condition.f32", [("condition", condition_value)])
    write_portable_golden(output / "sampled_latent.f32", [("sampled_latent", endpoint)])
    torch.save(endpoint.float(), output / "sampled_latent.pt")
    del model, sampler, endpoint
    gc.collect()
    return {
        "precision": precision,
        "weight_arithmetic": "FP32 parameters, buffers and execution; QKV casts bypassed" if precision == "fp32" else
            "FP32 parameters and buffers; CPU BF16 autocast; upstream QKV casts retained",
        "seed": args.seed,
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "timestep": timesteps.timesteps.tolist(),
        "reference": reference_metadata,
        "text_conditioning": "package conditioning/pos_emb.f32 and neg_emb.f32",
        "condition_sha256": _sha256(condition),
        "files": {name: {"path": f"{name}.f32", "sha256": _sha256(output / f"{name}.f32"), "layout": layout}
                  for name, layout in layouts.items()},
        "initial_noise": {"path": "initial_noise.f32", "sha256": _sha256(output / "initial_noise.f32")},
        "condition": {"path": "condition.f32", "sha256": _sha256(output / "condition.f32")},
        "sampled_latent": {"path": "sampled_latent.f32", "sha256": _sha256(output / "sampled_latent.f32")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--dit-checkpoint", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True, help="package root containing conditioning/")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32",), required=True)
    parser.add_argument("--steps", type=int, choices=(1,), default=1)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=20261019)
    args = parser.parse_args()
    for name in ("reference_root", "dit_checkpoint", "vae_checkpoint", "package", "image", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    for path in (args.dit_checkpoint, args.vae_checkpoint, args.image,
                 args.package / "conditioning/pos_emb.f32", args.package / "conditioning/neg_emb.f32"):
        if not path.is_file():
            parser.error(f"missing input file: {path}")
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    vae = _run_vae(args.reference_root, args.vae_checkpoint, args.image, args.precision, output)
    dit = _run_dit(args.reference_root, args.dit_checkpoint, args.package, output / "condition.pt",
                   args.precision, args, output)
    vae.update(_decode_endpoint(args.reference_root, args.vae_checkpoint, args.precision, output))
    manifest = {
        "kind": "seedvr2-fixed-rgb256-reference-golden",
        "precision": args.precision,
        "source_image_sha256": _sha256(args.image),
        "reference_commit": (args.reference_root / ".source-commit").read_text().strip(),
        "dit_checkpoint_sha256": _sha256(args.dit_checkpoint),
        "vae_checkpoint_sha256": _sha256(args.vae_checkpoint),
        "scope": "one-step fixed 256x256 RGB image; posterior mode; CFG; adapted CPU DiT",
        "unmodified_official": False,
        "official_upstream_vae": True,
        "quality_validated": False,
        "precision_semantics": {"vae": vae["weight_arithmetic"], "dit": dit["weight_arithmetic"]},
        "reference_device": "cpu",
        "generator_sha256": _sha256(Path(__file__)),
        "generation_command": [sys.executable, *sys.argv],
        "torch_version": torch.__version__,
        "vae": vae,
        "dit": dit,
        "source_hashes": imported_source_hashes(args.reference_root),
        "generation_seconds": time.perf_counter() - started,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(output / "manifest.json")


if __name__ == "__main__":
    main()
