#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 SeedVR2-ncnn contributors
"""Compare every fixed DiT boundary with an imported, explicitly adapted reference.

Imports the separately supplied Apache-2.0 SeedVR checkout without copying or
editing it. This CPU test needs its conversion-only dependencies (diffusers,
omegaconf, rotary_embedding_torch, einops); the published runtime does not.
FP32 attention is an architecture comparison, not FlashAttention/BF16 or
full-sampler quality acceptance. All compatibility changes are recorded.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import types

import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.export_dit_stack import _sha256
from tools.reference.dit_block import DIT_LAST_BLOCK_CONTRACT
from tools.reference.dit_stack import DIT_OUTPUT_MODULATION
from tools.validate_dit_stack import _read_golden

PINNED_REFERENCE = "e4de8c24441a67e1b7df56abea10645059bb1185"


def _sdpa_windows(q, k, v, cu_seqlens_q, cu_seqlens_k, **kwargs):
    # Observe casts in the imported model, not adapter-owned output casts.
    with _disable_current_modes():
        return _sdpa_windows_impl(q, k, v, cu_seqlens_q, cu_seqlens_k, **kwargs)


def _sdpa_windows_impl(q, k, v, cu_seqlens_q, cu_seqlens_k, **kwargs):
    if kwargs.get("dropout_p", 0) != 0 or kwargs.get("causal", False):
        raise ValueError("reference adapter supports unmasked, zero-dropout attention only")
    for name in ("window_size", "softcap", "alibi_slopes", "return_attn_probs"):
        if name in kwargs:
            raise ValueError(f"unsupported reference attention option: {name}")
    q_offsets, k_offsets = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    if len(q_offsets) != len(k_offsets) or q_offsets[0] != 0 or k_offsets[0] != 0:
        raise ValueError("invalid variable-length attention offsets")
    if q_offsets[-1] != len(q) or k_offsets[-1] != len(k) or len(k) != len(v):
        raise ValueError("attention offsets do not cover Q/K/V")
    outputs = []
    for a, b, c, d in zip(q_offsets[:-1], q_offsets[1:], k_offsets[:-1], k_offsets[1:]):
        if b <= a or d <= c:
            raise ValueError("empty attention windows are unsupported")
        output = torch.nn.functional.scaled_dot_product_attention(
            q[a:b].transpose(0, 1)[None].float(),
            k[c:d].transpose(0, 1)[None].float(),
            v[c:d].transpose(0, 1)[None].float(),
            scale=kwargs.get("softmax_scale"))
        outputs.append(output[0].transpose(0, 1).to(q.dtype))
    return torch.cat(outputs)


class _AttentionPrecision(TorchDispatchMode):
    def __init__(self, bypass):
        super().__init__()
        self.bypass, self.casts = bypass, 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        cast = ((func == torch.ops.aten._to_copy.default and kwargs.get("dtype") == torch.bfloat16)
                or (func == torch.ops.aten.to.dtype and len(args) > 1 and args[1] == torch.bfloat16))
        if cast:
            if args[0].ndim != 3 or args[0].shape[-1] != 128:
                raise ValueError("unexpected BF16 cast outside the reviewed Q/K/V boundary")
            self.casts += 1
            if self.bypass:
                return args[0]
        return func(*args, **kwargs)


def compare(args, result):
    reference, stack, checkpoint = args.reference_root.resolve(), args.stack.resolve(), args.checkpoint.resolve()
    commit_file = reference / ".source-commit"
    commit = (commit_file.read_text().strip() if commit_file.is_file() else
              subprocess.check_output(["git", "-C", str(reference), "rev-parse", "HEAD"], text=True).strip())
    if commit != PINNED_REFERENCE:
        raise ValueError("unreviewed reference commit; use the documented pinned snapshot")
    manifest = json.loads((stack / "dit_stack_manifest.json").read_text())
    if (manifest.get("source_shape") != [1, 16, 16] or manifest.get("text_tokens") not in (58, 64)
            or manifest.get("block_count") != 32 or manifest.get("output_modulation") != DIT_OUTPUT_MODULATION
            or manifest.get("last_block_contract") != DIT_LAST_BLOCK_CONTRACT):
        raise ValueError("expected a corrected fixed-256, 32-block DiT stack")
    checkpoint_hash, golden_hash = _sha256(checkpoint), _sha256(stack / "dit_stack_golden.f32")
    if checkpoint_hash != manifest.get("checkpoint_sha256") or golden_hash != manifest.get("golden_f32_sha256"):
        raise ValueError("checkpoint/golden SHA256 differs from stack manifest")
    golden = _read_golden(stack / "dit_stack_golden.f32")
    result.update(reference_commit=commit, reference_root=str(reference), checkpoint_sha256=checkpoint_hash,
                  stack_manifest_sha256=_sha256(stack / "dit_stack_manifest.json"), golden_sha256=golden_hash,
                  precision=args.attention_precision, dtype="float32 except explicitly retained BF16 Q/K/V",
                  inputs={k: list(golden[k].shape) for k in ("input_video_patches", "input_text", "timestep")},
                  adaptations=["official non-fused rms/layer norm alternatives",
                               "FlashAttention import replaced by independent per-window CPU SDPA",
                               "bounded lang RoPE prefix, checked equal on nested grids",
                               "output Ada layers=['out','unused']; first per-feature triplet",
                               "unused diffusers transformers integration disabled in this process"],
                  peak_gpu_memory_mib=None, device="cpu")
    if args.attention_precision == "fp32":
        result["adaptations"].append("96 Q/K/V BF16 casts bypassed for FP32 architecture comparison")
    sys.path.insert(0, str(reference))
    flash = types.ModuleType("flash_attn")
    flash.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    flash.flash_attn_varlen_func = _sdpa_windows
    sys.modules["flash_attn"] = flash
    import diffusers.utils.import_utils as integration
    integration._transformers_available = False
    from common.config import load_config
    from models.dit_v2.nadit import NaDiT
    from omegaconf import OmegaConf
    if Path(sys.modules[NaDiT.__module__].__file__).resolve() != reference / "models/dit_v2/nadit.py":
        raise ValueError("a different reference model is already imported")
    previous = Path.cwd()
    try:
        os.chdir(reference)
        config = load_config("configs_3b/main.yaml")
        options = OmegaConf.to_container(config.dit.model, resolve=True)
    finally:
        os.chdir(previous)
    options.pop("__object__")
    options.update(norm="rms", qk_norm="rms", vid_out_norm="rms", txt_in_norm="layer")
    with torch.device("meta"):
        model = NaDiT(**options)
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    loaded = model.load_state_dict(state, strict=True, assign=True)
    result.update(checkpoint_entries=len(state), missing_keys=loaded.missing_keys, unexpected_keys=loaded.unexpected_keys)
    for module in model.modules():
        for key, value in list(module._buffers.items()):
            if value is not None and value.device.type == "meta":
                if key != "dummy":
                    raise ValueError(f"unexpected uninitialized reference buffer: {key}")
                module._buffers[key] = torch.tensor(0)
    text_tokens = manifest["text_tokens"]
    for block in model.blocks:
        rotary = block.attn.rope.rope
        if rotary.freqs_for != "lang":
            raise ValueError("bounded RoPE requires language arange frequencies")
        big, small = rotary.get_axial_freqs(9, 7, 6), rotary.get_axial_freqs(6, 4, 3)
        torch.testing.assert_close(big[:6, :4, :3], small, atol=0, rtol=0)
        original = block.attn.rope.get_axial_freqs

        def bounded(*dims, _original=original):
            if dims == (1024, 128, 128):
                return _original(text_tokens + 1, 16, 16)
            if dims == (1024,):
                return _original(text_tokens)
            raise ValueError(f"unreviewed reference RoPE grid: {dims}")

        block.attn.rope.get_axial_freqs = bounded
    model.vid_out_ada.layers = ["out", "unused"]
    model.requires_grad_(False).eval()
    rows = result["stages"] = []

    def check(name, value):
        expected = torch.from_numpy(golden[name])
        delta = value.detach().cpu().float() - expected
        peak, maximum = float(expected.abs().max()), float(delta.abs().max())
        limit = 2e-5 + 2e-6 * peak
        row = dict(stage=name, max_abs=maximum, rmse=float(delta.square().mean().sqrt()),
                   reference_peak=peak, limit=limit, finite=bool(torch.isfinite(value).all()),
                   within_fp32_limit=maximum <= limit)
        rows.append(row)
        print(json.dumps(row), flush=True)

    model.vid_in.register_forward_hook(lambda m, i, o: check("input_video_hidden", o[0]))
    model.txt_in.register_forward_hook(lambda m, i, o: check("input_text_hidden", o))
    model.emb_in.register_forward_hook(lambda m, i, o: check("embedding", o))
    for index, block in enumerate(model.blocks):
        def hook(m, i, o, _index=index):
            check(f"block_{_index:02d}_video", o[0])
            check(f"block_{_index:02d}_text", o[1])
        block.register_forward_hook(hook)
    video = torch.from_numpy(golden["input_video_patches"]).reshape(16, 16, 2, 2, 33)
    video = video.permute(0, 2, 1, 3, 4).reshape(1024, 33)
    precision = _AttentionPrecision(bypass=args.attention_precision == "fp32")
    started = time.perf_counter()
    with torch.inference_mode(), precision:
        output = model(video, torch.from_numpy(golden["input_text"]), torch.tensor([[1, 32, 32]]),
                       torch.tensor([[text_tokens]]), torch.from_numpy(golden["timestep"])).vid_sample
        output = output.reshape(16, 2, 16, 2, 16).permute(0, 2, 1, 3, 4).reshape(256, 64)
        check("output_video", output)
    result.update(wall_seconds=time.perf_counter() - started, attention_casts_observed=precision.casts)
    result["peak_process_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    result["dependency_versions"] = {name: importlib.metadata.version(name) for name in
                                     ("torch", "diffusers", "omegaconf", "einops", "rotary_embedding_torch")}
    result["cpu_threads"] = torch.get_num_threads()
    result["reference_sources"] = {str(Path(m.__file__).resolve().relative_to(reference)): _sha256(Path(m.__file__))
        for m in list(sys.modules.values()) if getattr(m, "__file__", None)
        and Path(m.__file__).resolve().is_relative_to(reference) and Path(m.__file__).is_file()}
    for name in ("LICENSE", "configs_3b/main.yaml"):
        result["reference_sources"][name] = _sha256(reference / name)
    if precision.casts != 96 or len(rows) != 68 or not all(row["finite"] for row in rows):
        raise RuntimeError("incomplete reference execution or non-finite outputs")
    if args.attention_precision == "fp32" and not all(row["within_fp32_limit"] for row in rows):
        raise RuntimeError("reference architecture comparison exceeded its recorded FP32 limits")
    result["status"] = "pass" if args.attention_precision == "fp32" else "diagnostic-completed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-root", "stack", "checkpoint", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--attention-precision", choices=("fp32", "bf16"), default="fp32")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status="running", unmodified_official=False, scope="adapted CPU DiT architecture only",
                  command=[sys.executable, *sys.argv], torch_version=torch.__version__, system=platform.platform(),
                  repository_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  repository_dirty=bool(subprocess.check_output(["git", "status", "--porcelain", "-uno"], cwd=ROOT)),
                  sources={str(p.relative_to(ROOT)): _sha256(p) for p in
                           (Path(__file__), ROOT / "tools/reference/dit_block.py", ROOT / "tools/reference/dit_stack.py")})
    try:
        compare(args, result)
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
