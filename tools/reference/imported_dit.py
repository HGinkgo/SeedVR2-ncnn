# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 SeedVR2-ncnn contributors
"""Load the separately supplied SeedVR source with recorded CPU adaptations.

Reference code and weights are never bundled here. This helper is used only
for numerical validation, not export or the published application runtime.
"""
from __future__ import annotations

import importlib.machinery
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import types

import torch

from tools.compare_reference_dit import PINNED_REFERENCE, _sdpa_windows
from tools.export_dit_stack import _sha256


def load_reference_dit(reference: Path, checkpoint: Path, max_text_tokens: int):
    reference, checkpoint = reference.resolve(), checkpoint.resolve()
    commit_file = reference / ".source-commit"
    commit = (commit_file.read_text().strip() if commit_file.is_file() else
              subprocess.check_output(["git", "-C", str(reference), "rev-parse", "HEAD"], text=True).strip())
    if commit != PINNED_REFERENCE or max_text_tokens not in (58, 64):
        raise ValueError("expected the pinned SeedVR source and a reviewed 58/64-token bound")
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
    for module in model.modules():
        for key, value in list(module._buffers.items()):
            if value is not None and value.device.type == "meta":
                if key != "dummy":
                    raise ValueError(f"unexpected uninitialized reference buffer: {key}")
                module._buffers[key] = torch.tensor(0)
    for block in model.blocks:
        rotary = block.attn.rope.rope
        if rotary.freqs_for != "lang":
            raise ValueError("bounded RoPE requires language arange frequencies")
        big, small = rotary.get_axial_freqs(9, 7, 6), rotary.get_axial_freqs(6, 4, 3)
        torch.testing.assert_close(big[:6, :4, :3], small, atol=0, rtol=0)
        original = block.attn.rope.get_axial_freqs

        def bounded(*dims, _original=original):
            if dims == (1024, 128, 128):
                return _original(max_text_tokens + 1, 16, 16)
            if dims == (1024,):
                return _original(max_text_tokens)
            raise ValueError(f"unreviewed reference RoPE grid: {dims}")

        block.attn.rope.get_axial_freqs = bounded
    model.vid_out_ada.layers = ["out", "unused"]
    model.requires_grad_(False).eval()
    metadata = dict(reference_commit=commit, checkpoint_entries=len(state),
                    missing_keys=loaded.missing_keys, unexpected_keys=loaded.unexpected_keys,
                    adaptations=["official non-fused rms/layer norm alternatives",
                                 "independent per-window CPU FP32 SDPA instead of FlashAttention",
                                 "bounded language RoPE prefix checked on nested grids",
                                 "output Ada layers=['out','unused']; first per-feature triplet",
                                 "unused diffusers transformers integration disabled in this process"],
                    dependency_versions={name: importlib.metadata.version(name) for name in
                                         ("torch", "diffusers", "omegaconf", "einops", "rotary_embedding_torch")})
    return model, config, metadata


def imported_source_hashes(reference: Path):
    reference = reference.resolve()
    result = {str(Path(m.__file__).resolve().relative_to(reference)): _sha256(Path(m.__file__))
              for m in list(sys.modules.values()) if getattr(m, "__file__", None)
              and Path(m.__file__).resolve().is_relative_to(reference) and Path(m.__file__).is_file()}
    for name in ("LICENSE", "configs_3b/main.yaml"):
        result[name] = _sha256(reference / name)
    return result
