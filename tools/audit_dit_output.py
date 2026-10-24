#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 SeedVR2-ncnn contributors
"""Reproduce the upstream output-Ada mismatch and audit an explicit adaptation.

The reference checkout is read only. Its AdaSingle class is imported, not
copied. The adaptation changes only that instance's layer-selection metadata
from one triplet to two triplets per feature; all arithmetic stays upstream.
This is not an unmodified-official-model or sampling-quality acceptance test.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.convert_pt2 import convert
from tools.export_dit_stack import _run_pnnx, _sha256
from tools.reference.dit_stack import (DIT_OUTPUT_MODULATION, FixedDitOutput,
                                       load_dit_state, make_dit_stack_contract)
from tools.validate_dit_stack import _read_golden, _run_graph


def audit(args):
    destination = args.output_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"choose a fresh audit directory: {destination}")
    reference = args.reference_root.resolve()
    source = reference / "models/dit_v2/modulation.py"
    if not source.is_file():
        raise FileNotFoundError(source)
    sys.path.insert(0, str(reference))
    module = importlib.import_module("models.dit_v2.modulation")
    if Path(module.__file__).resolve() != source.resolve():
        raise ValueError("a different reference module is already imported")
    manifest = json.loads((args.stack / "dit_stack_manifest.json").read_text())
    checkpoint_hash = _sha256(args.checkpoint)
    if manifest.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("checkpoint SHA256 differs from the input stack manifest")
    if manifest.get("golden_f32_sha256") != _sha256(args.stack / "dit_stack_golden.f32"):
        raise ValueError("golden SHA256 differs from the input stack manifest")
    golden = _read_golden(args.stack / "dit_stack_golden.f32")
    contract = make_dit_stack_contract(manifest["text_tokens"], source_shape=tuple(manifest["source_shape"]))
    state = load_dit_state(args.checkpoint)
    model = FixedDitOutput(state, contract=contract).float().eval()
    hidden = torch.from_numpy(golden["block_31_video"])
    embedding = torch.from_numpy(golden["embedding"])
    original = module.AdaSingle(contract.hidden_dim, contract.embedding_dim, ["out"], ["in"]).eval()
    original.load_state_dict({"out_shift": model.shift, "out_scale": model.scale}, strict=True)
    normalized = torch.nn.functional.rms_norm(hidden, (contract.hidden_dim,), model.norm, eps=1e-5)
    failure = None
    with torch.inference_mode():
        try:
            original(normalized.clone(), embedding, layer="out", mode="in")
        except RuntimeError as error:
            failure = str(error)
        if failure is None or "size of tensor" not in failure:
            raise RuntimeError("pinned upstream failure did not reproduce; review the reference version")
        original.layers = ["out", "unused"]
        expected = model.proj(original(normalized.clone(), embedding, layer="out", mode="in"))
        actual = model(hidden, embedding)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        exact = float((actual - expected).abs().max())
        legacy = embedding[:, :contract.hidden_dim * 3].reshape(contract.hidden_dim, 3)
        old = model.proj(normalized * (legacy[:, 1] + model.scale) + legacy[:, 0] + model.shift)
        legacy_error = float((old - expected).abs().max())
    destination.mkdir(parents=True)
    pt2, ts = destination / "dit_output.pt2", destination / "dit_output.pt"
    torch.export.save(torch.export.export(model, (hidden, embedding)), pt2)
    torch.jit.trace(model, (hidden, embedding)).save(str(ts))
    torch.testing.assert_close(torch.export.load(pt2).module()(hidden, embedding), expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(torch.jit.load(str(ts))(hidden, embedding), expected, atol=2e-5, rtol=2e-5)
    metadata = {"exporter": "seedvr2-pt2-fixtures-v2", "model": "dit_output", "block_index": None,
                "source_shape": list(contract.source_shape), "text_tokens": contract.text_tokens,
                "checkpoint_sha256": checkpoint_hash, "torch_version": torch.__version__,
                "pt2_sha256": _sha256(pt2), "output_modulation": DIT_OUTPUT_MODULATION}
    pt2.with_suffix(".manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    result = {"status": "running", "contract": DIT_OUTPUT_MODULATION,
              "reference_root": str(reference), "reference_modulation_sha256": _sha256(source),
              "reference_nadit_sha256": _sha256(reference / "models/dit_v2/nadit.py"),
              "reference_commit": (reference / ".source-commit").read_text().strip()
                  if (reference / ".source-commit").is_file() else subprocess.check_output(
                      ["git", "-C", str(reference), "rev-parse", "HEAD"], text=True).strip(),
              "original_failure": failure, "adaptation": "AdaSingle.layers = ['out', 'unused']; layer='out'",
              "adapted_eager_max_abs": exact, "legacy_max_abs": legacy_error,
              "input_stack_manifest_sha256": _sha256(args.stack / "dit_stack_manifest.json"),
              "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "repository_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
              "torch_version": torch.__version__, "system": platform.platform(),
              "checkpoint_sha256": metadata["checkpoint_sha256"], "inputs": [list(hidden.shape), list(embedding.shape)],
              "precision": "float32", "peak_gpu_memory_mib": None, "native": [],
              "scope": "output boundary with explicit adaptation; not full official sampling",
              "input_golden_sha256": _sha256(args.stack / "dit_stack_golden.f32"),
              "input_tensor_sha256": {"hidden": hashlib_sha(hidden), "embedding": hashlib_sha(embedding)},
              "runtime_ncnn_commit": subprocess.check_output(
                  ["git", "-C", str(ROOT / "third_party/ncnn"), "rev-parse", "HEAD"], text=True).strip(),
              "command": [sys.executable, *sys.argv], "sources": {str(p.relative_to(ROOT)): _sha256(p) for p in
                  (ROOT / "tools/reference/dit_stack.py", ROOT / "tools/audit_dit_output.py", ROOT / "tools/convert_pt2.py")}}
    record = destination / "result.json"
    record.write_text(json.dumps(result, indent=2) + "\n")
    if args.pnnx:
        if not args.runner:
            raise ValueError("--pnnx requires --runner for native numerical acceptance")
        pt2_directory = destination / "pt2"
        convert(pt2, args.pnnx, pt2_directory)
        ts_directory = destination / "torchscript"
        ts_directory.mkdir()
        import shutil
        shutil.copy2(ts, ts_directory / ts.name)
        _run_pnnx(args.pnnx, ts_directory / ts.name, ts_directory,
                  f"[{hidden.shape[0]},{hidden.shape[1]}],[1,{embedding.shape[1]}]")
        result["pnnx_sha256"] = _sha256(args.pnnx)
        result["runner_sha256"] = _sha256(args.runner)
        hardware = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
                                   "--format=csv,noheader"], capture_output=True, text=True)
        result["gpu_inventory"] = hardware.stdout.strip() if hardware.returncode == 0 else None
        modes = ("cpu", "vulkan") if args.mode == "both" else (args.mode,)
        for mode in modes:
            paired = []
            for frontend, directory in (("pt2", pt2_directory), ("torchscript", ts_directory)):
                with tempfile.TemporaryDirectory(prefix="seedvr2-output-audit-") as name:
                    temporary = Path(name)
                    for suffix in (".ncnn.param", ".ncnn.bin"):
                        (temporary / f"graph{suffix}").symlink_to(directory / f"dit_output{suffix}")
                    row = _run_graph(root=ROOT, runner=args.runner.resolve(), mode=mode, name="dit_output",
                                     inputs=[("in0", hidden.numpy()), ("in1", embedding.numpy())],
                                     outputs=[("out0", expected.numpy())], temporary=temporary, verbose=False,
                                     atol=2e-5, rtol=2e-5)
                    row["frontend"], row["mode"] = frontend, mode
                    result["native"].append(row)
                    paired.append(np.fromfile(temporary / f"dit_output.{mode}.output0.f32", dtype="<f4"))
            error = float(np.max(np.abs(paired[0] - paired[1])))
            if error != 0:
                raise RuntimeError(f"PT2/TorchScript {mode} outputs differ: {error}")
            result.setdefault("paired_max_abs", {})[mode] = error
    result["status"] = "pass"
    record.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "original_failure", "adapted_eager_max_abs", "legacy_max_abs")}))


def hashlib_sha(tensor):
    import hashlib
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stack", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pnnx", type=Path)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--mode", choices=("cpu", "vulkan", "both"), default="both")
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
