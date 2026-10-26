#!/usr/bin/env python3

# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

"""Small ExportedProgram-to-ncnn conversion and weight regression."""

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 3)

    def forward(self, value):
        return torch.tanh(self.proj(value))


def load_generated_model(path):
    spec = importlib.util.spec_from_file_location("pt2_basic_pnnx", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load generated PNNX module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model().eval()


def main():
    pnnx = str(Path(sys.argv[1]).resolve())
    torch.manual_seed(17)
    model = TinyModel().eval()
    value = torch.randn(2, 4)
    expected = model(value)

    with tempfile.TemporaryDirectory(prefix="pnnx-pt2-basic-") as temporary:
        root = Path(temporary)
        archive = root / "tiny.pt2"
        torch.export.save(torch.export.export(model, (value,)), archive)

        result = subprocess.run(
            [pnnx, str(archive)], cwd=root, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True,
        )
        if result.returncode != 0:
            raise AssertionError(f"pnnx conversion failed:\n{result.stdout}")

        for suffix in (".pnnx.param", ".pnnx.bin", "_pnnx.py",
                       ".ncnn.param", ".ncnn.bin", "_ncnn.py"):
            artifact = root / f"tiny{suffix}"
            if not artifact.is_file() or artifact.stat().st_size == 0:
                raise AssertionError(f"missing or empty conversion artifact: {artifact.name}")

        param_lines = (root / "tiny.ncnn.param").read_text().splitlines()
        if not param_lines or param_lines[0].strip() != "7767517":
            raise AssertionError("generated ncnn param has an invalid header")
        counts = param_lines[1].split() if len(param_lines) > 1 else []
        if len(counts) != 2 or int(counts[0]) != len(param_lines) - 2:
            raise AssertionError("generated ncnn layer count does not match its graph")

        actual = load_generated_model(root / "tiny_pnnx.py")(value)
        if not torch.allclose(actual, expected, rtol=1e-5, atol=1e-6):
            error = float((actual - expected).abs().max())
            raise AssertionError(f"PT2 graph or weights changed during conversion: max_abs={error}")

    print("PT2 basic conversion passed: graph, weights, PNNX and ncnn artifacts")


if __name__ == "__main__":
    main()
