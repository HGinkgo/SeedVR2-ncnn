#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 SeedVR2-ncnn contributors
"""Compare two fixed-shape native output images with low-precision metrics.

This is a development-only diagnostic modeled after the mixed-precision checks
reported in ncnn Discussion #6988.  It compares the same-input ncnn FP32 and
BF16 outputs; it does not compare either output with a PyTorch golden and it
does not turn image metrics into a per-layer numerical acceptance gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image.convert("RGB"), dtype=np.float64)
    if value.shape != (256, 256, 3):
        raise ValueError(f"expected RGB 256x256 image, got {value.shape} from {path}")
    if not np.isfinite(value).all():
        raise ValueError(f"image contains non-finite values: {path}")
    return value


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {candidate.shape}")
    delta = candidate - reference
    squared = delta * delta
    mse = float(np.mean(squared))
    reference_flat = reference.reshape(-1)
    candidate_flat = candidate.reshape(-1)
    denominator = float(np.linalg.norm(reference_flat) * np.linalg.norm(candidate_flat))
    cosine = float(np.dot(reference_flat, candidate_flat) / denominator) if denominator else 1.0

    # Global SSIM keeps this helper dependency-free. It is a compact image
    # diagnostic, while the tensor-level ncnn checks remain the numerical gate.
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mean_reference = float(reference.mean())
    mean_candidate = float(candidate.mean())
    variance_reference = float(reference.var())
    variance_candidate = float(candidate.var())
    covariance = float(((reference - mean_reference) * (candidate - mean_candidate)).mean())
    ssim = ((2.0 * mean_reference * mean_candidate + c1) * (2.0 * covariance + c2) /
            ((mean_reference * mean_reference + mean_candidate * mean_candidate + c1) *
             (variance_reference + variance_candidate + c2)))
    return {
        "cosine": cosine,
        "psnr_db": float("inf") if mse == 0.0 else float(10.0 * np.log10((255.0 * 255.0) / mse)),
        "ssim_global": float(ssim),
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(mse)),
        "fraction_over_1": float(np.mean(np.abs(delta) > 1.0)),
        "fraction_over_5": float(np.mean(np.abs(delta) > 5.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32", type=Path, required=True, help="same-input ncnn FP32 output PNG")
    parser.add_argument("--bf16", type=Path, required=True, help="same-input ncnn BF16 output PNG")
    parser.add_argument("--output", type=Path, help="optional JSON result path; refuses to overwrite")
    args = parser.parse_args()
    fp32 = args.fp32.resolve()
    bf16 = args.bf16.resolve()
    if not fp32.is_file() or not bf16.is_file():
        raise SystemExit("both --fp32 and --bf16 must be existing files")
    result = {
        "status": "pass",
        "scope": "same-input fixed RGB256 ncnn FP32 versus BF16 output image",
        "reference": {"path": str(fp32), "sha256": sha256(fp32), "precision": "fp32"},
        "candidate": {"path": str(bf16), "sha256": sha256(bf16), "precision": "bf16-storage-fp32-accumulation"},
        "metrics": metrics(load_rgb(fp32), load_rgb(bf16)),
        "quality_validated": False,
        "tensor_acceptance_required": True,
    }
    if args.output:
        output = args.output.resolve()
        if output.exists():
            raise SystemExit(f"refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
