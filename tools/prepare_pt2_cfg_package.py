#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 SeedVR2-ncnn contributors
"""Assemble a fixed-256 opt-in PT2 DiT package, optionally with 64-token CFG.

Keep the provenance JSON beside the package, outside its runtime-only SHA256
manifest. Source graph manifests and successful native conversion records
are required; a directory name containing 'pt2' is insufficient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

OUTPUT_MODULATION = "per-feature-six-first-triplet-v1"
LAST_BLOCK_CONTRACT = "video-only-ada-and-mlp-v1"

DIT_GRAPHS = ("dit_input", "dit_embedding", *(f"dit_block_{i:02d}" for i in range(32)), "dit_output")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_stack(root: Path, tokens: int) -> dict:
    manifest = json.loads((root / "dit_stack_manifest.json").read_text())
    if (manifest.get("frontend") != "pt2" or manifest.get("source_shape") != [1, 16, 16]
            or manifest.get("text_tokens") != tokens or manifest.get("block_count") != 32):
        raise ValueError(f"expected a true fixed-256 PT2 stack with {tokens} text tokens: {root}")
    output_metadata = json.loads((root / "dit_output.manifest.json").read_text())
    if (manifest.get("output_modulation") != OUTPUT_MODULATION
            or output_metadata.get("output_modulation") != OUTPUT_MODULATION):
        raise ValueError(f"output Ada contract is missing or obsolete; re-export dit_output: {root}")
    last_metadata = json.loads((root / "dit_block_31.manifest.json").read_text())
    if (manifest.get("last_block_contract") != LAST_BLOCK_CONTRACT
            or last_metadata.get("last_block_contract") != LAST_BLOCK_CONTRACT):
        raise ValueError(f"last block contract is missing or obsolete; re-export dit_block_31: {root}")
    graphs = {g["name"]: g for g in manifest["graphs"]}
    if len(manifest["graphs"]) != 35 or len(graphs) != 35 or set(graphs) != set(DIT_GRAPHS):
        raise ValueError(f"expected exactly 35 distinct DiT graph records: {root}")
    for name in DIT_GRAPHS:
        graph = graphs[name]
        source = root / f"{name}.pt2"
        conversion = json.loads((root / f"{name}.conversion.json").read_text())
        if (graph.get("frontend") != "pt2" or graph.get("source_sha256") != sha256(source)
                or conversion.get("status") != "success" or conversion.get("frontend") != "native"
                or conversion.get("frontend_contract", {}).get("pt2_sha256") != graph["source_sha256"]):
            raise ValueError(f"PT2 source/conversion provenance mismatch: {root / name}")
        for suffix in (".ncnn.param", ".ncnn.bin"):
            if not (root / f"{name}{suffix}").is_file():
                raise FileNotFoundError(root / f"{name}{suffix}")
    return manifest


def prepare(base: Path, positive: Path, negative: Path | None, conditioning: Path | None,
            output: Path) -> Path:
    base, positive, output = base.resolve(), positive.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists; choose a new directory: {output}")
    if (negative is None) != (conditioning is None):
        raise ValueError("--negative-stack and --negative-conditioning are required together")
    manifests = {"positive": inspect_stack(positive, 58)}
    if negative is not None:
        negative = negative.resolve()
        manifests["negative"] = inspect_stack(negative, 64)
        if manifests["positive"]["checkpoint_sha256"] != manifests["negative"]["checkpoint_sha256"]:
            raise ValueError("positive and negative stacks must use the same checkpoint")
        if conditioning.stat().st_size != 64 * 5120 * 4:
            raise ValueError("negative conditioning must contain exactly 64x5120 float32 values")
    pos_conditioning = base / "conditioning/pos_emb.f32"
    if pos_conditioning.stat().st_size != 58 * 5120 * 4:
        raise ValueError("positive conditioning must contain exactly 58x5120 float32 values")

    output.parent.mkdir(parents=True, exist_ok=True)
    provenance = output.with_name(output.name + ".provenance.json")
    if provenance.exists():
        raise FileExistsError(provenance)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        # Use the registry's existing resolution-variant contract. A flat
        # package would also resolve for other sizes despite these static
        # PT2 graphs having been exported only for 256x256.
        graph_dir = staging / "256x256"
        graph_dir.mkdir()
        for name in ("vae_encode", "vae_decode"):
            for suffix in (".ncnn.param", ".ncnn.bin"):
                shutil.copy2(base / f"{name}{suffix}", graph_dir / f"{name}{suffix}")
        for source, target in [(positive, graph_dir)] + ([(negative, graph_dir / "cfg_64")] if negative else []):
            target.mkdir(exist_ok=True)
            for name in DIT_GRAPHS:
                for suffix in (".ncnn.param", ".ncnn.bin"):
                    shutil.copy2(source / f"{name}{suffix}", target / f"{name}{suffix}")
        (staging / "conditioning").mkdir()
        shutil.copy2(pos_conditioning, staging / "conditioning/pos_emb.f32")
        if conditioning is not None:
            shutil.copy2(conditioning, staging / "conditioning/neg_emb.f32")
        hashes = {p.relative_to(staging).as_posix(): sha256(p) for p in sorted(staging.rglob("*")) if p.is_file()}
        if len(hashes) != (146 if negative else 75):
            raise ValueError("unexpected runtime package artifact count")
        (staging / "manifest.sha256").write_text("".join(f"{value}  {name}\n" for name, value in hashes.items()))
        record = {"kind": "fixed-256-pt2-dit-package", "base_vae_package": str(base),
                  "vae_provenance": "inherited; inspect the base VAE conversion records separately",
                  "positive_stack": str(positive), "negative_stack": str(negative) if negative else None,
                  "stack_manifests": manifests, "runtime_sha256": hashes}
        staging.rename(output)
        provenance.write_text(json.dumps(record, indent=2) + "\n")
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-package", type=Path, required=True, help="source of VAE graphs and positive conditioning")
    parser.add_argument("--positive-stack", type=Path, required=True)
    parser.add_argument("--negative-stack", type=Path)
    parser.add_argument("--negative-conditioning", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.base_package, args.positive_stack, args.negative_stack,
                  args.negative_conditioning, args.output_dir))


if __name__ == "__main__":
    main()
