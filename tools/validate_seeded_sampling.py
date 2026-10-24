#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 SeedVR2-ncnn contributors
"""Validate opt-in normal noise on matched fixed-256 TS/PT2 Vulkan packages.

Runs sequentially on one GPU. Tests deployment reproducibility and frontend
consistency; it does not compare RNG sequences or image quality with PyTorch.
An optional historical result checks that the no-seed fixture is unchanged.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import platform
import struct
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.run_pt2_cfg_experiment import ROOT, capture, run_case, sha256
from tools.validate_pt2_cfg_runtime import compare, run_video


def avi_frame_hashes(path: Path) -> list[str]:
    """Read the native writer's RGB24 chunks with RIFF bounds/padding checks."""
    data = path.read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"AVI ":
        raise ValueError("expected native RIFF AVI")
    if len(data) != struct.unpack_from("<I", data, 4)[0] + 8:
        raise ValueError("AVI length differs from RIFF size")
    frames = []

    def chunks(start: int, end: int, in_movi: bool = False) -> None:
        cursor = start
        while cursor < end:
            if cursor + 8 > end:
                raise ValueError("truncated AVI chunk header")
            kind, size = struct.unpack_from("<4sI", data, cursor)
            begin, finish = cursor + 8, cursor + 8 + size
            padded = finish + (size & 1)
            if padded > end:
                raise ValueError("AVI chunk exceeds container")
            if kind == b"LIST":
                if size < 4:
                    raise ValueError("truncated AVI LIST type")
                group = data[begin:begin + 4]
                chunks(begin + 4, finish, in_movi or group == b"movi")
            elif in_movi and kind == b"00db":
                if size != 256 * 256 * 3:
                    raise ValueError("expected 256x256 RGB24 AVI frame")
                frames.append(hashlib.sha256(data[begin:finish]).hexdigest())
            cursor = padded

    chunks(12, len(data))
    if not frames:
        raise ValueError("AVI has no native RGB24 frames")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--control-package", required=True, type=Path,
                        help="TorchScript DiT with identical PT2 VAE and conditioning")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--video-input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="new evidence directory")
    parser.add_argument("--legacy-results", type=Path, help="passing pre-seed image experiment JSON")
    parser.add_argument("--binary", type=Path, default=ROOT / "build/seedvr2-ncnn")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=666)
    args = parser.parse_args()
    if args.gpu_id < 0 or not 0 <= args.seed <= 0xffffffff:
        parser.error("select an explicit GPU and a seed in [0, 4294967295]")
    binary, package, control = (path.resolve() for path in
                                (args.binary, args.package, args.control_package))
    image, video = args.input.resolve(), args.video_input.resolve()
    legacy = None
    if args.legacy_results:
        primary = json.loads(args.legacy_results.read_text())
        legacy = next((row for row in primary.get("runs", [])
                       if row.get("name") == "pt2-cfg1.5-s2"), None)
        phases = {f"{phase}:0" for phase in
                  ("condition-latent", "denoised-latent", "reconstruction")}
        if (primary.get("status") != "pass" or primary.get("input_sha256") != sha256(image)
                or primary.get("gpu_id") != args.gpu_id or not legacy
                or legacy.get("status") != "pass" or legacy.get("seed") is not None
                or legacy.get("package_manifest_sha256") != sha256(package / "manifest.sha256")
                or set(legacy.get("tensor_fingerprints", {})) != phases
                or sha256(Path(legacy["output"])) != legacy["output_sha256"]):
            parser.error("legacy evidence must pass for the same input/manifest/GPU and contain three handoffs")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = [ROOT / "CMakeLists.txt", ROOT / "tools/with-nvidia-vulkan-runtime.sh",
               ROOT / "tools/run_pt2_cfg_experiment.py", ROOT / "tools/validate_pt2_cfg_runtime.py",
               Path(__file__).resolve()]
    sources += sorted(path for path in (ROOT / "src").rglob("*") if path.is_file())
    record = {
        "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "repository_commit": capture(["git", "rev-parse", "HEAD"]),
        "repository_dirty": bool(capture(["git", "status", "--porcelain"])),
        "system": platform.platform(), "python": platform.python_version(),
        "ncnn_commit": capture(["git", "-C", "third_party/ncnn", "rev-parse", "HEAD"]),
        "ncnn_submodule_dirty": bool(capture(["git", "-C", "third_party/ncnn", "status", "--porcelain"])),
        "binary_sha256": sha256(binary), "runtime_ldd": capture(["ldd", str(binary)]),
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "input_sha256": sha256(image), "video_input_sha256": sha256(video),
        "candidate_manifest_sha256": sha256(package / "manifest.sha256"),
        "control_manifest_sha256": sha256(control / "manifest.sha256"),
        "gpu_id": args.gpu_id,
        "gpu": capture(["nvidia-smi", "-i", str(args.gpu_id),
                        "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"]),
        "noise_algorithm": "mt19937-box-muller-v1", "seed": args.seed,
        "rng_scope": "same binary/libm, logical C,D,H,W shape and absolute frame; not PyTorch RNG compatible",
        "precision": "runtime BF16 storage/FP32 arithmetic", "output_size": [256, 256],
        "sampling": {"steps": 2, "cfg_scale": 1.5},
        "quality_scope": "deployment contracts only; official sampler/temporal quality not accepted",
        "status": "running", "runs": [], "comparisons": {}}
    if args.legacy_results:
        record["legacy_result"] = {"path": str(args.legacy_results.resolve()),
                                    "sha256": sha256(args.legacy_results),
                                    "binary_sha256": primary["binary_sha256"]}
    result = output / "result.json"

    def save() -> None:
        result.write_text(json.dumps(record, indent=2) + "\n")

    def image_case(name: str, model: Path, seed: int | None) -> dict:
        row = run_case(binary, model, image, output, args.gpu_id, name, 2, 1.5, seed=seed)
        record["runs"].append(row)
        save()
        if row["status"] != "pass":
            raise RuntimeError(f"{name}: {row['status']}; inspect {row['log']}")
        return row

    def exact(name: str, first: dict, second: dict) -> None:
        metrics = compare(Path(first["output"]), Path(second["output"]))
        record["comparisons"][name] = metrics
        if (metrics["max_abs"] != 0 or metrics["first_sha256"] != metrics["second_sha256"]
                or first["tensor_fingerprints"] != second["tensor_fingerprints"]):
            raise RuntimeError(f"{name}: tensor handoffs or PNG differ")
        save()

    save()
    try:
        for model in (package, control):
            subprocess.run([str(binary), "--check-model", "--model-dir", str(model)], cwd=ROOT, check=True)
        shared = [f"256x256/{name}.ncnn.{suffix}" for name in ("vae_encode", "vae_decode")
                  for suffix in ("param", "bin")]
        shared += [f"conditioning/{sign}_emb.f32" for sign in ("pos", "neg")]
        record["matched_vae_and_conditioning_hashes"] = {name: sha256(package / name) for name in shared}
        if any(sha256(control / name) != digest for name, digest in
               record["matched_vae_and_conditioning_hashes"].items()):
            raise ValueError("candidate and control must use identical VAE/conditioning")
        if legacy:
            exact("legacy_fixture_unchanged", legacy, image_case("pt2-legacy", package, None))
        ts = image_case(f"torchscript-seed{args.seed}", control, args.seed)
        pt2 = image_case(f"pt2-seed{args.seed}", package, args.seed)
        exact("matched_frontends", ts, pt2)
        repeated = image_case(f"pt2-seed{args.seed}-repeat", package, args.seed)
        exact("same_seed_repeat", pt2, repeated)
        changed_seed = (args.seed + 1) & 0xffffffff
        changed = image_case(f"pt2-seed{changed_seed}", package, changed_seed)
        fingerprints, other = pt2["tensor_fingerprints"], changed["tensor_fingerprints"]
        record["comparisons"]["different_seed"] = compare(Path(pt2["output"]), Path(changed["output"]))
        if (fingerprints["condition-latent:0"] != other["condition-latent:0"]
                or any(fingerprints[f"{phase}:0"] == other[f"{phase}:0"]
                       for phase in ("initial-noise", "denoised-latent", "reconstruction"))
                or pt2["output_sha256"] == changed["output_sha256"]):
            raise RuntimeError("different seed must change noise/result and preserve conditioning")
        for name, frames, start in (("video", 2, 0), ("video-slice", 1, 1)):
            row = run_video(binary, package, video, output / name, args.gpu_id,
                            seed=args.seed, frames=frames, start_frame=start)
            record[name] = row
            save()
            if row["status"] != "pass":
                raise RuntimeError(f"{name}: {row['status']}; inspect {row['log']}")
            row["frame_payload_sha256"] = avi_frame_hashes(Path(row["output"]))
            if len(row["frame_payload_sha256"]) != frames:
                raise RuntimeError("AVI frame chunk count differs from requested frames")
        whole, sliced = record["video"], record["video-slice"]
        frame_one = {key: digest for key, digest in whole["tensor_fingerprints"].items() if key.endswith(":1")}
        if (frame_one != sliced["tensor_fingerprints"]
                or whole["frame_payload_sha256"][1:] != sliced["frame_payload_sha256"]
                or whole["tensor_fingerprints"]["initial-noise:0"] != fingerprints["initial-noise:0"]
                or whole["tensor_fingerprints"]["initial-noise:0"] == whole["tensor_fingerprints"]["initial-noise:1"]):
            raise RuntimeError("video absolute frame/noise-stream/slice consistency failed")
        record["video_slice_consistency"] = {"status": "pass", "absolute_frame": 1,
                                            "tensor_fingerprints": frame_one,
                                            "frame_payload_sha256": sliced["frame_payload_sha256"][0]}
        if (sha256(binary) != record["binary_sha256"]
                or any(sha256(ROOT / name) != digest for name, digest in record["source_sha256"].items())):
            raise RuntimeError("measured binary/source changed during experiment")
        record["status"] = "pass"
    except Exception as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        save()
    print(f"seeded-sampling: pass; evidence: {result}", flush=True)


if __name__ == "__main__":
    main()
