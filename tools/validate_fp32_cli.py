#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 SeedVR2-ncnn contributors
"""Validate the fixed RGB256 FP32 CLI against the independent image audit.

Development-only tool. Compiles a tiny helper against the runtime's Gaussian
generator, supplies its exact bytes to the original CPU reference, then checks
all four CLI tensor handoffs against the independently executed native audit.
The runtime executable itself needs no Python, PyTorch or CUDA.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_pt2_cfg_experiment import run_case, sha256


def fnv1a64(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & 0xffffffffffffffff
    return f"{value:016x}"


def audited_handoffs(audit, noise):
    """Allow complete numerical failures to remain usable for CLI equality only."""
    reference = json.loads((audit / "result.json").read_text())
    rows = reference.get("stages", []) + reference.get("image_stages", [])
    stages = {f"step_{step}_{name}" for step in range(2)
              for name in ("positive", "negative", "latent")} | {"vae_encode", "vae_decode"}
    if (reference.get("status") not in {"pass", "failed"} or len(rows) != 8
            or {row.get("stage") for row in rows} != stages
            or any(not isinstance(row.get("pass"), bool) for row in rows)
            or not all(reference.get(name) is True for name in (
                "sources_unchanged_at_end", "runner_unchanged_at_end", "fixtures_unchanged_at_end"))):
        raise ValueError("incomplete or changed independent image audit")
    if reference.get("initial_noise", {}).get("sha256") != sha256(noise):
        raise ValueError("independent image audit used different initial noise")
    numerical_pass = all(row["pass"] for row in rows)
    if numerical_pass != (reference["status"] == "pass"):
        raise ValueError("independent image audit status disagrees with numerical checks")
    expected = {"initial-noise:0": fnv1a64(noise.read_bytes())}
    for phase, name, count in (("condition-latent", "condition", 16384),
                               ("denoised-latent", "latent", 16384),
                               ("reconstruction", "reconstruction", 196608)):
        path = audit / f"native.{name}.f32"
        if (path.stat().st_size != count * 4
                or sha256(path) != reference.get("native_outputs", {}).get(name)):
            raise ValueError(f"independent image audit output changed: {name}")
        expected[f"{phase}:0"] = fnv1a64(path.read_bytes())
    return reference, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-root", "checkpoint", "vae-checkpoint", "package", "image", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/seedvr2-ncnn")
    parser.add_argument("--runner", type=Path, default=ROOT / "build/seedvr2-reference-sampling-smoke")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--baseline-result", type=Path,
                        help="optional previous BF16 run_case record, e.g. architecture-v2 acceptance result")
    args = parser.parse_args()
    if not 0 <= args.seed <= 0xffffffff or args.gpu_id < 0:
        parser.error("expected uint32 seed and a nonnegative GPU id")
    for name in ("reference_root", "checkpoint", "vae_checkpoint", "package", "image", "binary", "runner"):
        setattr(args, name, getattr(args, name).resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = ["CMakeLists.txt", "tools/validate_fp32_cli.py", "tools/compare_reference_image.py",
               "tools/compare_reference_sampling.py", "tools/run_pt2_cfg_experiment.py",
               "tools/with-nvidia-vulkan-runtime.sh", "src/main.cpp"]
    for directory in ("src/cli", "src/inference", "src/dit", "src/awa", "src/vae", "src/sampler", "src/model", "src/resolution"):
        sources.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT / directory).rglob("*"))
                       if p.suffix in {".cpp", ".h", ".comp"})
    result = dict(status="running", date_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  command=[sys.executable, *sys.argv], system=platform.platform(),
                  repository_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  repository_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
                  binary_sha256=sha256(args.binary), runner_sha256=sha256(args.runner),
                  sources={name: sha256(ROOT / name) for name in sources}, input_sha256=sha256(args.image),
                  noise_algorithm="runtime mt19937-box-muller-v1, stream/frame=0; exact bytes supplied to reference",
                  scope="one RGB256 image; FP32 two-step CFG1.5 numerical audit plus positive-only smoke",
                  image_quality_validated=False)
    record = output / "result.json"
    try:
        # Compile only repository sampler code; no alternate RNG implementation.
        helper = output / "noise.cpp"
        helper.write_text('''// SPDX-License-Identifier: BSD-3-Clause
#include "sampler/sampler.h"
#include <fstream>
#include <vector>
int main(int argc, char** argv) {
    if (argc != 3) return 2;
    std::vector<float> values(16384);
    if (!seedvr2::fill_gaussian_noise(values.data(), values.size(),
            static_cast<std::uint32_t>(std::stoull(argv[2])), 0)) return 1;
    std::ofstream output(argv[1], std::ios::binary);
    output.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    return output ? 0 : 1;
}
''')
        compiler = ["c++", "-std=c++17", "-O2", "-I", str(ROOT / "src"), str(helper),
                    str(ROOT / "src/sampler/sampler.cpp"), "-o", str(output / "write-noise")]
        subprocess.run(compiler, cwd=ROOT, check=True)
        noise = output / "noise.f32"
        subprocess.run([str(output / "write-noise"), str(noise), str(args.seed)], check=True)
        result.update(noise_sha256=sha256(noise), noise_helper_sha256=sha256(helper),
                      noise_binary_sha256=sha256(output / "write-noise"), noise_compile_command=compiler)
        audit = output / "reference-image"
        command = [sys.executable, str(ROOT / "tools/compare_reference_image.py")]
        for name in ("reference-root", "checkpoint", "vae-checkpoint", "package", "image", "runner"):
            command += [f"--{name}", str(getattr(args, name.replace("-", "_")))]
        command += ["--output-dir", str(audit), "--steps", "2", "--cfg-scale", "1.5",
                    "--seed", str(args.seed), "--gpu-id", str(args.gpu_id), "--initial-noise", str(noise)]
        result["reference_command"] = command
        record.write_text(json.dumps(result, indent=2) + "\n")
        with (output / "reference.log").open("w") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        result["reference_exit_code"] = completed.returncode
        reference, expected = audited_handoffs(audit, noise)
        if (completed.returncode == 0) != (reference["status"] == "pass"):
            raise ValueError("independent audit exit code disagrees with its result")
        result["numerical_status"] = reference["status"]
        result["reference_result_sha256"] = sha256(audit / "result.json")
        row = run_case(args.binary, args.package, args.image, output, args.gpu_id,
                       "fp32-cfg1.5-s2", 2, 1.5, args.seed, "fp32")
        result.update(fp32=row, expected_handoffs=expected,
                      handoffs_equal=row["tensor_fingerprints"] == expected)
        if row["status"] != "pass" or not result["handoffs_equal"]:
            raise ValueError("ordinary FP32 CLI handoffs differ from the independent native audit")
        positive = run_case(args.binary, args.package, args.image, output, args.gpu_id,
                            "fp32-cfg1-s1", 1, 1, args.seed, "fp32")
        result["positive_only"] = positive
        if positive["status"] != "pass":
            raise ValueError("positive-only FP32 CLI contract failed")
        if args.baseline_result:
            baseline = json.loads(args.baseline_result.read_text())
            candidates = baseline.get("runs", [baseline.get("run", {})])
            old = next(r for r in candidates if r.get("steps") == 2 and r.get("cfg_scale") == 1.5
                       and r.get("seed") == args.seed and r.get("status") == "pass")
            fallback = run_case(args.binary, args.package, args.image, output, args.gpu_id,
                                "default-bf16-cfg1.5-s2", 2, 1.5, args.seed)
            result["bf16_regression"] = fallback
            result["bf16_unchanged"] = (fallback["status"] == "pass"
                and fallback["tensor_fingerprints"] == old["tensor_fingerprints"]
                and fallback.get("output_sha256") == old["output_sha256"])
            if not result["bf16_unchanged"]:
                raise ValueError("default BF16 handoffs or PNG changed")
        result["integration_status"] = "pass"
        if result["numerical_status"] != "pass":
            raise ValueError("CLI integration passed; independent original/native numerical audit failed")
        result["status"] = "pass"
    except Exception as error:
        result.update(status="failed", error=str(error))
        raise
    finally:
        result["sources_binaries_noise_unchanged_at_end"] = (
            all(sha256(ROOT / name) == digest for name, digest in result["sources"].items())
            and sha256(args.binary) == result["binary_sha256"] and sha256(args.runner) == result["runner_sha256"]
            and sha256(args.image) == result["input_sha256"]
            and "noise_sha256" in result and sha256(noise) == result["noise_sha256"])
        if not result["sources_binaries_noise_unchanged_at_end"]:
            result.update(status="failed", integrity_error="measured sources, binaries, input or noise changed")
        record.write_text(json.dumps(result, indent=2) + "\n")
        if not result["sources_binaries_noise_unchanged_at_end"]:
            raise ValueError("measured sources, binaries or noise changed")
    print(record)


if __name__ == "__main__":
    main()
