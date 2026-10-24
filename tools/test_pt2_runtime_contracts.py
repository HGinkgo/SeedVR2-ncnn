#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 SeedVR2-ncnn contributors
"""Check runtime evidence rejection without executing a model or using a GPU."""

import json
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from tools.run_pt2_cfg_experiment import ROOT, run_case, sha256
from tools.validate_seeded_sampling import avi_frame_hashes


class RuntimeEvidenceTests(unittest.TestCase):
    def test_fp32_requires_release_between_cfg_branches(self):
        handoffs = "".join(f"profile name=tensor-fingerprint phase={phase} frame=0 "
                            "fnv1a64=0123456789abcdef finite=1\n"
                            for phase in ("condition-latent", "denoised-latent", "reconstruction"))
        load = "stage=load-dit-branch step=0 branch=0 precision=fp32\n"
        release = "stage=release-dit-branch step=0 branch=0\n"
        negative = "stage=load-dit-branch step=0 branch=1 precision=fp32\n"
        release_negative = "stage=release-dit-branch step=0 branch=1\n"
        end = "stage=load-decode\nseedvr2-image-inference: ok\n" + handoffs
        cases = (
            ("sequential", 1.5, load + release + negative + release_negative + end, "pass"),
            ("resident-both", 1.5, load + negative + release + release_negative + end, "failed-runtime-contract"),
            ("not-released", 1.5, load + release + negative + end, "failed-runtime-contract"),
            ("positive-only", 1, load + release + end, "pass"),
            ("unexpected-negative", 1, load + release + negative + release_negative + end, "failed-runtime-contract"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "manifest.sha256").write_text("fixture manifest\n")
            for name, scale, log, expected in cases:
                with self.subTest(name=name):
                    def launch(command, *, cwd, stdout, stderr):
                        self.assertEqual(command[-2:], ["--precision", "fp32"])
                        stdout.write(log)
                        Path(command[command.index("--output") + 1]).write_bytes(b"fixture output")
                        process = Mock(returncode=0)
                        process.wait.return_value = 0
                        return process
                    with patch("tools.run_pt2_cfg_experiment.subprocess.Popen", side_effect=launch), \
                            patch("tools.run_pt2_cfg_experiment.gpu_memory", return_value=None):
                        row = run_case(output / "unused", output, output / "unused-input", output,
                                       0, name, 1, scale, precision="fp32")
                    self.assertEqual(row["status"], expected)

    def test_success_marker_requires_handoffs_and_cfg_lifecycle(self):
        phases = ("condition-latent", "denoised-latent", "reconstruction")
        handoffs = "".join(
            f"profile name=tensor-fingerprint phase={phase} frame=0 "
            "fnv1a64=0123456789abcdef finite=1\n" for phase in phases)
        lifecycle = ("stage=load-dit-stack-negative\nstage=dit-stack step=1\n"
                     "stage=release-dit-stacks-for-decode\nstage=load-decode\n"
                     "profile name=total ms=1.0 peak-rss-mib=1\n"
                     "seedvr2-image-inference: ok\n")
        noise = ("stage=noise-create distribution=normal seed=666 frame=0 "
                 "algorithm=mt19937-box-muller-v1\n")
        initial = ("profile name=tensor-fingerprint phase=initial-noise frame=0 "
                   "fnv1a64=123456789abcdef0 finite=1\n")
        cases = [
            ("valid", lifecycle + handoffs, "pass"),
            ("missing", lifecycle, "failed-tensor-contract"),
            ("nonfinite", lifecycle + handoffs.replace("finite=1", "finite=0", 1),
             "failed-tensor-contract"),
            ("duplicate", lifecycle + handoffs + handoffs.splitlines()[0] + "\n",
             "failed-tensor-contract"),
            ("wrong-frame", lifecycle + handoffs.replace("frame=0", "frame=1", 1),
             "failed-tensor-contract"),
            ("missing-release", lifecycle.replace("stage=release-dit-stacks-for-decode\n", "")
             + handoffs, "failed-runtime-contract"),
            ("missing-step", lifecycle.replace("stage=dit-stack step=1\n", "") + handoffs,
             "failed-runtime-contract"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "manifest.sha256").write_text("fixture manifest\n")
            for name, log, expected in cases:
                with self.subTest(name=name):
                    process = Mock(returncode=0)
                    process.wait.return_value = 0

                    def launch(command, *, cwd, stdout, stderr):
                        stdout.write(log)
                        Path(command[command.index("--output") + 1]).write_bytes(b"fixture output")
                        return process

                    with patch("tools.run_pt2_cfg_experiment.subprocess.Popen", side_effect=launch), \
                            patch("tools.run_pt2_cfg_experiment.gpu_memory", return_value=None):
                        row = run_case(output / "unused-binary", output, output / "unused-input",
                                       output, 0, name, 2, 1.5)
                    self.assertEqual(row["status"], expected)
            for name, text, expected in (
                    ("seed-valid", lifecycle + handoffs + initial + noise, "pass"),
                    ("seed-missing-noise", lifecycle + handoffs + noise, "failed-tensor-contract"),
                    ("seed-wrong-stream", lifecycle + handoffs + initial + noise.replace("frame=0", "frame=1"),
                     "failed-noise-contract")):
                with self.subTest(name=name):
                    def launch_seeded(command, *, cwd, stdout, stderr):
                        self.assertEqual(command[-2:], ["--seed", "666"])
                        stdout.write(text)
                        Path(command[command.index("--output") + 1]).write_bytes(b"fixture output")
                        process = Mock(returncode=0)
                        process.wait.return_value = 0
                        return process

                    with patch("tools.run_pt2_cfg_experiment.subprocess.Popen", side_effect=launch_seeded), \
                            patch("tools.run_pt2_cfg_experiment.gpu_memory", return_value=None):
                        row = run_case(output / "unused-binary", output, output / "unused-input",
                                       output, 0, name, 2, 1.5, seed=666)
                    self.assertEqual(row["status"], expected)

    def test_avi_payload_reader_respects_bounds_and_chunk_structure(self):
        def chunk(kind, body):
            return kind + struct.pack("<I", len(body)) + body + (b"\0" if len(body) & 1 else b"")

        frame = b"\x01\x02\x03" * (256 * 256)
        body = b"AVI " + chunk(b"JUNK", b"00db") + chunk(
            b"LIST", b"movi" + chunk(b"JUNK", b"x") + chunk(b"00db", frame))
        data = chunk(b"RIFF", body)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.avi"
            path.write_bytes(data)
            self.assertEqual(len(avi_frame_hashes(path)), 1)
            path.write_bytes(data[:-1])
            with self.assertRaisesRegex(ValueError, "length differs"):
                avi_frame_hashes(path)
            broken = data.replace(struct.pack("<I", len(frame)), struct.pack("<I", len(data) * 2))
            path.write_bytes(broken)
            with self.assertRaisesRegex(ValueError, "exceeds container"):
                avi_frame_hashes(path)

    def test_legacy_evidence_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            binary, image = directory / "unused-binary", directory / "input"
            binary.write_bytes(b"not an executable")
            image.write_bytes(b"fixture input")
            names = ("torchscript-s1", "torchscript-s2", "pt2-cfg1-s1", "pt2-cfg1-s2",
                     "pt2-cfg1.5-s1", "pt2-cfg1.5-s2")
            evidence = directory / "legacy.json"
            evidence.write_text(json.dumps({
                "status": "pass", "binary_sha256": sha256(binary), "input_sha256": sha256(image),
                "gpu_id": 0, "runs": [{"name": name, "status": "pass"} for name in names]}))
            destination = directory / "must-not-exist"
            result = subprocess.run([
                sys.executable, str(ROOT / "tools/validate_pt2_cfg_runtime.py"),
                "--image-results", str(evidence), "--binary", str(binary),
                "--input", str(image), "--package", str(directory),
                "--control-package", str(directory), "--video-input", str(image),
                "--output-dir", str(destination)], text=True, capture_output=True, cwd=ROOT)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("image evidence lacks valid tensor handoffs", result.stderr)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
