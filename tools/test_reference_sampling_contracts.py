# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 SeedVR2-ncnn contributors
"""Checkpoint-free layout and stale-package guards for the reference sampler."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from tools.compare_reference_sampling import load_condition, load_initial_noise, model_video, patch_output, verify_package
from tools.validate_fp32_cli import audited_handoffs, sha256


class ReferenceSamplingContractTests(unittest.TestCase):
    def test_failed_numerical_audit_retains_handoffs_but_rejects_incomplete_or_changed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = Path(directory)
            noise = audit / "noise.f32"
            noise.write_bytes(bytes(16384 * 4))
            reference = dict(status="failed", sources_unchanged_at_end=True,
                runner_unchanged_at_end=True, fixtures_unchanged_at_end=True,
                initial_noise={"sha256": sha256(noise)}, native_outputs={},
                stages=[dict(stage=f"step_{s}_{n}", **{"pass": s == 0})
                        for s in range(2) for n in ("positive", "negative", "latent")],
                image_stages=[dict(stage=n, **{"pass": True}) for n in ("vae_encode", "vae_decode")])
            for name, count in (("condition", 16384), ("latent", 16384), ("reconstruction", 196608)):
                path = audit / f"native.{name}.f32"
                path.write_bytes(bytes(count * 4))
                reference["native_outputs"][name] = sha256(path)
            record = audit / "result.json"
            record.write_text(json.dumps(reference))
            actual, handoffs = audited_handoffs(audit, noise)
            self.assertEqual(actual["status"], "failed")
            self.assertEqual(len(handoffs), 4)
            reference["stages"][0]["stage"] = "step_1_latent"
            record.write_text(json.dumps(reference))
            with self.assertRaisesRegex(ValueError, "incomplete or changed"):
                audited_handoffs(audit, noise)
            reference["stages"][0]["stage"] = "step_0_positive"
            reference["status"] = "pass"
            record.write_text(json.dumps(reference))
            with self.assertRaisesRegex(ValueError, "status disagrees"):
                audited_handoffs(audit, noise)
            reference["status"] = "failed"
            record.write_text(json.dumps(reference))
            (audit / "native.latent.f32").write_bytes(bytes(4))
            with self.assertRaisesRegex(ValueError, "output changed: latent"):
                audited_handoffs(audit, noise)

    def test_exact_initial_noise_bytes_and_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "noise.f32"
            values = np.arange(16384, dtype="<f4")
            values.tofile(path)
            actual, metadata = load_initial_noise(path, torch.Generator())
            self.assertEqual(actual.shape, (16, 1, 32, 32))
            np.testing.assert_array_equal(actual.numpy().ravel(), values)
            self.assertEqual(len(metadata["sha256"]), 64)
            for invalid in (values[:-1], np.append(values, np.float32(0)), values * np.float32("nan")):
                invalid.astype("<f4").tofile(path)
                with self.assertRaisesRegex(ValueError, "16384 finite"):
                    load_initial_noise(path, torch.Generator())

    def test_saved_vae_condition_layout_dtype_and_finiteness(self):
        value = torch.arange(16 * 32 * 32, dtype=torch.float32).reshape(16, 1, 32, 32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "condition.pt"
            torch.save(value, path)
            actual, metadata = load_condition(path, torch.Generator())
            torch.testing.assert_close(actual, value, atol=0, rtol=0)
            self.assertEqual(len(metadata["sha256"]), 64)
            invalid_values = [value[:, 0], value.bfloat16(), {"latent": value}, value.clone()]
            invalid_values[-1][0, 0, 0, 0] = float("nan")
            for invalid in invalid_values:
                torch.save(invalid, path)
                with self.assertRaisesRegex(ValueError, "finite float32"):
                    load_condition(path, torch.Generator())

    def test_pixel_channel_and_patch_order(self):
        sample = torch.arange(16 * 32 * 32, dtype=torch.float32).reshape(16, 1, 32, 32)
        condition = sample + 20000
        video, patches = model_video(sample, condition), patch_output(sample)
        for y, x in ((0, 0), (0, 31), (13, 17), (31, 31)):
            torch.testing.assert_close(video[y * 32 + x, :16], sample[:, 0, y, x])
            torch.testing.assert_close(video[y * 32 + x, 16:32], condition[:, 0, y, x])
            self.assertEqual(video[y * 32 + x, 32], 1)
            patch = (y // 2) * 16 + x // 2
            offset = ((y % 2) * 2 + x % 2) * 16
            torch.testing.assert_close(patches[patch, offset:offset+16], sample[:, 0, y, x])

    def test_invalid_layouts_and_legacy_package_rejected(self):
        good = torch.zeros(16, 1, 32, 32)
        for invalid in (torch.zeros(16, 32, 32), torch.zeros(16, 1, 16, 32)):
            with self.assertRaisesRegex(ValueError, "fixed"):
                patch_output(invalid)
            with self.assertRaisesRegex(ValueError, "matching fixed"):
                model_video(good, invalid)
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "package"
            package.mkdir()
            sidecar = package.with_name(package.name + ".provenance.json")
            sidecar.write_text(json.dumps({"kind": "legacy-ts-package"}))
            with self.assertRaisesRegex(ValueError, "genuine PT2"):
                verify_package(package, "unused")
            sidecar.write_text(json.dumps({"kind": "fixed-256-pt2-dit-package",
                "stack_manifests": {str(n): {"text_tokens": n} for n in (58, 64)}}))
            with self.assertRaisesRegex(ValueError, "corrected fixed-256"):
                verify_package(package, "unused")


if __name__ == "__main__":
    unittest.main()
