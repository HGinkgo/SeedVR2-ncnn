# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 SeedVR2-ncnn contributors
"""Small, checkpoint-free regression for the final Ada feature packing."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch
from tools.reference.dit_stack import DIT_STACK_CONTRACT, FixedDitOutput
from tools.reference.dit_block import FixedDitBlock


class DitOutputContractTests(unittest.TestCase):
    def fixture(self):
        contract = replace(DIT_STACK_CONTRACT, hidden_dim=8, embedding_dim=48, latent_channels=2)
        generator = torch.Generator().manual_seed(11)
        state = {"vid_out_norm.weight": torch.ones(8),
                 "vid_out_ada.out_shift": torch.randn(8, generator=generator),
                 "vid_out_ada.out_scale": torch.randn(8, generator=generator),
                 "vid_out.proj.weight": torch.eye(8), "vid_out.proj.bias": torch.zeros(8)}
        return FixedDitOutput(state, contract=contract).eval(), torch.randn(3, 8, generator=generator)

    def test_feature_packing_and_both_export_formats(self):
        model, hidden = self.fixture()
        # Each feature and group has a distinct tag; random or zero embeddings
        # can conceal a mistaken contiguous-half slice.
        embedding = torch.arange(48, dtype=torch.float32).reshape(1, 48) / 10
        normalized = torch.nn.functional.rms_norm(hidden, (8,), model.norm, eps=1e-5)
        indices = torch.arange(8) * 6
        expected = normalized * (embedding[0, indices + 1] + model.scale)
        expected = expected + embedding[0, indices] + model.shift
        legacy = embedding[:, :24].reshape(8, 3)
        incorrect = normalized * (legacy[:, 1] + model.scale) + legacy[:, 0] + model.shift
        self.assertGreater(float((incorrect - expected).abs().max()), 1)
        with torch.inference_mode():
            torch.testing.assert_close(model(hidden, embedding), expected)
            traced = torch.jit.trace(model, (hidden, embedding))
            exported = torch.export.export(model, (hidden, embedding))
            torch.testing.assert_close(traced(hidden, embedding), expected)
            torch.testing.assert_close(exported.module()(hidden, embedding), expected)

    def test_invalid_signatures_fail(self):
        model, hidden = self.fixture()
        for embedding in (torch.zeros(48), torch.zeros(2, 48), torch.zeros(1, 24)):
            with self.subTest(shape=tuple(embedding.shape)), self.assertRaisesRegex(ValueError, "one packed"):
                model(hidden, embedding)
        with self.assertRaisesRegex(ValueError, "rank-2 video"):
            model(hidden.unsqueeze(0), torch.zeros(1, 48))

    def test_legacy_export_cannot_be_resumed_or_packaged(self):
        from tools.export_dit_stack import export_stack
        from tools.prepare_pt2_cfg_package import inspect_stack
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "dit_output.pt2").write_bytes(b"legacy fixture")
            (root / "dit_output.manifest.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "obsolete output Ada"):
                export_stack(root / "not-needed.pth", root, None, 11, format="pt2", resume=True)
            (root / "dit_stack_manifest.json").write_text(json.dumps({
                "frontend": "pt2", "source_shape": [1, 16, 16], "text_tokens": 58, "block_count": 32}))
            with self.assertRaisesRegex(ValueError, "output Ada contract is missing or obsolete"):
                inspect_stack(root, 58)

    def test_last_block_skips_text_ada_and_mlp_but_keeps_attention(self):
        class Forbidden(torch.nn.Module):
            def forward(self, *args):
                raise AssertionError("last block must bypass text Ada and MLP")

        torch.manual_seed(19)
        block = FixedDitBlock(source_shape=(1, 1, 2), text_tokens=2,
                              vid_dim=8, txt_dim=8, emb_dim=48,
                              heads=1, head_dim=128, is_last_layer=True).eval()
        self.assertFalse(hasattr(block.ada, "txt"))
        self.assertFalse(hasattr(block.mlp, "txt"))
        block.ada.txt = Forbidden()
        block.mlp.txt = Forbidden()
        block.mlp_norm.txt = Forbidden()
        video, text, embedding = torch.randn(2, 8), torch.randn(2, 8), torch.randn(1, 48)
        with torch.inference_mode():
            normalized_text = torch.nn.functional.rms_norm(text, (8,), eps=1e-5)
            video_in = block.ada.vid(block.attn_norm.vid(video), embedding, "attn", "in")
            attended_video, attended_text = block.attn(video_in, normalized_text)
            video_residual = video + block.ada.vid(attended_video, embedding, "attn", "out")
            mlp_in = block.ada.vid(block.mlp_norm.vid(video_residual), embedding, "mlp", "in")
            expected_video = video_residual + block.ada.vid(block.mlp.vid(mlp_in), embedding, "mlp", "out")
            expected_text = 2 * (attended_text + text)
            expected = (expected_video, expected_text)
            torch.testing.assert_close(block(video, text, embedding), expected)
            torch.testing.assert_close(torch.jit.trace(block, (video, text, embedding))(
                video, text, embedding), expected)
            torch.testing.assert_close(torch.export.export(block, (video, text, embedding)).module()(
                video, text, embedding), expected)

    def test_legacy_last_block_cannot_be_resumed_or_packaged(self):
        from tools.export_dit_stack import export_stack
        from tools.prepare_pt2_cfg_package import inspect_stack, OUTPUT_MODULATION
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "dit_block_31.pt2").write_bytes(b"legacy last block")
            (root / "dit_block_31.manifest.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "obsolete last block"):
                export_stack(root / "not-needed.pth", root, None, 11, format="pt2", resume=True)
            (root / "dit_output.manifest.json").write_text(json.dumps({"output_modulation": OUTPUT_MODULATION}))
            (root / "dit_stack_manifest.json").write_text(json.dumps({
                "frontend": "pt2", "source_shape": [1, 16, 16], "text_tokens": 58,
                "block_count": 32, "output_modulation": OUTPUT_MODULATION}))
            with self.assertRaisesRegex(ValueError, "last block contract is missing or obsolete"):
                inspect_stack(root, 58)

    def test_reference_precision_adapter_preserves_selected_precision(self):
        from tools.compare_reference_dit import _AttentionPrecision, _sdpa_windows
        torch.manual_seed(23)
        q, k, v = (torch.randn(3, 2, 128) for _ in range(3))
        offsets = torch.tensor([0, 3], dtype=torch.int32)
        for bypass, dtype in ((True, torch.float32), (False, torch.bfloat16)):
            mode = _AttentionPrecision(bypass)
            with torch.inference_mode(), mode:
                actual = _sdpa_windows(q.bfloat16(), k.bfloat16(), v.bfloat16(), offsets, offsets)
            self.assertEqual(mode.casts, 3)
            self.assertEqual(actual.dtype, dtype)
            tensors = [x if bypass else x.bfloat16().float() for x in (q, k, v)]
            expected = torch.nn.functional.scaled_dot_product_attention(
                *(x.transpose(0, 1)[None] for x in tensors))[0].transpose(0, 1).to(dtype)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
