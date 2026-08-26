import subprocess
import sys
from pathlib import Path

import pytest
import torch
from rotary_embedding_torch import apply_rotary_emb

import tools.reference.dit_block as dit_block
import tools.export_dit_block as exporter


def test_fixed_contract_is_explicit():
    contract = dit_block.DIT_BLOCK_CONTRACT

    assert contract.source_shape == (1, 45, 80)
    assert contract.window_shape == (4, 3, 3)
    assert contract.window_count == 9
    assert contract.text_tokens == 58
    assert contract.vid_dim == 2560
    assert contract.txt_dim == 2560
    assert contract.emb_dim == 15360
    assert contract.video_tokens == 3600
    assert contract.rope_dim == 128
    assert contract.rope_frequency_dim == 126


def test_pnnx_export_uses_absolute_model_path_and_fixed_block_inputs(tmp_path, monkeypatch):
    output_dir = tmp_path / "export"
    output_dir.mkdir()
    pnnx = tmp_path / "pnnx"
    model = output_dir / "dit_block_0.pt"
    pnnx.touch()
    model.touch()
    invocation = {}

    def fake_run(command, *, cwd, check):
        invocation["command"] = command
        invocation["cwd"] = cwd
        invocation["check"] = check

    monkeypatch.setattr(exporter.subprocess, "run", fake_run)
    exporter.run_pnnx(pnnx, model, output_dir)

    assert invocation["command"] == [
        str(pnnx.resolve()),
        str(model.resolve()),
        "inputshape=[3600,2560],[58,2560],[1,15360]",
    ]
    assert invocation["cwd"] == output_dir.resolve()
    assert invocation["check"] is True


def test_export_script_can_run_directly_from_the_repository_root():
    result = subprocess.run(
        [sys.executable, "tools/export_dit_block.py", "--help"],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_shape_parser_accepts_only_the_supported_probe_shape():
    assert exporter._shape("1,45,80") == (1, 45, 80)
    with pytest.raises(ValueError, match="supported"):
        exporter._shape("1,45,79")
    with pytest.raises(ValueError, match="positive"):
        exporter._shape("1,0,80")


def test_fixed_mmrope_matches_upstream_coordinate_formula():
    contract = dit_block.DIT_BLOCK_CONTRACT
    video_freqs, text_freqs = dit_block.build_mmrope_frequencies(
        contract.source_shape,
        contract.text_tokens,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    expected_video, expected_text = dit_block.upstream_mmrope_frequencies(
        contract.source_shape,
        contract.text_tokens,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert video_freqs.shape == (contract.video_tokens, contract.rope_frequency_dim)
    assert text_freqs.shape == (contract.text_tokens, contract.rope_frequency_dim)
    assert torch.allclose(video_freqs, expected_video, atol=0.0, rtol=0.0)
    assert torch.allclose(text_freqs, expected_text, atol=0.0, rtol=0.0)


def test_fixed_window_attention_is_deterministic_and_finite():
    pack = dit_block.FixedWindowAttention(
        source_shape=(1, 2, 3),
        window_shape=(1, 1, 2),
        text_tokens=2,
        heads=2,
        head_dim=3,
    )
    video = torch.arange(1 * 2 * 3 * 3 * 2 * 3, dtype=torch.float32).reshape(
        1 * 2 * 3, 3, 2, 3
    )
    text = torch.arange(2 * 3 * 2 * 3, dtype=torch.float32).reshape(2, 3, 2, 3)

    first_video, first_text = pack(video, text)
    second_video, second_text = pack(video, text)
    assert first_video.shape == (1, 2, 3, 2, 3)
    assert first_text.shape == (2, 2, 3)
    assert torch.isfinite(first_video).all()
    assert torch.isfinite(first_text).all()
    assert torch.equal(first_video, second_video)
    assert torch.equal(first_text, second_text)


def test_fixed_ada_keeps_the_official_dim_layer_gate_axis_order():
    branch = dit_block.FixedAdaBranch(dim=2)
    with torch.no_grad():
        branch.attn_shift.copy_(torch.tensor([10.0, 20.0]))
        branch.attn_scale.copy_(torch.tensor([1.0, 2.0]))
        branch.attn_gate.copy_(torch.tensor([3.0, 4.0]))
    hidden = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    emb = torch.arange(12, dtype=torch.float32).reshape(1, 12)

    modulation = emb.reshape(1, 2, 2, 3)[0, :, 0, :]
    shift, scale, gate = modulation.transpose(0, 1)
    expected_in = hidden * (scale + branch.attn_scale) + (shift + branch.attn_shift)
    expected_out = hidden * (gate + branch.attn_gate)
    assert torch.equal(branch(hidden, emb, "attn", "in"), expected_in)
    assert torch.equal(branch(hidden, emb, "attn", "out"), expected_out)


def test_fixed_rotary_matches_rotary_embedding_torch():
    torch.manual_seed(20260824)
    hidden = torch.randn(5, 2, 128)
    freqs = torch.randn(5, 126)
    expected = apply_rotary_emb(freqs, hidden.transpose(0, 1)).transpose(0, 1)
    actual = dit_block._apply_fixed_rotary(hidden, freqs)
    assert torch.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-6)


def test_fixed_rms_norm_keeps_scale_in_fp32_until_the_output_cast():
    hidden = torch.tensor(
        [[0.6172, -1.7578, 2.7656, -3.8125]], dtype=torch.bfloat16
    )
    norm = dit_block.FixedRMSNorm(dim=4, eps=1.0e-5, elementwise_affine=False)

    variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
    expected = (hidden * torch.rsqrt(variance + 1.0e-5)).to(dtype=hidden.dtype)

    assert torch.equal(norm(hidden), expected)


def test_rewrite_dit_param_uses_existing_awa_layers(tmp_path):
    param_path = tmp_path / "dit_block_0.ncnn.param"
    param_path.write_text(
        "7767517\n"
        "12 15\n"
        "Input vid 0 1 vid\n"
        "Input txt 0 1 txt\n"
        "MemoryData pack_indices 0 1 pack_indices 0=894\n"
        "Reshape reshape_qkv 1 1 vid qkv 0=3 1=2 11=3 2=6\n"
        "Concat cat_txt 2 1 txt qkv merged 0=0\n"
        "torch.index_select old_pack 2 1 merged pack_indices packed 0=0\n"
        "Split splitncnn_0 1 2 packed qkv0 qkv1\n"
        "MatMul attention 2 1 qkv0 qkv1 attended 0=1\n"
        "Split splitncnn_1 1 2 attended video_attended text_attended\n"
        "MemoryData video_indices 0 1 video_indices 0=6\n"
        "torch.index_select old_video 2 1 video_attended video_indices video_selected 0=0\n"
        "Reshape reshape_video 1 1 video_selected out0 0=3 1=2 11=3 2=6\n"
        "MemoryData text_indices 0 1 text_indices 0=2\n"
        "torch.index_select old_text 2 1 text_attended text_indices text_selected 0=0\n"
        "Reshape reshape_text 1 1 text_selected text_repeated 0=3 1=2 11=3 2=6\n"
        "Reduction old_mean 1 1 text_repeated out1 0=3 1=0 -23303=1,0 4=0 5=1\n"
    )

    exporter.rewrite_ncnn_param(param_path)

    lines = param_path.read_text().splitlines()
    assert int(lines[1].split()[0]) == len(lines) - 2
    assert any(line.startswith("SeedVR2AWAPack awa_pack") for line in lines)
    assert any(line.startswith("SeedVR2AWAUnpack awa_unpack") for line in lines)
    assert not any(line.startswith("torch.index_select") for line in lines)
    assert not any(line.startswith("Reduction old_mean") for line in lines)


def test_rewrite_dit_param_preserves_the_block_tail_after_awa_unpack(tmp_path):
    param_path = tmp_path / "dit_block_0.ncnn.param"
    param_path.write_text(
        "7767517\n"
        "17 20\n"
        "Input vid 0 1 vid\n"
        "Input txt 0 1 txt\n"
        "Input emb 0 1 emb\n"
        "Reshape reshape_vid_qkv 1 1 vid vid_qkv 0=128 1=20 11=3 2=3600\n"
        "Reshape reshape_txt_qkv 1 1 txt txt_qkv 0=128 1=20 11=3 2=58\n"
        "Concat concat_pack 2 1 vid_qkv txt_qkv merged 0=0\n"
        "MemoryData pack_indices 0 1 pack_indices 0=100\n"
        "torch.index_select old_pack 2 1 merged pack_indices packed 0=0\n"
        "MatMul attention 2 1 packed packed attended 0=1\n"
        "MemoryData video_indices 0 1 video_indices 0=100\n"
        "torch.index_select old_video 2 1 attended video_indices video_selected 0=0\n"
        "Reshape reshape_video 1 1 video_selected video_out 0=128 1=20 2=3600\n"
        "MemoryData text_indices 0 1 text_indices 0=100\n"
        "torch.index_select old_text 2 1 attended text_indices text_selected 0=0\n"
        "Reshape reshape_text 1 1 text_selected text_repeated 0=128 1=20 2=522\n"
        "Reduction reduce_text 1 1 text_repeated txt_out 0=3 1=0 -23303=1,0 4=0 5=1\n"
        "InnerProduct project_vid 1 1 video_out vid_projected 0=2560\n"
        "InnerProduct project_txt 1 1 txt_out txt_projected 0=2560\n"
        "Reshape reshape_text_batch 1 1 txt_projected text_batch 0=2560 1=58 12=233 13=1\n"
    )

    exporter.rewrite_ncnn_param(param_path)

    lines = param_path.read_text().splitlines()
    assert any(
        line.startswith("SeedVR2AWAPack awa_pack")
        and " vid_qkv txt_qkv packed awa_cu_seqlens " in line
        for line in lines
    )
    assert any(
        line.startswith("SeedVR2AWAUnpack awa_unpack")
        and " attended video_out txt_out " in line
        for line in lines
    )
    assert any(line.startswith("InnerProduct project_vid") for line in lines)
    assert any(line.startswith("InnerProduct project_txt") for line in lines)
    assert any(
        line.startswith("Reshape reshape_text_batch") and "12=233 13=0" in line
        for line in lines
    )
    assert any(
        line.startswith("Reshape reshape_awa_video_batch") and "12=233 13=0" in line
        for line in lines
    )
    assert any("project_vid" in line and "awa_video_batch" in line for line in lines)
    assert not any(line.startswith("torch.index_select old_") for line in lines)
    assert not any(line.startswith("Reshape reshape_video") for line in lines)
    assert not any(line.startswith("Reduction reduce_text") for line in lines)


def test_rewrite_existing_stack_param_updates_video_batch_token_count(tmp_path):
    param_path = tmp_path / "dit_block_0.ncnn.param"
    param_path.write_text(
        "7767517\n"
        "6 7\n"
        "Input vid 0 1 vid\n"
        "SeedVR2AWAPack awa_pack 2 2 vid vid packed cu 0=1 1=8 2=8 3=4 4=3 5=3 6=58 7=0\n"
        "SeedVR2AWAUnpack awa_unpack 1 2 packed video_out text_out 0=1 1=8 2=8 3=4 4=3 5=3 6=58 7=0\n"
        "Reshape reshape_text_batch 1 1 text_out text_batch 0=2560 1=58 12=233 13=1\n"
        "Reshape reshape_awa_video_batch 1 1 video_out awa_video_batch 0=2560 1=3600 12=233 13=0\n"
        "InnerProduct project_vid 1 1 awa_video_batch vid_projected 0=2560\n"
    )

    exporter.rewrite_ncnn_param(param_path)

    lines = param_path.read_text().splitlines()
    assert any(
        line.startswith("Reshape reshape_awa_video_batch") and "1=64" in line
        for line in lines
    )
    assert any(line.startswith("Reshape reshape_text_batch") and "13=0" in line for line in lines)


def test_rewrite_dit_param_replaces_pnnx_mmrope_without_consuming_its_weights(tmp_path):
    param_path = tmp_path / "dit_block_0.ncnn.param"
    param_path.write_text(
        "7767517\n"
        "22 28\n"
        "Input vid 0 1 vid\n"
        "Input txt 0 1 txt\n"
        "Concat cat_pack 2 1 vid txt merged 0=0\n"
        "MemoryData pack_indices 0 1 pack_indices 0=16\n"
        "torch.index_select old_pack 2 1 merged pack_indices packed 0=0\n"
        "Slice rope_unbind 1 3 packed rope_q rope_k rope_v -23300=3,-233,-233,-233 1=0\n"
        "MemoryData rope_cos 0 1 rope_cos 0=6 1=1 2=16\n"
        "Split rope_cos_split 1 2 rope_cos rope_cos0 rope_cos1\n"
        "BinaryOp rope_mul 2 1 rope_q rope_cos0 rope_q_rotated 0=2\n"
        "Concat rope_stack 3 1 rope_q_rotated rope_k rope_v rope_qkv 0=0\n"
        "Reshape rope_end 1 1 rope_qkv qkv_rotated 0=8 1=1 2=3\n"
        "Split attention_split 1 4 qkv_rotated att_q att_k att_v extra\n"
        "MatMul attention 2 1 att_q att_k attended 0=1\n"
        "MemoryData video_indices 0 1 video_indices 0=16\n"
        "torch.index_select old_video 2 1 attended video_indices video_selected 0=0\n"
        "Reshape reshape_video 1 1 video_selected out0 0=8 1=1 2=16\n"
        "MemoryData text_indices 0 1 text_indices 0=2\n"
        "torch.index_select old_text 2 1 attended text_indices text_selected 0=0\n"
        "Reshape reshape_text 1 1 text_selected text_repeated 0=8 1=1 2=2\n"
        "Reduction reduce_text 1 1 text_repeated out1 0=3 1=0 -23303=1,0 4=0 5=1\n"
    )

    exporter.rewrite_ncnn_param(param_path)

    lines = param_path.read_text().splitlines()
    assert any(
        line.startswith("SeedVR2MMRoPE mmrope") and " packed qkv_rotated " in line
        for line in lines
    )
    assert any(line.startswith("MemoryData rope_cos") for line in lines)
    assert not any(line.startswith("Slice rope_unbind") for line in lines)
    assert not any(line.startswith("Concat rope_stack") for line in lines)
    assert not any(line.startswith("Reshape rope_end") for line in lines)


def test_rewrite_dit_param_replaces_static_window_attention_after_mmrope(tmp_path):
    param_path = tmp_path / "dit_block_0.ncnn.param"
    branches = " ".join(f"head_{index}" for index in range(27))
    attention = "".join(
        f"MatMul attention_{index} 2 1 head_{index * 3} head_{index * 3 + 1} attended_{index} 0=1\n"
        for index in range(9)
    )
    attended = " ".join(f"attended_{index}" for index in range(9))
    param_path.write_text(
        "7767517\n"
        "42 52\n"
        "Input vid 0 1 vid\n"
        "Input txt 0 1 txt\n"
        "Concat cat_pack 2 1 vid txt merged 0=0\n"
        "MemoryData pack_indices 0 1 pack_indices 0=16\n"
        "torch.index_select old_pack 2 1 merged pack_indices packed 0=0\n"
        "Slice rope_unbind 1 3 packed rope_q rope_k rope_v -23300=3,-233,-233,-233 1=0\n"
        "MemoryData rope_cos 0 1 rope_cos 0=6 1=1 2=16\n"
        "Split rope_cos_split 1 2 rope_cos rope_cos0 rope_cos1\n"
        "BinaryOp rope_mul 2 1 rope_q rope_cos0 rope_q_rotated 0=2\n"
        "Concat rope_stack 3 1 rope_q_rotated rope_k rope_v rope_qkv 0=0\n"
        "Reshape rope_end 1 1 rope_qkv qkv_rotated 0=8 1=1 2=3\n"
        f"Split attention_split 1 27 qkv_rotated {branches}\n"
        + attention
        + f"Concat cat_windows 9 1 {attended} attended 0=0\n"
        "Split attended_split 1 2 attended video_attended text_attended\n"
        "MemoryData video_indices 0 1 video_indices 0=16\n"
        "torch.index_select old_video 2 1 video_attended video_indices video_selected 0=0\n"
        "Reshape reshape_video 1 1 video_selected out0 0=8 1=1 2=16\n"
        "MemoryData text_indices 0 1 text_indices 0=2\n"
        "torch.index_select old_text 2 1 text_attended text_indices text_selected 0=0\n"
        "Reshape reshape_text 1 1 text_selected text_repeated 0=8 1=1 2=2\n"
        "Reduction reduce_text 1 1 text_repeated out1 0=3 1=0 -23303=1,0 4=0 5=1\n"
    )

    exporter.rewrite_ncnn_param(param_path)

    lines = param_path.read_text().splitlines()
    assert any(
        line.startswith("SeedVR2WindowAttention awa_attention")
        and " qkv_rotated attended " in line
        for line in lines
    )
    assert any(line.startswith("MemoryData rope_cos") for line in lines)
    assert not any(line.startswith("Split attention_split") for line in lines)
    assert not any(line.startswith("MatMul attention_") for line in lines)
    assert not any(line.startswith("Concat cat_windows") for line in lines)
