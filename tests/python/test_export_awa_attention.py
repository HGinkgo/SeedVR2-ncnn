import sys
from pathlib import Path

import pytest

import tools.export_awa_attention as exporter
import tools.export_vae as vae_exporter


def test_pnnx_export_uses_an_absolute_model_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "converted"
    pnnx = tmp_path / "pnnx"
    pnnx.touch()
    invocation = {}

    def fake_run(command, *, cwd, check):
        invocation["command"] = command
        invocation["cwd"] = cwd
        invocation["check"] = check

    monkeypatch.setattr(exporter.subprocess, "run", fake_run)
    monkeypatch.setattr(exporter, "rewrite_ncnn_param", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_awa_attention.py",
            "--output-dir",
            "converted",
            "--pnnx",
            str(pnnx),
        ],
    )

    exporter.main()

    assert Path(invocation["command"][1]) == (output_dir / "awa_attention.pt").resolve()
    assert Path(invocation["cwd"]) == output_dir.resolve()
    assert invocation["check"] is True


def test_rewrite_ncnn_param_uses_awa_custom_layers(tmp_path):
    param_path = tmp_path / "awa_attention.ncnn.param"
    param_path.write_text(
        "7767517\n"
        "12 16\n"
        "Input in0 0 1 in0\n"
        "Input in1 0 1 in1\n"
        "MemoryData pack 0 1 2 0=894\n"
        "Reshape reshape_8 1 1 in0 3 0=3 1=2 11=3 2=874\n"
        "Concat cat_0 2 1 3 in1 4 0=0\n"
        "torch.index_select old_pack 2 1 4 2 5\n"
        "Split splitncnn_0 1 2 5 6 7\n"
        "MatMul attention 2 1 6 7 8\n"
        "Concat cat_1 2 1 8 8 9 0=0\n"
        "Split splitncnn_1 1 2 9 10 11\n"
        "MemoryData pnnx_unique_0 0 1 12 0=874\n"
        "torch.index_select old_video 2 1 11 12 13\n"
        "Reshape old_video_reshape 1 1 13 out0 0=3 1=2 11=23 2=19\n"
        "MemoryData pnnx_fold_155 0 1 14 0=20\n"
        "torch.index_select old_text 2 1 10 14 15\n"
        "Reshape old_text_reshape 1 1 15 16 0=3 1=2 11=5 2=4\n"
        "Reduction old_mean 1 1 16 out1 0=3 1=0 -23303=1,0 4=0 5=1\n"
    )

    exporter.rewrite_ncnn_param(
        param_path,
        size=(2, 19, 23),
        windows=(4, 3, 3),
        text_tokens=5,
        shifted=True,
    )

    lines = param_path.read_text().splitlines()
    assert lines[1] == "13 16"
    assert any(line.startswith("SeedVR2AWAPack awa_pack") and " in0 in1 4 awa_cu_seqlens " in line for line in lines)
    assert any(line.startswith("SeedVR2AWAUnpack awa_unpack") and " 9 out0 out1 " in line for line in lines)
    assert not any(line.startswith("torch.index_select") for line in lines)
    assert not any(line.startswith("Reshape old_video") for line in lines)
    assert not any(line.startswith("Reduction old_mean") for line in lines)


def test_vae_shape_parser_requires_five_positive_dimensions():
    assert vae_exporter._shape("1,3,1,128,128") == (1, 3, 1, 128, 128)
    with pytest.raises(Exception):
        vae_exporter._shape("1,3,0,128,128")


def test_rewrite_vae_param_replaces_causal_temporal_tile(tmp_path):
    param_path = tmp_path / "vae_encode.ncnn.param"
    param_path.write_text(
        "7767517\n"
        "5 7\n"
        "Input in0 0 1 in0\n"
        "Split splitncnn_0 1 2 in0 original first_frame\n"
        "torch.tile torch.tile_370 1 1 first_frame repeated\n"
        "Concat cat_0 2 1 repeated original padded 0=1\n"
        "Convolution3D conv3d_25 1 1 padded out0 0=128\n"
    )

    vae_exporter.rewrite_ncnn_param(param_path)

    lines = param_path.read_text().splitlines()
    assert lines[1] == "3 7"
    assert "SeedVR2TemporalPad temporal_pad_0 1 1 in0 padded 0=2" in lines
    assert not any("torch.tile" in line for line in lines)
    assert not any(line.startswith("Split splitncnn_0") for line in lines)
    assert not any(line.startswith("Concat cat_0") for line in lines)


def test_rewrite_vae_param_replaces_upsample_depth_to_space_triplet(tmp_path):
    param_path = tmp_path / "vae_decode.ncnn.param"
    param_path.write_text(
        "7767517\n"
        "6 8\n"
        "Input in0 0 1 in0\n"
        "Convolution3D conv 1 1 in0 conv_out 0=4096\n"
        "Reshape reshape_218 1 1 conv_out reshaped 6=\"16,16,1,512,2,2,2\"\n"
        "Permute permute_153 1 1 reshaped permuted 0=0\n"
        "Reshape reshape_219 1 1 permuted out0 0=32 1=32 11=2 2=512\n"
        "Split splitncnn_17 1 2 out0 a b\n"
    )

    vae_exporter.rewrite_ncnn_param(param_path)

    lines = param_path.read_text().splitlines()
    assert lines[1] == "4 8"
    assert "SeedVR2DepthToSpace depth_to_space_0 1 1 conv_out out0 0=2 1=2 2=2" in lines
    assert not any(line.startswith("Reshape reshape_218") for line in lines)
    assert not any(line.startswith("Permute permute_153") for line in lines)
    assert not any(line.startswith("Reshape reshape_219") for line in lines)


def test_rewrite_vae_param_fixes_fixed_temporal_slice_start(tmp_path):
    param_path = tmp_path / "vae_decode.ncnn.param"
    param_path.write_text(
        "7767517\n"
        "2 4\n"
        "Input in0 0 1 in0\n"
        "Crop slice_1 1 1 in0 out0 -23310=1,-233 -23311=1,1 -23309=1,2\n"
    )

    vae_exporter.rewrite_ncnn_param(param_path)

    lines = param_path.read_text().splitlines()
    assert lines[1] == "2 4"
    assert "-23309=1,1" in lines[3]
