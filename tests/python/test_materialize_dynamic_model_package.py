from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys

import pytest


def _write_graph(root: Path, name: str, param: str) -> None:
    (root / f"{name}.ncnn.param").write_text(param)
    (root / f"{name}.ncnn.bin").write_bytes(f"{name}-weights".encode("ascii"))


def _awa_param() -> str:
    return (
        "7767517\n"
        "5 6\n"
        "Input video 0 1 video\n"
        "SeedVR2AWAPack pack 1 1 video packed 0=1 1=16 2=16 3=1 4=4 5=3 6=3\n"
        "SeedVR2AWAUnpack unpack 1 2 packed video_out text_out 0=1 1=16 2=16 3=1 4=4 5=3 6=3\n"
        "Reshape reshape_awa_video_batch 1 1 video_out video_batch 0=2560 1=256 12=233 13=0\n"
        "SeedVR2MMRoPE rope 1 1 video_batch rope_out 0=1 1=16 2=16 3=1 4=4 5=3 6=3\n"
    )


def _create_source_package(root: Path) -> tuple[Path, Path]:
    variant = root / "256x256"
    conditioning = root / "conditioning"
    variant.mkdir(parents=True)
    conditioning.mkdir()

    vae_encode_param = "7767517\n2 2\nInput in0 0 1 in0\nReshape spatial 1 1 in0 out0 0=2 1=3 2=4 11=0\n"
    vae_decode_param = (
        "7767517\n"
        "3 3\n"
        "Input in0 0 1 in0\n"
        "SeedVR2DepthToSpace depth_to_space_2 1 1 in0 upscaled 0=1 1=2 2=2\n"
        "Reshape spatial 1 1 upscaled out0 0=2 1=3 2=4 11=0\n"
    )
    gemm_param = (
        "7767517\n2 2\nInput in0 0 1 in0\n"
        "Gemm projection 1 1 in0 out0 7=256 8=2560 9=132\n"
    )
    passthrough_param = "7767517\n1 1\nInput in0 0 1 in0\n"
    _write_graph(variant, "vae_encode", vae_encode_param)
    _write_graph(variant, "vae_decode", vae_decode_param)
    for name in ("dit_input", "dit_output"):
        _write_graph(variant, name, gemm_param)
    _write_graph(variant, "dit_embedding", passthrough_param)
    for index in range(32):
        _write_graph(variant, f"dit_block_{index:02d}", _awa_param())
    conditioning_file = conditioning / "pos_emb.f32"
    conditioning_file.write_bytes(b"conditioning")
    return variant, conditioning


def test_materialize_dynamic_model_package_flattens_and_normalizes(tmp_path: Path):
    from tools.materialize_dynamic_model_package import materialize_dynamic_model_package

    source_variant, source_conditioning = _create_source_package(tmp_path / "source")
    destination = tmp_path / "dynamic-package"

    materialize_dynamic_model_package(source_variant, source_conditioning, destination)

    assert not (destination / "256x256").exists()
    assert (destination / "vae_encode.ncnn.bin").read_bytes() == b"vae_encode-weights"
    assert '6="0w,0h,0c,1"' in (destination / "vae_encode.ncnn.param").read_text()
    assert "SeedVR2DepthToSpace depth_to_space_2 1 1 in0 upscaled 0=2 1=2 2=1" in (
        destination / "vae_decode.ncnn.param"
    ).read_text()
    assert "7=0 8=2560 9=132" in (destination / "dit_input.ncnn.param").read_text()
    block_param = (destination / "dit_block_00.ncnn.param").read_text()
    assert block_param.count("0=-1 1=-1 2=-1") == 3
    assert "reshape_awa_video_batch 1 1 video_out video_batch 0=2560 1=-1" in block_param
    assert (destination / "conditioning" / "pos_emb.f32").read_bytes() == b"conditioning"
    assert not any(path.is_symlink() for path in destination.rglob("*"))

    manifest = destination / "manifest.sha256"
    entries = manifest.read_text().splitlines()
    assert len(entries) == 75
    first_digest, first_name = entries[0].split("  ", 1)
    assert first_name == "conditioning/pos_emb.f32"
    assert first_digest == hashlib.sha256((destination / first_name).read_bytes()).hexdigest()

    with pytest.raises(FileExistsError):
        materialize_dynamic_model_package(source_variant, source_conditioning, destination)


def test_materialize_dynamic_model_package_accepts_pre_normalized_vae_templates(tmp_path: Path):
    from tools.export_vae import normalize_dynamic_vae_template
    from tools.materialize_dynamic_model_package import materialize_dynamic_model_package

    source_variant, source_conditioning = _create_source_package(tmp_path / "source")
    for graph_name in ("vae_encode", "vae_decode"):
        normalize_dynamic_vae_template(source_variant / f"{graph_name}.ncnn.param")

    destination = tmp_path / "dynamic-package"
    materialize_dynamic_model_package(source_variant, source_conditioning, destination)

    assert '6="0w,0h,0c,1"' in (destination / "vae_encode.ncnn.param").read_text()


def test_materialize_dynamic_model_package_cli_runs_outside_repository(tmp_path: Path):
    source_variant, source_conditioning = _create_source_package(tmp_path / "source")
    destination = tmp_path / "dynamic-package"
    repository_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            str(repository_root / "tools" / "materialize_dynamic_model_package.py"),
            "--source-variant-dir",
            str(source_variant),
            "--source-conditioning-dir",
            str(source_conditioning),
            "--output-dir",
            str(destination),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert (destination / "manifest.sha256").is_file()
