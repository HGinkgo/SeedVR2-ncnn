from pathlib import Path
import subprocess
import sys

import pytest
import torch


def test_prepare_conditioning_writes_float32_record(tmp_path: Path):
    from tools.prepare_conditioning import prepare_conditioning

    source = tmp_path / "pos_emb.pt"
    torch.save(torch.ones((58, 5120), dtype=torch.bfloat16), source)
    output = tmp_path / "pos_emb.f32"

    manifest = prepare_conditioning(source, output)

    assert manifest["shape"] == [58, 5120]
    assert manifest["dtype"] == "float32"
    assert manifest["count"] == 58 * 5120
    assert output.stat().st_size == 58 * 5120 * 4
    assert output.read_bytes()[:4] == bytes.fromhex("0000803f")


def test_prepare_conditioning_rejects_wrong_token_count(tmp_path: Path):
    from tools.prepare_conditioning import prepare_conditioning

    source = tmp_path / "neg_emb.pt"
    torch.save(torch.zeros((64, 5120), dtype=torch.bfloat16), source)

    with pytest.raises(ValueError, match=r"expected shape \(58, 5120\)"):
        prepare_conditioning(source, tmp_path / "neg_emb.f32")


def test_prepare_conditioning_cli_accepts_explicit_negative_shape(tmp_path: Path):
    source = tmp_path / "neg_emb.pt"
    output = tmp_path / "neg_emb.f32"
    torch.save(torch.zeros((64, 5120), dtype=torch.bfloat16), source)

    result = subprocess.run(
        [
            sys.executable,
            "tools/prepare_conditioning.py",
            str(source),
            str(output),
            "--tokens",
            "64",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output.stat().st_size == 64 * 5120 * 4
