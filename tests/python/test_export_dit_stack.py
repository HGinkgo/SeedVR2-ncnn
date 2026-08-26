import subprocess
import sys
from pathlib import Path

import pytest
import torch


def test_full_dit_stack_contract_is_explicit():
    from tools.reference.dit_stack import DIT_STACK_CONTRACT

    assert DIT_STACK_CONTRACT.source_shape == (1, 8, 8)
    assert DIT_STACK_CONTRACT.window_shape == (4, 3, 3)
    assert DIT_STACK_CONTRACT.text_tokens == 58
    assert DIT_STACK_CONTRACT.video_patch_shape == (64, 132)
    assert DIT_STACK_CONTRACT.text_input_shape == (58, 5120)
    assert DIT_STACK_CONTRACT.embedding_shape == (1, 15360)
    assert DIT_STACK_CONTRACT.block_count == 32
    assert DIT_STACK_CONTRACT.block_shifted(0) is False
    assert DIT_STACK_CONTRACT.block_shifted(1) is True
    assert DIT_STACK_CONTRACT.block_shifted(31) is True
    assert DIT_STACK_CONTRACT.output_shape == (64, 64)
    assert DIT_STACK_CONTRACT.graph_names == (
        "dit_input",
        "dit_embedding",
        *(f"dit_block_{index:02d}" for index in range(32)),
        "dit_output",
    )


def test_dit_stack_supports_the_official_positive_and_negative_token_specs():
    from tools.reference.dit_stack import make_dit_stack_contract

    positive = make_dit_stack_contract(58)
    negative = make_dit_stack_contract(64)

    assert positive.text_input_shape == (58, 5120)
    assert negative.text_input_shape == (64, 5120)
    assert negative.video_patch_shape == positive.video_patch_shape


def test_dit_stack_rejects_unfrozen_text_token_specs():
    from tools.reference.dit_stack import make_dit_stack_contract

    with pytest.raises(ValueError, match="58 or 64"):
        make_dit_stack_contract(60)


def test_full_dit_stack_export_script_has_a_stable_cli():
    result = subprocess.run(
        [sys.executable, "tools/export_dit_stack.py", "--help"],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--checkpoint" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--text-tokens" in result.stdout


def test_portable_golden_pack_has_stable_little_endian_layout(tmp_path):
    from tools.export_dit_stack import GOLDEN_MAGIC, write_portable_golden

    golden_path = tmp_path / "golden.f32"
    manifest = write_portable_golden(
        golden_path,
        [
            ("input_video", torch.tensor([[1.0, 2.0]], dtype=torch.float32)),
            ("output", torch.tensor([[-3.0]], dtype=torch.float32)),
        ],
    )

    assert manifest["version"] == 1
    assert manifest["record_count"] == 2
    assert [record["name"] for record in manifest["records"]] == ["input_video", "output"]
    assert golden_path.read_bytes()[: len(GOLDEN_MAGIC)] == GOLDEN_MAGIC
    assert manifest["records"][0]["shape"] == [1, 2]
    assert manifest["records"][0]["count"] == 2


def test_vulkan_stack_test_target_is_declared():
    cmake = (Path(__file__).parents[2] / "CMakeLists.txt").read_text()

    assert "seedvr2-dit-stack-vulkan-test" in cmake
