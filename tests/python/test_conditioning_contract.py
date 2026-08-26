from pathlib import Path

import pytest
import torch


def test_conditioning_pair_preserves_official_prompt_lengths(tmp_path: Path):
    from tools.reference.conditioning import load_conditioning_pair

    positive_path = tmp_path / "pos_emb.pt"
    negative_path = tmp_path / "neg_emb.pt"
    torch.save(torch.ones((58, 5120), dtype=torch.bfloat16), positive_path)
    torch.save(torch.zeros((64, 5120), dtype=torch.bfloat16), negative_path)

    positive, negative = load_conditioning_pair(positive_path, negative_path)

    assert positive.shape == (58, 5120)
    assert negative.shape == (64, 5120)
    assert positive.dtype == torch.float32
    assert negative.dtype == torch.float32


def test_conditioning_pair_rejects_wrong_embedding_width(tmp_path: Path):
    from tools.reference.conditioning import load_conditioning_pair

    positive_path = tmp_path / "pos_emb.pt"
    negative_path = tmp_path / "neg_emb.pt"
    torch.save(torch.ones((58, 5119), dtype=torch.bfloat16), positive_path)
    torch.save(torch.zeros((64, 5120), dtype=torch.bfloat16), negative_path)

    with pytest.raises(ValueError, match=r"positive.*expected width 5120"):
        load_conditioning_pair(positive_path, negative_path)


def test_cfg_matches_official_formula():
    from tools.reference.conditioning import classifier_free_guidance

    positive = torch.tensor([[2.0, 4.0]])
    negative = torch.tensor([[1.0, 1.0]])

    result = classifier_free_guidance(positive, negative, scale=7.5)

    assert torch.equal(result, negative + 7.5 * (positive - negative))


def test_uniform_trailing_timesteps_match_seedvr2_defaults():
    from tools.reference.conditioning import uniform_trailing_timesteps

    timesteps = uniform_trailing_timesteps()

    assert timesteps.shape == (50,)
    assert timesteps[0].item() == pytest.approx(1000.0)
    assert timesteps[-1].item() == pytest.approx(20.0)


def test_timestep_transform_matches_official_128px_single_frame_case():
    from tools.reference.conditioning import transform_timestep

    result = transform_timestep(torch.tensor([500.0]), torch.tensor([[1, 16, 16]]))

    assert result.shape == (1,)
    assert result.item() == pytest.approx(470.89947, rel=1e-6)
