from pathlib import Path

import torch


def test_fixed_dit_pipeline_contract_uses_vae_latent_grid():
    from tools.reference.dit_pipeline import (
        DIT_PIPELINE_CONTRACT,
        FixedDitPipeline,
    )

    assert DIT_PIPELINE_CONTRACT.video_shape == (1, 33, 1, 16, 16)
    assert DIT_PIPELINE_CONTRACT.text_shape == (58, 5120)
    assert DIT_PIPELINE_CONTRACT.embedding_shape == (1, 15360)
    assert DIT_PIPELINE_CONTRACT.output_shape == (1, 16, 1, 16, 16)

    model = FixedDitPipeline(checkpoint=Path("ckpts/seedvr2_ema_3b.pth"))
    video = torch.zeros(DIT_PIPELINE_CONTRACT.video_shape)
    text = torch.zeros(DIT_PIPELINE_CONTRACT.text_shape)
    embedding = torch.zeros(DIT_PIPELINE_CONTRACT.embedding_shape)
    output = model(video, text, embedding)
    assert tuple(output.shape) == DIT_PIPELINE_CONTRACT.output_shape
    assert torch.isfinite(output).all()
