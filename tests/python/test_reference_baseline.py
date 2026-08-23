import json

import pytest
import torch
import torch.nn.functional as F

from tools.reference.awa_attention import (
    AwaAttentionPack,
    AwaAttentionUnpack,
    AwaWindowAttention,
)
from tools.reference.awa_window import AwaWindowPartition, AwaWindowReverse
from tools.reference.seedvr2_baseline import (
    awa_round_trip,
    build_manifest,
    make_reference_input,
    make_windows,
    save_tensor,
)


def test_awa_window_contract_includes_clipped_boundaries():
    windows = make_windows((1, 45, 80), (4, 3, 3), shifted=False)

    assert len(windows) == 9
    assert windows[0] == ((0, 1), (0, 15), (0, 27))
    assert windows[-1] == ((0, 1), (30, 45), (54, 80))

    with pytest.raises(ValueError, match="positive"):
        make_windows((1, 0, 23), (4, 3, 3), shifted=False)


def test_awa_round_trip_preserves_every_feature_value_for_both_modes():
    source = torch.arange(2 * 19 * 23 * 3, dtype=torch.float32).reshape(2, 19, 23, 3)

    for shifted in (False, True):
        restored, metadata = awa_round_trip(source, (4, 3, 3), shifted=shifted)
        partition = AwaWindowPartition((2, 19, 23), (4, 3, 3), shifted=shifted)
        reverse = AwaWindowReverse((2, 19, 23), (4, 3, 3), shifted=shifted)
        partitioned = partition(source)
        traced_partition = torch.jit.trace(partition, source)
        traced_reverse = torch.jit.trace(reverse, partitioned)

        assert torch.equal(reverse(partitioned), source)
        assert torch.equal(partitioned, source.reshape(-1, 3).index_select(0, partition.target_index))
        assert torch.equal(traced_partition(source), partitioned)
        assert torch.equal(traced_reverse(partitioned), source)
        assert torch.equal(restored, source)
        assert metadata["source_shape"] == [2, 19, 23, 3]
        assert metadata["window_count"] == len(metadata["windows"])

        heads, head_dim, text_tokens = 2, 3, 5
        vid_qkv = (
            torch.arange(2 * 19 * 23 * 3 * heads * head_dim, dtype=torch.float32)
            .reshape(2, 19, 23, 3, heads, head_dim)
            / 100
        )
        txt_qkv = (
            torch.arange(text_tokens * 3 * heads * head_dim, dtype=torch.float32)
            .reshape(text_tokens, 3, heads, head_dim)
            / 100
        )
        pack = AwaAttentionPack(
            (2, 19, 23), (4, 3, 3), text_tokens, shifted=shifted
        )
        unpack = AwaAttentionUnpack(
            (2, 19, 23), (4, 3, 3), text_tokens, shifted=shifted
        )
        packed, cu_seqlens = pack(vid_qkv, txt_qkv)

        expected_segments = []
        source_index = torch.arange(2 * 19 * 23).reshape(2, 19, 23)
        for window in make_windows((2, 19, 23), (4, 3, 3), shifted=shifted):
            time, height, width = window
            expected_segments.append(
                vid_qkv.reshape(-1, 3, heads, head_dim).index_select(
                    0,
                    source_index[
                        time[0] : time[1], height[0] : height[1], width[0] : width[1]
                    ].reshape(-1),
                )
            )
            expected_segments.append(txt_qkv)
        expected = torch.cat(expected_segments)
        assert torch.equal(packed, expected)
        expected_cu_seqlens = [0, 214, 428, 661, 894] if shifted else [0, 423, 846, 870, 894]
        assert cu_seqlens.tolist() == expected_cu_seqlens

        video_out, text_out = unpack(packed[:, 0])
        assert torch.equal(video_out, vid_qkv[:, :, :, 0])
        assert torch.equal(text_out, txt_qkv[:, 0])

        attention = AwaWindowAttention(cu_seqlens, head_dim=head_dim)
        attended = attention(packed)
        expected_attended = []
        for start, end in zip(expected_cu_seqlens[:-1], expected_cu_seqlens[1:]):
            segment = packed[start:end]
            expected_attended.append(
                F.scaled_dot_product_attention(
                    segment[:, 0].transpose(0, 1),
                    segment[:, 1].transpose(0, 1),
                    segment[:, 2].transpose(0, 1),
                ).transpose(0, 1)
            )
        expected_attended = torch.cat(expected_attended)
        assert torch.allclose(attended, expected_attended, atol=1e-6, rtol=1e-6)

        traced_pack = torch.jit.trace(pack, (vid_qkv, txt_qkv))
        traced_unpack = torch.jit.trace(unpack, (packed[:, 0],))
        traced_attention = torch.jit.trace(attention, (packed,))
        traced_packed, traced_cu_seqlens = traced_pack(vid_qkv, txt_qkv)
        traced_video, traced_text = traced_unpack(packed[:, 0])
        assert torch.equal(traced_packed, packed)
        assert torch.equal(traced_cu_seqlens, cu_seqlens)
        assert torch.equal(traced_video, video_out)
        assert torch.equal(traced_text, text_out)
        assert torch.allclose(traced_attention(packed), attended, atol=1e-6, rtol=1e-6)


def test_reference_artifacts_are_deterministic_and_checksummed(tmp_path):
    first = make_reference_input((1, 8, 12), seed=666)
    second = make_reference_input((1, 8, 12), seed=666)

    assert first.shape == (3, 1, 8, 12)
    assert torch.equal(first, second)
    assert float(first.min()) >= -1.0
    assert float(first.max()) <= 1.0

    tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    path = tmp_path / "tensor.pt"

    digest = save_tensor(path, tensor)
    manifest = build_manifest({"tensor": path}, metadata={"seed": 666})

    assert digest == manifest["artifacts"]["tensor"]["sha256"]
    assert manifest["artifacts"]["tensor"]["shape"] == [1, 2]
    json.dumps(manifest)
