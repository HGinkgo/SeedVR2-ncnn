import json

import pytest
import torch

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
