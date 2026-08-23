import json

import pytest
import torch

from tools.reference.seedvr2_baseline import (
    awa_round_trip,
    build_manifest,
    make_reference_input,
    make_windows,
    save_tensor,
)


def test_make_windows_matches_clipped_3x3_order():
    windows = make_windows((1, 45, 80), (4, 3, 3), shifted=False)

    assert len(windows) == 9
    assert windows[0] == ((0, 1), (0, 15), (0, 27))
    assert windows[-1] == ((0, 1), (30, 45), (54, 80))


@pytest.mark.parametrize("shifted", [False, True])
def test_awa_round_trip_preserves_every_feature_value(shifted):
    source = torch.arange(2 * 19 * 23 * 3, dtype=torch.float32).reshape(2, 19, 23, 3)

    restored, metadata = awa_round_trip(source, (4, 3, 3), shifted=shifted)

    assert torch.equal(restored, source)
    assert metadata["source_shape"] == [2, 19, 23, 3]
    assert metadata["window_count"] == len(metadata["windows"])


def test_make_windows_rejects_non_positive_dimensions():
    with pytest.raises(ValueError, match="positive"):
        make_windows((1, 0, 23), (4, 3, 3), shifted=False)


def test_manifest_contains_tensor_sha256(tmp_path):
    tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    path = tmp_path / "tensor.pt"

    digest = save_tensor(path, tensor)
    manifest = build_manifest({"tensor": path}, metadata={"seed": 666})

    assert digest == manifest["artifacts"]["tensor"]["sha256"]
    assert manifest["artifacts"]["tensor"]["shape"] == [1, 2]
    json.dumps(manifest)


def test_make_reference_input_is_deterministic_and_normalized():
    first = make_reference_input((1, 8, 12), seed=666)
    second = make_reference_input((1, 8, 12), seed=666)

    assert first.shape == (3, 1, 8, 12)
    assert torch.equal(first, second)
    assert float(first.min()) >= -1.0
    assert float(first.max()) <= 1.0
