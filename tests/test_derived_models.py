"""The adaptive layer's minimum of collected rows before an update."""

import pytest
import torch


from xkan.layers import AdaptiveKANLayer


def test_adaptive_layer_min_rows():
    with pytest.raises(ValueError, match="min_rows"):
        AdaptiveKANLayer(in_dim=2, out_dim=2, min_rows=0)
    layer = AdaptiveKANLayer(in_dim=2, out_dim=2, min_rows=1, update_at=(1, 2))
    layer.train()
    for _ in range(3):
        layer(torch.randn(8, 2))
    assert layer.n_updates == 2


