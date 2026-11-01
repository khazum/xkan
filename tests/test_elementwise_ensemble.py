"""The elementwise KAN layer (one edge per input, the outputs not summed) and ensembles of fitted members."""
import pytest
import torch

from xkan import KANNetwork, fit
from xkan.layers import ElementwiseKANLayer, KANLayer
from xkan.nn import Ensemble, fit_ensemble


def test_elementwise_is_the_unsummed_edges_of_a_one_output_layer():
    torch.manual_seed(0)
    el = ElementwiseKANLayer(4, num=5, k=3, base_fun="silu")
    full = KANLayer(4, 1, num=5, k=3, base_fun="silu")
    full.load_state_dict(el.state_dict())
    x = torch.rand(30, 4) * 2 - 1
    y = el(x)
    assert y.shape == (30, 4)
    assert torch.allclose(y, full(x, return_edges=True).postacts[:, 0, :], atol=1e-6)
    assert torch.allclose(y.sum(1, keepdim=True), full(x), atol=1e-5)
    ev = el(x, return_edges=True)
    assert torch.allclose(ev.y, y, atol=1e-6) and ev.postacts.shape == (30, 1, 4)


def test_elementwise_options_grid_update_and_training():
    with pytest.raises(ValueError):
        ElementwiseKANLayer(3, aggregation="mean")
    with pytest.raises(ValueError):
        ElementwiseKANLayer(3, bias=True)
    torch.manual_seed(1)
    el = ElementwiseKANLayer(2, num=5, k=3, out_of_grid="clamp")
    x = torch.randn(500, 2) * 2
    el.update_grid_from_samples(x)
    target = torch.stack([torch.sin(x[:, 0]), x[:, 1] ** 2 / 4], 1)
    opt = torch.optim.Adam(el.parameters(), lr=1e-2)
    with torch.no_grad():
        start = float(((el(x) - target) ** 2).mean())
    for _ in range(400):
        opt.zero_grad()
        loss = ((el(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
    assert float(loss.detach()) < start / 10
    assert torch.isfinite(el(torch.tensor([[1e4, -1e4]]))).all()


def test_ensemble_mean_agreement_and_selection():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(200, 2, generator=g) * 2 - 1
    y = torch.sin(torch.pi * x[:, :1]) + x[:, 1:] ** 2
    train, val = {"x": x[:150], "y": y[:150]}, {"x": x[150:], "y": y[150:]}

    def score(m):
        with torch.no_grad():
            return float(((m(val["x"]) - val["y"]) ** 2).mean())
    ens, info = fit_ensemble(lambda i: KANNetwork([2, 3, 1], grid=5, k=3, seed=i),
                             lambda m, i: fit(m, train, steps=30, opt="adam", seed=i), n=4, keep=2, score=score)
    assert len(ens.members) == 2 and len(info["scores"]) == 4
    assert sorted(info["scores"])[:2] == sorted(info["scores"][i] for i in info["kept"])
    out = ens(val["x"])
    assert torch.allclose(out, torch.stack([m(val["x"]) for m in ens.members]).mean(0))
    a = ens.agreement(val["x"])
    assert 0 < a <= 1
    assert Ensemble([ens.members[0]]).agreement(val["x"]) is None
    with pytest.raises(ValueError):
        fit_ensemble(lambda i: None, lambda m, i: None, n=2, keep=3)
