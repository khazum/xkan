"""Spline weight decay in its own parameter group, the spline-share and out-of-grid diagnostics, the SplineShare
callback."""
import pytest
import torch
from torch import nn

from xkan import KANNetwork
from xkan.interp import layer_inputs, outside_grid, spline_share
from xkan.layers import KANLayer
from xkan.losses import MSE, CompositeLoss
from xkan.train import SplineShare, Trainer


def data(n=200):
    g = torch.Generator().manual_seed(0)
    x = torch.rand(n, 2, generator=g) * 2 - 1
    return x, torch.sin(torch.pi * x[:, :1]) + x[:, 1:] ** 2


def test_spline_weight_decay_groups():
    # 0 (default): the splines in a group of their own without decay, the optimizer's decay on the rest
    net = KANNetwork([2, 5, 1], grid=5, k=3, seed=3)
    for opt in ("adam", "adamw", torch.optim.SGD):
        tr = Trainer(net, CompositeLoss({"y": MSE()}), opt, 1e-2, optimizer_options={"weight_decay": 0.1})
        other, spline = tr.optimizer.param_groups
        names = {id(p): n for n, p in net.named_parameters()}
        assert sorted(names[id(p)].split(".")[-1] for p in spline["params"]) == ["coef", "coef", "scale_sp", "scale_sp"]
        assert spline["weight_decay"] == 0. and other["weight_decay"] == 0.1
        assert not {id(p) for p in spline["params"]} & {id(p) for p in other["params"]}
    tr = Trainer(net, CompositeLoss({"y": MSE()}), "adam", 1e-2, optimizer_options={"weight_decay": 0.1},
                 spline_weight_decay=1e-3)
    assert [g["weight_decay"] for g in tr.optimizer.param_groups] == [0.1, 1e-3]
    # None: one group, the optimizer's decay for every parameter
    tr = Trainer(net, CompositeLoss({"y": MSE()}), "adam", 1e-2, optimizer_options={"weight_decay": 0.1},
                 spline_weight_decay=None)
    assert len(tr.optimizer.param_groups) == 1 and tr.optimizer.param_groups[0]["weight_decay"] == 0.1
    # LBFGS (no weight decay) and optimizer instances (their own groups) take the default, not a decay
    assert len(Trainer(net, CompositeLoss({"y": MSE()}), "lbfgs").optimizer.param_groups) == 1
    adam = torch.optim.Adam(net.parameters())
    assert Trainer(net, CompositeLoss({"y": MSE()}), adam).optimizer is adam
    with pytest.raises(ValueError, match="LBFGS"):
        Trainer(net, CompositeLoss({"y": MSE()}), "lbfgs", spline_weight_decay=1e-3)
    with pytest.raises(ValueError, match="instance"):
        Trainer(net, CompositeLoss({"y": MSE()}), torch.optim.Adam(net.parameters()), spline_weight_decay=1e-3)


def test_default_groups_train_as_one_group():
    # without weight decay the two groups train exactly as one
    x, y = data()
    params = []
    for swd in (0., None):
        net = KANNetwork([2, 5, 1], grid=5, k=3, seed=3)
        Trainer(net, CompositeLoss({"y": MSE()}), "adam", 1e-2, spline_weight_decay=swd, seed=0).fit({"x": x, "y": y}, steps=50)
        params.append(torch.cat([p.detach().flatten() for p in net.parameters()]))
    assert torch.equal(params[0], params[1])


def test_spline_weight_decay_keeps_the_splines():
    # Adam's (coupled) weight decay shrinks the spline scales; exempting them keeps them and the fit
    x, y = data()
    out = {}
    for swd in (None, 0.):
        net = KANNetwork([2, 5, 1], grid=5, k=3, seed=3)
        tr = Trainer(net, CompositeLoss({"y": MSE()}), "adam", 1e-2, optimizer_options={"weight_decay": 2e-2},
                     spline_weight_decay=swd, seed=0)
        hist = tr.fit({"x": x, "y": y}, steps=600)
        out[swd] = (float(net.layers[0].scale_sp.detach().abs().mean()), hist["train/y"][-1])
    assert out[None][0] < 0.3 and out[0.][0] > 0.9
    assert out[0.][1] < out[None][1] / 10


def test_spline_share_bounds_and_collapse():
    x, _ = data()
    net = KANNetwork([2, 5, 1], grid=5, k=3, seed=3)
    share = spline_share(net, x)
    assert list(share) == ["layers.0", "layers.1"] and all(0 < v < 1 for v in share.values())
    for layer in net.layers:
        layer.scale_sp.data.zero_()
    assert all(v == 0 for v in spline_share(net, x).values())
    for layer in net.layers:
        layer.scale_base.data.zero_()
        layer.scale_sp.data.fill_(1.)
    assert all(v == pytest.approx(1) for v in spline_share(net, x).values())


def test_diagnostics_work_on_any_module_of_kan_layers():
    torch.manual_seed(0)
    model = nn.Sequential(KANLayer(3, 4, num=3, k=2), nn.LayerNorm(4), KANLayer(4, 2, num=3, k=2))
    x = torch.rand(50, 3) * 2 - 1
    inputs = layer_inputs(model, x)
    assert list(inputs) == ["0", "2"] and inputs["2"][1].shape == (50, 4)
    assert set(spline_share(model, x)) == {"0", "2"}
    assert outside_grid(model, x)["0"] == 0.
    assert outside_grid(model, x * 5)["0"] > 0.5


def test_spline_share_callback_records():
    x, y = data()
    net = KANNetwork([2, 5, 1], grid=5, k=3, seed=3)
    cb = SplineShare(x, every=10)
    Trainer(net, CompositeLoss({"y": MSE()}), "adam", 1e-2, callbacks=[cb]).fit({"x": x, "y": y}, steps=25)
    assert [r["step"] for r in cb.records] == [0, 10, 20, 25]
    assert set(cb.records[0]) == {"step", "layers.0", "layers.1"}
    with pytest.raises(ValueError):
        SplineShare(x, every=0)
