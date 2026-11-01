"""out_of_grid: what an edge does beyond its grid."""
import pytest
import torch

from xkan import KANNetwork, fit
from xkan.io import load_checkpoint, save_checkpoint
from xkan.layers import KANLayer

MODES = ("zero", "clamp", "soft_clamp")


def layer(mode, seed=0, **kw):
    torch.manual_seed(seed)
    out = KANLayer(3, 2, num=5, k=3, out_of_grid=mode, base_fun="silu", **kw)
    out.update_grid_from_samples(torch.rand(400, 3) * 2 - 1)
    return out


def test_options_are_validated():
    with pytest.raises(ValueError, match="out_of_grid"):
        KANLayer(2, 2, out_of_grid="extrapolate")


def test_inside_the_grid_every_mode_is_the_same():
    # the same parameters under each mode (the modes refit differently at a grid update)
    x = torch.rand(50, 3) * 1.6 - 0.8
    lay = layer("zero")
    ref = lay(x)
    for mode in MODES[1:]:
        lay.out_of_grid = mode
        assert torch.equal(lay(x), ref)
        assert torch.allclose(lay(x, return_edges=True).y, ref, atol=1e-6)


def test_clamp_holds_the_edge_at_the_boundary():
    lay = layer("clamp")
    lo, hi = lay.grid[:, lay.k], lay.grid[:, -lay.k - 1]
    far = torch.stack([hi + 50, lo - 50])
    assert torch.allclose(lay(far), lay(torch.stack([hi, lo])), atol=1e-6)
    x = (torch.stack([hi + 3])).requires_grad_(True)
    lay(x).sum().backward()
    assert torch.equal(x.grad, torch.zeros_like(x.grad))


def test_soft_clamp_saturates_within_one_interval_and_keeps_a_gradient():
    lay = layer("soft_clamp")
    k = lay.k
    lo, hi = lay.grid[:, k], lay.grid[:, -k - 1]
    h_lo, h_hi = lay.grid[:, k] - lay.grid[:, k - 1], lay.grid[:, -k] - lay.grid[:, -k - 1]
    x = torch.stack([hi + 1e-4, hi + 50, lo - 1e-4, lo - 50]).requires_grad_(True)
    xs = lay.squash_to_grid(x)
    assert torch.allclose(xs[0], hi, atol=2e-4) and torch.allclose(xs[2], lo, atol=2e-4)  # continuous at the ends
    assert (xs[1] <= hi + h_hi).all() and (xs[1] > hi + 0.99 * h_hi).all() and (xs[3] >= lo - h_lo).all()
    inside = torch.rand(30, 3) * (hi - lo) * 0.9 + lo + 0.05 * (hi - lo)
    assert torch.equal(lay.squash_to_grid(inside), inside)
    xs[[0, 2]].sum().backward()  # slope 1 at the ends: the edges do not go flat there
    assert torch.allclose(x.grad[[0, 2]], torch.ones_like(x.grad[[0, 2]]), atol=1e-3)
    assert torch.allclose(lay(torch.stack([hi + 50])), lay(torch.stack([hi + 500])), atol=1e-6)  # bounded


def test_soft_clamp_trains_where_the_hard_clamp_collapses():
    assert KANLayer(2, 2).out_of_grid == "zero"
    torch.manual_seed(0)
    x = torch.rand(300, 3) * 2 - 1
    y = torch.sin(torch.pi * x[:, :1]) + x[:, 1:2] ** 2
    loss = {}
    for mode in ("soft_clamp", "clamp"):
        net = KANNetwork([3, 6, 4, 1], grid=4, k=3, seed=1, out_of_grid=mode)
        loss[mode] = fit(net, {"x": x, "y": y}, steps=40, opt="adam", lr=2e-2, lamb=1e-2)["train/y"][-1]
    assert loss["soft_clamp"] < 0.1 < loss["clamp"]


def test_grid_update_refits_as_the_bounded_edges_evaluate():
    # inputs beyond the initial grid (-1, 1): the refit of the curves as the bounded edges evaluate them keeps the
    # function several times better than a refit of the raw curves
    def change(mode, seed, raw=False):
        torch.manual_seed(seed)
        lay = KANLayer(2, 2, num=5, k=3, out_of_grid=mode)
        x = torch.randn(1000, 2)
        before = lay(x).detach()
        if raw:
            lay.out_of_grid = "zero"
        lay.update_grid_from_samples(x)
        lay.out_of_grid = mode
        return float(((lay(x).detach() - before) ** 2).mean().sqrt())
    for mode in ("clamp", "soft_clamp"):
        for seed in range(3):
            assert change(mode, seed) < 0.03 and change(mode, seed) < change(mode, seed, raw=True) / 3


def test_grid_update_after_drift_under_soft_clamp():
    # inputs 30% wider than the grid: the soft clamp's update changes the function as little as pykan's, the hard
    # clamp's (its curves flat beyond the old range, their new knots inside it) more
    def change(mode, seed):
        torch.manual_seed(seed)
        lay = KANLayer(2, 2, num=5, k=3, out_of_grid=mode)
        x = torch.randn(1000, 2)
        lay.update_grid_from_samples(x)
        with torch.no_grad():
            lay.coef.add_(0.3 * torch.randn_like(lay.coef))
        before = lay(1.3 * x).detach()
        lay.update_grid_from_samples(1.3 * x)
        return float((lay(1.3 * x).detach() - before).abs().max())
    for seed in range(3):
        assert change("soft_clamp", seed) < 1.25 * change("zero", seed) and change("soft_clamp", seed) < change("clamp", seed) / 2


def test_network_options_and_checkpoints(tmp_path):
    net = KANNetwork([3, 4, 1], grid=3, k=2, seed=0, out_of_grid="clamp")
    assert all(lay.out_of_grid == "clamp" for lay in net.layers)
    save_checkpoint(net, tmp_path / "ck")
    back = load_checkpoint(tmp_path / "ck")
    x = torch.rand(20, 3) * 10 - 5
    assert torch.equal(back(x), net(x))
    assert back.layers[0].out_of_grid == "clamp"


def test_refine_keeps_a_layers_own_mode():
    from xkan.interp import refine
    net = KANNetwork([2, 3, 1], grid=3, k=3, seed=0)
    net.layers[0].out_of_grid = "clamp"
    x = torch.rand(200, 2) * 2 - 1
    r = refine(net, 6, x)
    assert [lay.out_of_grid for lay in r.layers] == ["clamp", "zero"]
