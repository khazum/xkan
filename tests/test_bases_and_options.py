"""Layer options: basis ('bspline', 'rbf', 'chebyshev'), knot_placement='zero_aware', dropkan, aggregation
('sum', 'mean'), bias: bases, ranges, refits across bases, initial curves, fused and per-edge forwards, gradients,
grid updates, validation."""
from pathlib import Path
import pytest
import torch

import xkan

from xkan.layers import KANLayer, basis_values
from xkan import spline as REV

# the checkout (or installation) under test: the folder holding the xkan package, for the tests that start a fresh
# Python process
REV_DIR = Path(xkan.__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


BASES = ("bspline", "rbf", "rswaf", "chebyshev")


def inputs(kind, n=500, d=4, seed=0, device="cpu", dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    if kind == "dense":
        x = torch.randn(n, d, generator=g)
    elif kind == "zero-inflated":  # counts-like: exact zeros, positive values
        x = torch.log1p(torch.poisson(torch.rand(n, d, generator=g) * 3) * (torch.rand(n, d, generator=g) < 0.4))
    elif kind == "ties":
        x = torch.randint(0, 4, (n, d), generator=g).float() * 0.5
    elif kind == "constant":
        x = torch.randn(n, d, generator=g)
        x[:, 1] = 0.3
        x[:, 2] = 0.
    elif kind == "negative-zeros":  # zeros and negatives: not zero-inflated in the knot rule's sense
        x = torch.randn(n, d, generator=g) * (torch.rand(n, d, generator=g) < 0.5)
    else:
        raise ValueError(kind)
    return x.to(device=device, dtype=dtype)


def edge_curves(layer, x):
    # the layer's spline curves (batch, in_dim, out_dim), any basis
    return torch.einsum("ijk,jlk->ijl", layer.basis_values(x), layer.coef)


def test_zero_aware_grid_properties():
    x = inputs("zero-inflated", n=2000)
    g = REV.zero_aware_grid(x, 5, 3)
    assert (g[:, 1:] - g[:, :-1] > 0).all()  # strictly increasing (the minimum knot separation)
    zero_inflated = (x == 0).any(0) & (x > 0).any(0) & (x >= 0).all(0)
    assert zero_inflated.all() and torch.allclose(g[:, 3], torch.zeros(4), atol=1e-6)  # a knot at 0
    for bad in (torch.zeros(0, 3), torch.zeros(5)):
        with pytest.raises(ValueError):
            REV.zero_aware_grid(bad, 5, 3)


def test_basis_values_dispatch():
    x = inputs("dense")
    grid = REV.extend_grid(torch.linspace(-2, 2, 6)[None].repeat(4, 1), k_extend=3)
    from xkan.layers import BASES as REGISTRY, basis_values
    assert torch.equal(basis_values(x, grid, 3, "bspline"), REV.B_batch(x, grid, 3))
    with pytest.raises(ValueError, match="basis"):
        basis_values(x, grid, 3, "fourier")
    assert tuple(REGISTRY) == BASES


def test_basis_ranges():
    # rbf in (0, 1] with a peak of 1 at each centre; chebyshev bounded by 1, T_0 = 1
    num, k = 5, 3
    grid = REV.extend_grid(torch.linspace(-1, 1, num + 1)[None], k_extend=k)
    centres = torch.linspace(-1, 1, num + k)[:, None]
    r = REV.rbf_bases(centres, grid, k, num)[:, 0]
    assert torch.allclose(torch.diagonal(r), torch.ones(num + k)) and (r > 0).all() and (r <= 1).all()
    c = REV.chebyshev_bases(torch.linspace(-50, 50, 101)[:, None], grid, k, num)[:, 0]
    assert (c.abs() <= 1 + 1e-6).all() and torch.equal(c[:, 0], torch.ones(101))
    # rswaf: sech^2 bumps where the Gaussians are, peak 1 at each centre, in (0, 1]; close to the Gaussians' shape
    w = REV.rswaf_bases(centres, grid, k, num)[:, 0]
    assert torch.allclose(torch.diagonal(w), torch.ones(num + k)) and (w > 0).all() and (w <= 1).all()
    xs = torch.linspace(-1, 1, 200)[:, None]
    assert float((REV.rswaf_bases(xs, grid, k, num) - REV.rbf_bases(xs, grid, k, num)).abs().max()) < 0.25


# ---------------------------------------------------------------------------------------------------------------
# refits in any basis


@pytest.mark.parametrize("device", DEVICES)
def test_refit_coef_bases_matches_refit_coef_for_bsplines(device):
    torch.manual_seed(0)
    x = inputs("dense", n=400, device=device)
    old = REV.extend_grid(torch.linspace(-2, 2, 4)[None].repeat(4, 1).to(device), k_extend=3)
    new = REV.extend_grid(torch.linspace(-2.5, 2.5, 11)[None].repeat(4, 1).to(device), k_extend=3)
    coef = torch.randn(4, 6, 6, device=device)
    a = REV.refit_coef(x, old, coef, 3, new)
    b = REV.refit_coef_bases(REV.B_batch(x, old, 3), REV.B_batch(x, new, 3), coef)
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)
    # blocks of one input: the same solutions
    c = REV.refit_coef_bases(REV.B_batch(x, old, 3), REV.B_batch(x, new, 3), coef, max_elements=400 * 19)
    assert torch.allclose(b, c, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("parent_basis", BASES)
@pytest.mark.parametrize("child_basis", BASES)
def test_refit_across_bases(parent_basis, child_basis):
    # _refit_from: the curves of one basis refitted in another, the least-squares optimum at the samples (how close
    # it comes depends on the bases: B-spline knots at quantiles fit Gaussians at evenly spaced centres slowly)
    torch.manual_seed(0)
    x = inputs("dense", n=1000) * 0.5
    xs = torch.sort(x, 0)[0]
    parent = KANLayer(4, 3, num=5, k=3, basis=parent_basis)
    parent.update_grid_from_samples(x)
    child = KANLayer(4, 3, num=12, k=3, basis=child_basis)
    child.grid.data = REV.extend_grid(REV.zero_aware_grid(x, 12, 3)[:, 3:-3], k_extend=3)
    child.coef.data = child._refit_from(xs, parent.grid, parent.coef, parent.k, parent_basis, child.grid)
    with torch.no_grad():
        err = (edge_curves(parent, xs) - edge_curves(child, xs)).abs().amax(dim=(0, 2))
        mat = basis_values(xs.double(), child.grid.double(), 3, child_basis).permute(1, 0, 2)
        y = torch.einsum("ijk,jlk->ijl", basis_values(xs.double(), parent.grid.double(), 3, parent_basis),
                         parent.coef.double()).permute(1, 0, 2)
        best = (mat @ torch.linalg.lstsq(mat, y, driver="gelsd").solution - y).abs().amax(dim=(1, 2))
    assert (err <= best.float() + 1e-4 + 1e-3 * y.abs().max()).all(), (err, best)
    if child_basis == "chebyshev" and parent_basis == "chebyshev":
        assert err.max() < 1e-5  # nested polynomial spaces: exact (the float64 solve, see refit_coef_bases)


# ---------------------------------------------------------------------------------------------------------------
# KANLayer with each basis


def test_default_layer_unchanged():
    torch.manual_seed(3)
    a = KANLayer(3, 2, num=5, k=3)
    torch.manual_seed(3)
    b = KANLayer(3, 2, num=5, k=3, basis="bspline", dropkan=0., aggregation="sum", bias=False)
    assert a.state_dict().keys() == b.state_dict().keys()
    assert "agg_scale" not in a.state_dict() and "bias" not in a.state_dict()
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    assert a.agg_scale == 1.0 and a.bias is None


@pytest.mark.parametrize("bad", [dict(basis="fourier"), dict(aggregation="max"), dict(dropkan=1.0), dict(dropkan=-0.1)])
def test_layer_options_validated(bad):
    with pytest.raises(ValueError):
        KANLayer(3, 2, **bad)


@pytest.mark.parametrize("basis", BASES)
def test_initial_curves(basis):
    # every basis starts from the same kind of curves: the noise at the interior knots (least squares, exact)
    torch.manual_seed(0)
    layer = KANLayer(3, 2, num=5, k=3, basis=basis, noise_scale=0.5)
    pts = layer.grid[:, 3:-3].permute(1, 0)
    torch.manual_seed(0)
    noises = (torch.rand(6, 3, 2) - 1 / 2) * 0.5 / 5
    assert torch.allclose(edge_curves(layer, pts), noises, atol=1e-5)


def test_initial_spline_scale_fan_in(tmp_path):
    # scale_sp starts at scale_sp / sqrt(in_dim) (pykan 0.2.8, default) or, with scale_sp_fan_in=False, at scale_sp
    # (pykan 0.2.7 and earlier); everything else of the initialisation is the same draw
    torch.manual_seed(0)
    divided = KANLayer(9, 2, scale_sp=0.5)
    torch.manual_seed(0)
    full = KANLayer(9, 2, scale_sp=0.5, scale_sp_fan_in=False)
    assert torch.allclose(divided.scale_sp, torch.full((9, 2), 0.5 / 3))
    assert torch.equal(full.scale_sp, torch.full((9, 2), 0.5))
    assert torch.equal(divided.coef, full.coef) and torch.equal(divided.scale_base, full.scale_base)
    # a network option: every layer, kept by the structural tools and checkpoints
    from xkan import KANNetwork
    from xkan.interp import expand_width
    from xkan.io import load_checkpoint, save_checkpoint
    net = KANNetwork([4, 3, 1], seed=0, scale_sp_fan_in=False)
    assert all(torch.equal(layer.scale_sp, torch.ones_like(layer.scale_sp)) for layer in net.layers)
    wider = expand_width(net, 1, 2)
    assert not wider.config.layer["scale_sp_fan_in"]
    assert torch.equal(wider.layers[0].scale_sp, torch.ones_like(wider.layers[0].scale_sp))
    save_checkpoint(net, tmp_path / "net")
    assert not load_checkpoint(tmp_path / "net").config.layer["scale_sp_fan_in"]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("basis", BASES)
@pytest.mark.parametrize("aggregation", ["sum", "mean"])
@pytest.mark.parametrize("bias", [False, True])
def test_fused_matches_per_edge_forward(device, dtype, basis, aggregation, bias):
    torch.manual_seed(0)
    layer = KANLayer(4, 3, num=5, k=3, basis=basis, aggregation=aggregation, bias=bias).to(device=device, dtype=dtype)
    if bias:
        layer.bias.data = torch.randn(3, device=device, dtype=dtype)
    layer.mask.data[1, 2] = 0.
    x = inputs("dense", device=device, dtype=dtype)
    y, preacts, postacts, postspline = layer.edge_forward(x)
    tol = 1e-5 if dtype == torch.float32 else 1e-12
    assert torch.allclose(layer(x), y, atol=tol, rtol=tol)
    assert torch.equal(layer(x, return_edges=True).postacts, postacts)
    expect = postacts.sum(dim=2) + (layer.bias if bias else 0.)
    assert torch.allclose(y, expect, atol=tol, rtol=tol)
    assert postacts.shape == (x.shape[0], 3, 4) and postspline.shape == (x.shape[0], 3, 4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("basis", BASES)
def test_gradients_finite(device, basis):
    torch.manual_seed(0)
    layer = KANLayer(4, 3, num=5, k=3, basis=basis, bias=True).to(device)
    x = (inputs("zero-inflated", device=device) * 3).requires_grad_()  # inputs outside the grid too
    layer(x).square().sum().backward()
    for p in (layer.coef, layer.scale_base, layer.scale_sp, layer.bias, x):
        assert p.grad is not None and torch.isfinite(p.grad).all()


@pytest.mark.parametrize("basis", BASES)
@pytest.mark.parametrize("placement", ["quantile", "nonzero_quantile", "zero_aware"])
@pytest.mark.parametrize("mode", ["sample", "sample+uniform", "grid"])
@pytest.mark.parametrize("kind", ["dense", "zero-inflated"])
def test_grid_update_keeps_curves(basis, placement, mode, kind):
    torch.manual_seed(0)
    layer = KANLayer(4, 3, num=5, k=3, basis=basis, knot_placement=placement)
    x = inputs(kind, n=2000)
    layer.update_grid_from_samples(x)  # the knots onto the samples, then curves to keep
    layer.coef.data = layer.coef.data + 0.1 * torch.randn_like(layer.coef)
    before = edge_curves(layer, x)
    layer.update_grid_from_samples(x, mode=mode)
    after = edge_curves(layer, x)
    assert torch.isfinite(layer.coef).all() and torch.isfinite(layer.grid).all()
    if mode == "grid" and placement == "quantile" and kind == "zero-inflated" and basis != "bspline":
        # mode 'grid' refits at 2 * num + 1 quantile points, most of them on the zeros: a global basis is
        # underdetermined there (the B-splines' local support hides it); what 'nonzero_quantile' / 'zero_aware' are for
        pytest.xfail("mode 'grid' with quantile knots on zero-inflated inputs: too few distinct refit points")
    assert (after - before).abs().max() < 0.05 * before.abs().max() + 1e-4
    if placement == "zero_aware":
        assert torch.equal(layer.grid, REV.zero_aware_grid(x, 5, 3, layer.grid_eps))


# ---------------------------------------------------------------------------------------------------------------
# DropKAN, aggregation, bias


def test_dropkan_layer():
    torch.manual_seed(0)
    layer = KANLayer(6, 4, num=5, k=3, dropkan=0.25)
    plain = KANLayer(6, 4, num=5, k=3)
    plain.load_state_dict(layer.state_dict())
    x = inputs("dense", n=4000, d=6)
    layer.eval()
    assert torch.equal(layer.edge_forward(x).y, plain.edge_forward(x).y) and torch.equal(layer(x), plain(x))
    layer.train()
    ref = plain.edge_forward(x).postacts
    postacts = layer.edge_forward(x).postacts
    kept = postacts != 0
    assert abs(1 - kept.float().mean().item() - 0.25) < 0.01  # each edge of each sample dropped with probability p
    assert torch.allclose(postacts[kept], ref[kept] / 0.75, atol=1e-6)  # the kept ones scaled by 1 / (1 - p)
    mean = torch.stack([layer(x[:200]) for _ in range(400)]).mean(0)
    assert (mean - plain(x[:200])).abs().max() < 0.1 * plain(x[:200]).abs().max()  # unbiased
    g = torch.Generator().manual_seed(5)
    layer._dropkan_generator = g
    a = layer(x)  # in training mode with DropKAN: the per-edge path
    g.manual_seed(5)
    assert torch.equal(layer.edge_forward(x).y, a)  # the masks from the generator when set


@pytest.mark.parametrize("device", DEVICES)
def test_mean_factor_follows_device_and_dtype(device):
    m = KANLayer(4, 2, aggregation="mean").to(device=device, dtype=torch.float64)
    assert m.agg_scale.device.type == device and m.agg_scale.dtype == torch.float64
    y = m(inputs("dense", device=device, dtype=torch.float64))
    assert y.dtype == torch.float64


# ---------------------------------------------------------------------------------------------------------------
# MultKAN with each basis and option


@pytest.mark.parametrize("placement", ["quantile", "nonzero_quantile", "zero_aware"])
@pytest.mark.parametrize("mode", ["sample", "sample+uniform", "grid"])
def test_grid_update_grid_eps_for_one_update(placement, mode):
    # update_grid_from_samples(grid_eps=...): this update's blend of evenly spaced and quantile knots, the same as a
    # layer built with it; the layer's own grid_eps unchanged
    x = inputs("zero-inflated", n=800)
    torch.manual_seed(0)
    a = KANLayer(4, 3, num=5, k=3, knot_placement=placement, grid_eps=0.02)
    torch.manual_seed(0)
    b = KANLayer(4, 3, num=5, k=3, knot_placement=placement, grid_eps=0.5)
    torch.manual_seed(0)
    c = KANLayer(4, 3, num=5, k=3, knot_placement=placement, grid_eps=0.02)
    a.update_grid_from_samples(x, mode=mode, grid_eps=0.5)
    b.update_grid_from_samples(x, mode=mode)
    c.update_grid_from_samples(x, mode=mode, grid_eps=None)
    assert torch.equal(a.grid, b.grid) and torch.equal(a.coef, b.coef) and a.grid_eps == 0.02
    assert not torch.equal(a.grid, c.grid)  # None: the layer's 0.02


def test_grid_update_unknown_mode():
    layer = KANLayer(3, 2)
    with pytest.raises(ValueError, match="unknown mode"):
        layer.update_grid_from_samples(torch.randn(50, 3), mode="uniform")  # pykan 0.2.8: silently 'sample'
