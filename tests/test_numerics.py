"""B-spline bases at coincident knots, refits in blocks, knot placement options, the tied-sample refit."""
from pathlib import Path
import pytest
import torch

import xkan

from xkan.layers import KANLayer
from xkan.layers import AdaptiveKANLayer
from xkan.symbolic.fitting import fit_params, fit_params_batched
from xkan.symbolic.library import SYMBOLIC_LIB

# the checkout (or installation) under test: the folder holding the xkan package, for the tests that start a fresh
# Python process
REV_DIR = Path(xkan.__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


# ---------------------------------------------------------------- J, M


# ---------------------------------------------------------------- H, L
@pytest.mark.parametrize("device", DEVICES)
def test_min_knot_gap(device):
    torch.manual_seed(0)
    x = torch.randn(300, 3, device=device)
    x[:, 1] = 0.7  # constant input
    x[:250, 2] = 0.0  # mostly tied
    plain = KANLayer(in_dim=3, out_dim=2, num=5, k=3, device=device)
    gapped = KANLayer(in_dim=3, out_dim=2, num=5, k=3, device=device, min_knot_gap=1e-3)
    gapped.load_state_dict(plain.state_dict())
    plain.update_grid_from_samples(x)
    gapped.update_grid_from_samples(x)
    d = torch.diff(gapped.grid, dim=1)
    assert float(d[1].min()) >= 1e-4 * 0.999 and float(d[2].min()) > 0
    assert float(torch.diff(plain.grid, dim=1)[1].min()) == 0.0  # without the gap: all knots on one point
    assert torch.equal(plain.grid[0], gapped.grid[0])  # a spread input keeps its knots
    assert torch.isfinite(gapped.coef).all()


@pytest.mark.parametrize("device", DEVICES)
def test_nonzero_quantile_knots(device):
    torch.manual_seed(0)
    n = 1000
    x = torch.randn(n, 3, device=device)
    nz = torch.rand(n, device=device) < 0.1
    x[:, 1] = torch.where(nz, torch.rand(n, device=device) * 3 + 0.5, torch.zeros(n, device=device))  # sparse, >= 0
    layer = KANLayer(in_dim=3, out_dim=2, num=4, k=2, device=device, knot_placement="nonzero_quantile", grid_eps=0.0)
    base = KANLayer(in_dim=3, out_dim=2, num=4, k=2, device=device, grid_eps=0.0)
    base.load_state_dict(layer.state_dict())
    layer.update_grid_from_samples(x)
    base.update_grid_from_samples(x)
    interior = layer.grid[1, 2:-2]
    vals = torch.sort(x[nz, 1])[0]
    expect = torch.stack([vals[int(round(q * (len(vals) - 1)))] for q in (0, 1 / 3, 2 / 3, 1)])
    assert float(interior[0]) == 0.0 and torch.allclose(interior[1:], expect)
    assert torch.equal(layer.grid[0], base.grid[0]) and torch.equal(layer.grid[2], base.grid[2])  # other inputs
    assert int((torch.diff(base.grid[1, 2:-2]) == 0).sum()) > 0  # quantiles of all samples pile knots on 0


# ---------------------------------------------------------------- N
def test_adaptive_layer_schedule():
    torch.manual_seed(0)
    layer = AdaptiveKANLayer(in_dim=3, out_dim=2, num=4, k=2, update_at=(1, 5, 9), min_rows=100)
    grid0 = layer.grid.clone()
    for call in range(1, 4):  # 30 rows per call: the update due at call 1 waits for 100 rows (call 4)
        layer(torch.randn(30, 3) * 3)
        assert layer.n_updates == 0 and torch.equal(layer.grid, grid0), call
    layer(torch.randn(30, 3) * 3)
    assert layer.n_updates == 1 and not torch.equal(layer.grid, grid0)
    for _ in range(5):
        layer(torch.randn(30, 3) * 3)  # calls 5..9: updates at 5 and 9
    assert layer.n_updates == 3
    g = layer.grid.clone()
    for _ in range(10):
        layer(torch.randn(30, 3) * 3)
    assert layer.n_updates == 3 and torch.equal(layer.grid, g) and layer._buffer == []
    layer.eval()
    layer(torch.randn(30, 3))
    assert layer.n_calls == 9


def test_adaptive_layer_uses_recent_rows_and_no_eval_updates():
    layer = AdaptiveKANLayer(in_dim=1, out_dim=1, num=4, k=2, update_at=(3,), min_rows=64)
    layer.eval()
    for _ in range(5):
        layer(torch.randn(64, 1))
    assert layer.n_calls == 0 and layer.n_updates == 0
    layer.train()
    layer(torch.full((64, 1), 100.0))
    layer(torch.full((64, 1), -1.0))
    layer(torch.linspace(0, 1, 64)[:, None])  # call 3: the update uses the most recent 64 rows only
    assert layer.n_updates == 1
    assert abs(float(layer.grid[0, 2]) - 0.0) < 0.05 and abs(float(layer.grid[0, -3]) - 1.0) < 0.05


# ---------------------------------------------------------------- K
@pytest.mark.parametrize("device", DEVICES)
def test_fit_params_batched_matches(device):
    torch.manual_seed(0)
    x = torch.rand(6, 200, device=device) * 2 - 1
    y = torch.stack([5.0 * torch.sin(3.0 * x[0] + 2.0) + 0.7, torch.exp(x[1]), x[2] ** 2 - 1, torch.tanh(4 * x[3]),
                     torch.randn(200, device=device), torch.zeros(200, device=device) + 0.3])
    for name in ("sin", "exp", "x^2", "tanh", "x", "gaussian"):
        fun = SYMBOLIC_LIB[name][0]
        p_b, r2_b = fit_params_batched(x, y, fun)
        for e in range(6):
            p_s, r2_s = fit_params(x[e], y[e], fun, verbose=False, device=device)
            assert abs(float(r2_b[e]) - float(r2_s)) < 1e-4, (name, e, float(r2_b[e]), float(r2_s))
            # the fitted curves agree; a and b need not (a flat optimum lands on a neighbouring grid point when summed in
            # another order, exp has no identified b, sin's (a, b) and (-a, -b) differ by c's sign)
            curve = lambda p: p[2] * torch.nan_to_num(fun(p[0] * x[e] + p[1])) + p[3]
            scale = float(y[e].abs().max()) + 1e-6
            assert float((curve(p_b[e]) - curve(p_s.to(p_b.dtype))).abs().max()) <= 2e-3 * scale + 1e-5, (name, e)


