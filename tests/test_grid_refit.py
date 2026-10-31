"""Grid refits of a layer: zero-inflated (tied) samples, refit_guard, refit mode 'sample+uniform'."""
import pytest
import torch


from xkan.layers import KANLayer
from xkan.spline import B_batch, coef2curve, rank_cutoff

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def refit(nonzero=3, mode="sample", guard=None, device="cpu", k=2, num=3, seed=0, in_dim=50):
    """one layer grid refit; inputs zero-inflated (nonzero of 32 samples non-zero) or Gaussian (nonzero = 0, 256
    samples); the largest coefficient before / after and the largest change of the curves over the samples' range and
    at the samples"""
    torch.manual_seed(seed)
    layer = KANLayer(in_dim=in_dim, out_dim=4, num=num, k=k, device=device, refit_guard=guard)
    if nonzero:
        x = torch.zeros(32, in_dim)
        x[:nonzero] = torch.rand(nonzero, in_dim) * 3
    else:
        x = torch.randn(256, in_dim)
    x = x.to(device)
    lo, hi = x.min(0).values, x.max(0).values
    dense = lo + (hi - lo) * torch.linspace(0, 1, 200, device=device)[:, None]
    y0, ys0 = coef2curve(dense, layer.grid, layer.coef, layer.k), coef2curve(x, layer.grid, layer.coef, layer.k)
    c0 = float(layer.coef.abs().max())
    old_grid, old_coef = layer.grid.data.clone(), layer.coef.data.clone()
    layer.update_grid_from_samples(x, mode=mode)
    y1, ys1 = coef2curve(dense, layer.grid, layer.coef, layer.k), coef2curve(x, layer.grid, layer.coef, layer.k)
    # the exact refit at the samples: the minimum-norm least-squares solution in float64 at the solver's cutoff
    mat = B_batch(x, layer.grid, layer.k).permute(1, 0, 2)
    target = coef2curve(x, old_grid, old_coef, layer.k).permute(1, 0, 2)
    best = (torch.linalg.pinv(mat.double(), rtol=rank_cutoff(mat)) @ target.double()).permute(0, 2, 1).to(x.dtype)
    ye = coef2curve(x, layer.grid, best, layer.k)
    return {"coef_before": c0, "coef_after": float(layer.coef.abs().max()), "finite": bool(torch.isfinite(layer.coef).all()),
            "curve_change_range": float((y1 - y0).abs().max()), "curve_change_samples": float((ys1 - ys0).abs().max()),
            "exact_change_samples": float((ye - ys0).abs().max())}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("nonzero", [3, 6, 0])
def test_refit_is_finite_and_the_least_squares_fit_at_the_samples(device, nonzero):
    # tied samples make the refit rank-deficient (pykan 0.2.8: NaN or 1e20 on CUDA, a misjudged rank with some gelsy
    # builds): the refit is the exact minimum-norm one
    r = refit(nonzero=nonzero, device=device)
    assert r["finite"] and r["coef_after"] < 1e3
    assert abs(r["curve_change_samples"] - r["exact_change_samples"]) <= 1e-4 * max(1.0, r["exact_change_samples"]), r


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("nonzero", [3, 6])
def test_tied_refit_cuda_matches_cpu(nonzero):
    a, b = refit(nonzero=nonzero, device="cpu"), refit(nonzero=nonzero, device="cuda")
    assert all(abs(a[key] - b[key]) <= 1e-4 * max(1.0, abs(a[key])) for key in ("coef_after", "curve_change_range"))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("nonzero", [3, 6])
def test_sample_uniform_keeps_tied_refit_close_between_samples(device, nonzero):
    sample = refit(nonzero=nonzero, device=device)
    both = refit(nonzero=nonzero, device=device, mode="sample+uniform")
    assert both["finite"]
    assert both["curve_change_range"] < 0.5 * sample["curve_change_range"], (sample, both)


@pytest.mark.parametrize("device", DEVICES)
def test_guard_caps_growth_and_is_inert_on_smooth_inputs(device):
    r = refit(nonzero=6, device=device, guard=10)
    assert r["coef_after"] <= 10 * max(r["coef_before"], 1.0) + 1e-6
    assert refit(nonzero=0, device=device) == refit(nonzero=0, device=device, guard=10)


