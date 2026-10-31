"""The LBFGS optimizer and failed adaptive grid updates."""
import pytest
import torch


from xkan.layers import AdaptiveKANLayer
from xkan.train.lbfgs import LBFGS


@pytest.mark.parametrize("line_search", [None, "strong_wolfe"])
def test_lbfgs_solves_a_quadratic(line_search):
    torch.manual_seed(0)
    a = torch.randn(6, 6)
    q = a @ a.T + torch.eye(6)
    b = torch.randn(6)
    w = torch.zeros(6, requires_grad=True)
    opt = LBFGS([w], lr=1 if line_search else 0.1, max_iter=50, history_size=10, line_search_fn=line_search)

    def closure():
        opt.zero_grad()
        loss = 0.5 * w @ q @ w - b @ w
        loss.backward()
        return loss
    for _ in range(20):
        opt.step(closure)
    assert torch.allclose(w.detach(), torch.linalg.solve(q, b), atol=1e-3)


def test_failed_adaptive_update_keeps_the_layer(monkeypatch):
    torch.manual_seed(0)
    layer = AdaptiveKANLayer(in_dim=2, out_dim=2, min_rows=8, update_at=(1,))
    grid, coef = layer.grid.clone(), layer.coef.clone()

    def fail(*args, **kwargs):
        layer.grid.data += 1.0  # the grid moved before the refit failed
        raise RuntimeError("simulated refit failure")
    monkeypatch.setattr(layer, "update_grid_from_samples", fail)
    layer.train()
    layer(torch.randn(16, 2))
    assert layer.failed_updates == 1 and layer.n_updates == 0
    assert torch.equal(layer.grid, grid) and torch.equal(layer.coef, coef)


