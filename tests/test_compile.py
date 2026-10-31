"""torch.compile of the layer: the fused and the per-edge forward compile to one graph (fullgraph=True), with the eager
outputs and gradients."""
import pytest
import torch

from xkan.layers import BASES, KANLayer

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def layer(basis="bspline", device="cpu", **kw):
    torch.manual_seed(0)
    L = KANLayer(6, 4, num=5, k=3, basis=basis, device=device, **kw)
    x = (torch.rand(50, 6) * 2 - 1).to(device)
    return L, x


def forward_backward(fn, L, x):
    L.zero_grad()
    xg = x.clone().requires_grad_(True)
    y = fn(xg)
    y.square().sum().backward()
    return y.detach(), xg.grad, [p.grad.clone() for p in L.parameters() if p.grad is not None]


def assert_same(eager, compiled, tol):
    (y0, gx0, gp0), (y1, gx1, gp1) = eager, compiled
    assert torch.allclose(y0, y1, atol=tol, rtol=tol)
    assert torch.allclose(gx0, gx1, atol=tol, rtol=tol)
    assert len(gp0) == len(gp1) and all(torch.allclose(a, b, atol=tol, rtol=tol) for a, b in zip(gp0, gp1))


@pytest.mark.parametrize("basis", sorted(BASES))
@pytest.mark.parametrize("edges", [False, True])
def test_layer_compiles_to_one_graph(basis, edges):
    L, x = layer(basis, bias=True, aggregation="mean")
    fn = (lambda v: L(v, return_edges=True).postacts.sum(dim=2)) if edges else L
    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="aot_eager", fullgraph=True)
    assert_same(forward_backward(fn, L, x), forward_backward(compiled, L, x), 1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_layer_compiles_with_inductor(device):
    L, x = layer(device=device)
    torch._dynamo.reset()
    compiled = torch.compile(L, fullgraph=True)
    assert_same(forward_backward(L, L, x), forward_backward(compiled, L, x), 1e-5)
    # a grid update between calls: the compiled layer follows the new grid and coefficients
    L.update_grid_from_samples(x * 0.5)
    assert_same(forward_backward(L, L, x), forward_backward(compiled, L, x), 1e-5)
