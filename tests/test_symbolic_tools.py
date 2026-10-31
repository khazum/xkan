"""Symbolic tools on KANNetworks: an edge whose curve is a known function is recovered by fix_symbolic and
suggest_symbolic (the network unchanged by suggestions); auto_symbolic's sequential, batched and network-selection
passes agree, keep the network close to its function and honour the zero gate and the r2 threshold; formulas of
networks, with options, and of multi-head models; libraries."""
import copy

import numpy as np
import pytest
import sympy
import torch

from xkan import KANNetwork, MultiHead, fit
from xkan.spline import curve2coef
from xkan.symbolic import (SymbolicLibrary, auto_symbolic, default_library, fix_symbolic, suggest_symbolic,
                          symbolic_formula, unfix_symbolic)

LIB = ["x", "x^2", "sin", "exp", "tanh", "0", "1/x", "sqrt", "abs"]


def edge_with_curve(f, n=400):
    """a [1, 1] network whose only edge is the curve f on [-1, 1] (spline fitted to it, no base function)"""
    torch.manual_seed(0)
    net = KANNetwork([1, 1], grid=20, k=3, seed=1, grid_range=(-1, 1))
    x = torch.linspace(-1, 1, n)[:, None]
    layer = net.layers[0]
    with torch.no_grad():
        layer.scale_base.zero_()
        layer.scale_sp.fill_(1.)
        layer.coef.copy_(curve2coef(x, f(x)[:, :, None], layer.grid, layer.k))
    return net, x


def trained(width=(2, 3, 1), steps=60):
    torch.manual_seed(0)
    x = torch.rand(400, width[0]) * 2 - 1
    y = torch.sin(torch.pi * x[:, :1]) + x[:, 1:2] ** 2
    net = KANNetwork(list(width), grid=5, k=3, seed=1)
    fit(net, {"x": x, "y": y}, steps=steps, opt="adam", lr=2e-2, lamb=1e-3)
    return net, x, y


def test_fix_symbolic_recovers_a_known_curve():
    f = lambda t: 0.7 * torch.sin(1.3 * t + 0.1) - 0.2  # noqa: E731
    net, x = edge_with_curve(f)
    r2 = fix_symbolic(net, 0, 0, 0, "sin", x)
    assert float(r2) > 0.9999
    a, b, c, d = net.symbolic[0].affine[0, 0].tolist()
    assert torch.allclose(c * torch.sin(a * x + b) + d, f(x), atol=2e-3)  # (the parameters up to sin's symmetries)
    assert net.layers[0].mask[0, 0] == 0 and net.symbolic[0].mask[0, 0] == 1 and net.symbolic[0].names[0][0] == "sin"
    assert torch.allclose(net(x), f(x), atol=2e-3)
    unfix_symbolic(net, 0, 0, 0)
    assert net.layers[0].mask[0, 0] == 1 and not net.symbolic[0].active and net.symbolic[0].names[0][0] == "0"
    fix_symbolic(net, 0, 0, 0, torch.cos, fit=False)
    assert net.symbolic[0].names[0][0] == "anonymous" and net.symbolic[0].affine[0, 0].tolist() == [1., 0., 1., 0.]


@pytest.mark.parametrize("standardize", [False, True])
def test_suggest_symbolic_ranks_the_true_function_first(standardize):
    net, x = edge_with_curve(lambda t: 1.5 * t ** 2 - 0.3)
    before = copy.deepcopy(net.state_dict())
    name, f, r2, c = suggest_symbolic(net, 0, 0, 0, x, lib=LIB, verbose=False, input_standardize=standardize)
    assert name == "x^2" and r2 > 0.9999 and c == default_library()["x^2"].complexity
    assert all(torch.equal(v, net.state_dict()[k]) for k, v in before.items())
    with torch.no_grad():
        net.layers[0].mask.zero_()  # a removed edge: every candidate gives the zero function
    assert suggest_symbolic(net, 0, 0, 0, x, lib=["x", "0", "sin"], verbose=False)[0] == "0"


@pytest.mark.parametrize("kw", [dict(), dict(input_standardize=True), dict(a_range_relative=False),
                                dict(a_range_relative=True)])
def test_sequential_and_batched_passes_agree(kw):
    net, x, _ = trained()
    a, b = copy.deepcopy(net), copy.deepcopy(net)
    fa = auto_symbolic(a, x, lib=LIB, verbose=0, **kw)
    fb = auto_symbolic(b, x, lib=LIB, verbose=0, batched=True, **kw)
    assert {e: v[0] for e, v in fa.items()} == {e: v[0] for e, v in fb.items()}
    assert len(fa) == sum(net.width_in[l] * net.width_out[l + 1] for l in range(net.depth))
    # the same edges (their parameters may differ where they are not identifiable: a against c for 'x', a and b of '0')
    assert [la.names for la in a.symbolic] == [lb.names for lb in b.symbolic]
    # (the sequential and the batched grid searches end within their resolution of each other)
    assert float((a(x) - b(x)).abs().max()) < 5e-3 * float(net(x).std())


def test_auto_symbolic_keeps_the_function_and_its_formula():
    net, x, y = trained()
    ref = net(x).detach()
    sym = copy.deepcopy(net)
    auto_symbolic(sym, x, lib=LIB, verbose=0, batched=True, selection="network")
    assert not any(layer.mask.any() for layer in sym.layers) and all(s.active for s in sym.symbolic)
    out = sym(x).detach()
    assert float(torch.sqrt(torch.mean((out - ref) ** 2))) < 0.2 * float(ref.std())
    formula, variables = symbolic_formula(sym)
    f = sympy.lambdify(variables, formula[0])
    assert np.allclose(f(*x.double().numpy().T), out[:, 0].double().numpy(), rtol=1e-3, atol=1e-3)


def test_zero_gate_and_r2_threshold():
    net, x, _ = trained()
    gated, ungated = copy.deepcopy(net), copy.deepcopy(net)
    from xkan.interp import attribute
    scores = attribute(net, x, per_output=False).edge
    auto_symbolic(gated, x, lib=LIB, verbose=0, batched=True, zero_gate=True, zero_gate_threshold=1e-3)
    for l in range(net.depth):
        for j in range(net.width_out[l + 1]):
            for i in range(net.width_in[l]):
                if scores[l][j, i] >= 1e-3:
                    assert gated.symbolic[l].names[j][i] != "0"
    fixed = auto_symbolic(ungated, x, lib=["x", "0"], verbose=0, r2_threshold=2.)  # no fit reaches r2 = 2
    assert not fixed and all(layer.mask.all() for layer in ungated.layers)


def same_expr(e, f, variables, n=50):
    g = sympy.lambdify(variables, e - f)
    pts = np.random.default_rng(0).uniform(-1, 1, size=(len(variables), n))
    return np.allclose(g(*pts), 0, atol=1e-9)


def test_formula_options_and_multihead():
    net = KANNetwork([2, 1], seed=1)
    fix_symbolic(net, 0, 0, 0, "x", fit=False)
    fix_symbolic(net, 0, 1, 0, "x^2", fit=False)
    with torch.no_grad():
        net.symbolic[0].affine[0, 1] = torch.tensor([2., 1., 3., 0.5])  # 3 (2 x_2 + 1)^2 + 0.5
    x1, x2 = sympy.symbols("x_1 x_2")
    target = x1 + 3 * (2 * x2 + 1) ** 2 + 0.5
    (f,), variables = symbolic_formula(net)
    assert variables == [x1, x2] and same_expr(f, target, variables)
    a, b = sympy.symbols("a b")
    assert same_expr(symbolic_formula(net, var=["a", "b"])[0][0], target.subs({x1: a, x2: b}), [a, b])
    (g,), _ = symbolic_formula(net, normalizer=([0.1, 0.2], [2., 3.]))
    assert same_expr(g, target.subs({x1: (x1 - 0.1) / 2, x2: (x2 - 0.2) / 3}), variables)
    (h,), _ = symbolic_formula(net, output_normalizer=([1.], [2.]))
    assert same_expr(h, 2 * target + 1, variables)
    fix_symbolic(net, 0, 1, 0, "sqrt", fit=False)
    (r,), _ = symbolic_formula(net, real=True)
    assert r.has(sympy.Abs)
    # a multi-head model: the trunk's formulas are the heads' inputs
    trunk, head = KANNetwork([2, 2], seed=1), KANNetwork([2, 1], seed=2)
    for m in (trunk, head):
        for i in range(m.width_in[0]):
            for j in range(m.width_out[1]):
                fix_symbolic(m, 0, i, j, "x", fit=False)
    formulas, (y1, y2) = symbolic_formula(MultiHead(trunk, {"y": head}))
    assert same_expr(formulas["y"][0], 2 * (y1 + y2), [y1, y2])


def test_library():
    lib = default_library()
    assert {"x", "sin", "0", "gaussian"} <= set(lib) and lib["sin"].complexity == 2
    sub = lib.subset(["x", "sin"])
    assert list(sub) == ["x", "sin"] and len(sub) == 2
    lib.add("cube_root", lambda t: torch.sign(t) * t.abs() ** (1 / 3), lambda t: sympy.cbrt(t), 3)
    assert lib["cube_root"].fn(torch.tensor([8.])).item() == pytest.approx(2.)
    assert default_library().resolve(["x"]).__class__ is SymbolicLibrary
    with pytest.raises(KeyError):
        lib["nonsense"]


def test_a_range_relative_modes():
    # True divides a_range by the largest |x| of each edge; 'widen' only when that is below 1; False keeps a_range
    from xkan.symbolic.fitting import _symbolic_fit_inputs
    x = torch.tensor([[0.1, -0.25], [2., -4.]])  # edges with largest |x| 0.25 and 4
    for mode, lo in ((True, [-40., -2.5]), ("widen", [-40., -10.]), (False, -10.)):
        _, (a0, a1), _, _ = _symbolic_fit_inputs(x, (-10., 10.), False, mode)
        assert torch.allclose(torch.as_tensor(a0, dtype=torch.float32), torch.tensor(lo)) and torch.allclose(
            torch.as_tensor(a1), -torch.as_tensor(a0))
    _, rng, _, _ = _symbolic_fit_inputs(x[1], (-10., 10.), False, "widen")
    assert rng == (-10., 10.)  # one edge: Python floats, as before
    with pytest.raises(ValueError, match="widen"):
        _symbolic_fit_inputs(x, (-10., 10.), False, "narrow")


def test_blocked_grid_search_agrees_with_the_whole_grid():
    # the grid search in cache-sized blocks of rows of a (SWEEP_BLOCK_ELEMENTS) against the whole grid at once: the same
    # fits up to the rounding of the sums. The test compares the fitted curves c f(a x + b) + d and their r2, since
    # c (a x + b)^2 + d gives one curve for every (k a, k b, c / k^2): which of those grid cells wins depends on the
    # rounding, and so on the number of threads.
    import xkan.symbolic.fitting as F
    torch.manual_seed(0)
    x = torch.rand(400) * 2 - 1
    y = torch.sin(2 * x + 0.3) + 0.05 * torch.randn(400)
    X, Y = torch.rand(5, 300) * 2 - 1, torch.cos(3 * torch.rand(5, 300))
    fns = (torch.sin, torch.exp, lambda t: t ** 2)

    def curve(p, fn, inputs):
        return p[..., 2:3] * fn(p[..., 0:1] * inputs + p[..., 1:2]) + p[..., 3:4]

    saved = F.SWEEP_BLOCK_ELEMENTS
    try:
        out = {}
        for block in (None, 2**14):
            F.SWEEP_BLOCK_ELEMENTS = block
            out[block] = [F.fit_params(x, y, fn, verbose=False) for fn in fns]
            out[block].append(F.fit_params_batched(X, Y, torch.cos))
        cases = [(fn, x) for fn in fns] + [(torch.cos, X)]
        for (fn, inputs), (pa, ra), (pb, rb) in zip(cases, out[None], out[2**14]):
            assert torch.allclose(ra, rb, atol=1e-6)
            assert torch.allclose(curve(pa, fn, inputs), curve(pb, fn, inputs), rtol=1e-4, atol=1e-4)
    finally:
        F.SWEEP_BLOCK_ELEMENTS = saved
