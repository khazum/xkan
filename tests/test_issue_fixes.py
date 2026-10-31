"""Fixes of computational problems reported in the pykan GitHub issues (KindXiaoming/pykan, #<number>), and of
problems found while fixing them, at the level of layers, solvers, symbolic fits and functions, formulas and the
library (the networks' and tools': test_behaviour.py)."""
import copy
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
import pytest
import torch

import xkan

from xkan.layers import KANLayer
from xkan import spline as REV

# the checkout (or installation) under test: the folder holding the xkan package, for the tests that start a fresh
# Python process
REV_DIR = Path(xkan.__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def run_py(code, threads=1):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), TQDM_DISABLE="1")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    line = [l for l in r.stdout.splitlines() if l.startswith("OUT ")]
    assert line, r.stderr[-3000:]
    return json.loads(line[0][4:])


# ---------------------------------------------------------------- R2


def test_get_subset_keeps_options():
    layer = KANLayer(3, 2, num=5, k=3, grid_eps=0.3, sp_trainable=False, refit_guard=10., knot_placement="nonzero_quantile",
                     min_knot_gap=0.02)
    sub = layer.get_subset([0, 2], [1])
    assert (sub.grid_eps, sub.refit_guard, sub.knot_placement, sub.min_knot_gap) == (0.3, 10., "nonzero_quantile", 0.02)
    assert not sub.scale_sp.requires_grad and sub.scale_base.requires_grad
    assert torch.equal(sub.coef, layer.coef[[0, 2]][:, [1]])


# ---------------------------------------------------------------- R3


# ---------------------------------------------------------------- R4
def test_float64_work_on_cpu_for_other_devices():
    assert REV._float64_device(torch.empty(1, device="meta")) == torch.device("cpu")
    assert REV._float64_device(torch.empty(1)) == torch.device("cpu")
    if torch.cuda.is_available():
        t = torch.empty(1, device="cuda")
        assert REV._float64_device(t) == t.device


@pytest.mark.parametrize("device", DEVICES)
def test_singular_check_returns_on_input_device(device):
    mat = torch.ones(2, 6, 3, device=device)
    out = REV.numerically_singular(mat)
    assert out.device == mat.device and out.tolist() == [True, True]


# ---------------------------------------------------------------- U1


@pytest.mark.parametrize("device", DEVICES)
def test_layer_non_finite_column_others_unchanged(device):
    torch.manual_seed(0)
    layer = KANLayer(3, 2, num=5, k=3).to(device)
    ref = copy.deepcopy(layer)
    g0, c0 = layer.grid.clone(), layer.coef.clone()
    x = torch.randn(200, 3, device=device)
    xn = x.clone()
    xn[5, 1] = float("nan")
    xn[7, 1] = float("inf")
    layer.update_grid_from_samples(xn)
    ref.update_grid_from_samples(x)
    assert torch.equal(layer.grid[1], g0[1]) and torch.equal(layer.coef[1], c0[1])
    assert torch.allclose(layer.grid[[0, 2]], ref.grid[[0, 2]])
    assert torch.allclose(layer.coef[[0, 2]], ref.coef[[0, 2]], atol=1e-6)


# ---------------------------------------------------------------- U7: the input map of prune_input


# ---------------------------------------------------------------- U15 and the symbolic copy in refine


# ---------------------------------------------------------------- state_id


# ================================================================ group 2: device, dtype, memory (U2-U6)
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


# ---------------------------------------------------------------- U2


# ---------------------------------------------------------------- U3


# ---------------------------------------------------------------- U4


# ---------------------------------------------------------------- U5


# ---------------------------------------------------------------- U6
LOAD_CODE = """
import sys, json, warnings, torch
sys.path.insert(0, {repo!r})

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    m = KAN.loadckpt({path!r}{extra})
x = torch.load({xpath!r}, map_location='cpu')
print('OUT ' + json.dumps({{'device': str(m.device), 'out': m(x.to(m.device)).cpu().tolist(), 'dtype': str(m.dtype),
                            'warned': any('CUDA unavailable' in str(i.message) for i in w)}}))
"""


# ================================================================ group 3: checkpoints, copy, auto_save (U7, U8)


# ================================================================ group 4: attribution, regularisation, plot (U10-U12)


# ================================================================ group 5: fit, data, API (U13, U14, U16)


def test_batch_jacobian_unknown_mode():
    from xkan.utils import batch_jacobian
    with pytest.raises(ValueError, match="unknown mode"):
        batch_jacobian(lambda x: x ** 2, torch.rand(5, 2), mode="matrix")  # 0.2.8: None


def test_feynman_jackson_2_11_formula():
    import sympy
    from xkan.contrib.feynman import get_feynman_dataset
    symbol, expr, f, ranges = get_feynman_dataset("Jackson 2.11")
    torch.manual_seed(0)
    x = torch.rand(50, 5, dtype=torch.float64) + 0.5
    ref = sympy.lambdify(symbol, expr, "numpy")(*[x[:, i].numpy() for i in range(5)])
    assert torch.allclose(f(x)[:, 0], torch.as_tensor(ref), rtol=1e-6)  # 0.2.8: indexed columns by values (tpi is float32)


def test_avoiding_functions_return_params_and_value():
    from xkan.symbolic.library import SYMBOLIC_LIB
    x = torch.linspace(-3, 3, 7)[:, None]
    for name, entry in SYMBOLIC_LIB.items():
        out = entry[3](x, torch.tensor(10.))
        assert isinstance(out, tuple) and len(out) == 2, name  # 0.2.8: x^1.5 a tensor, tan three elements
        assert out[1].shape == x.shape, name


def test_avoiding_functions_finite_gradients():
    from xkan.symbolic import library as REVU
    y_th = torch.tensor(10.)
    for name, pts in (("f_inv", [0.]), ("f_inv2", [0.]), ("f_inv3", [0.]), ("f_inv4", [0.]), ("f_inv5", [0.]),
                      ("f_sqrt", [0.]), ("f_invsqrt", [0.]), ("f_log", [0.]), ("f_arctanh", [1., -1.]),
                      ("f_exp", [50., 89., 100.])):
        x = torch.tensor(pts)[:, None].requires_grad_(True)
        v = getattr(REVU, name)(x, y_th)[1]
        v.sum().backward()
        assert torch.isfinite(v).all() and torch.isfinite(x.grad).all(), name  # 0.2.8: NaN


def fully_symbolic(width, mult_arity):
    from xkan import KANNetwork
    from xkan.symbolic import fix_symbolic
    torch.manual_seed(0)
    m = KANNetwork(width, mult_arity=mult_arity, grid=3, k=3, seed=0)
    for l in range(m.depth):
        for i in range(m.width_in[l]):
            for j in range(m.width_out[l + 1]):
                fix_symbolic(m, l, i, j, "sin" if (i + j) % 2 else "x", fit=False, random=True)
    return m


def formula_error(m, n_in):
    import sympy
    x = torch.rand(40, n_in) * 2 - 1
    net = m(x)[:, 0].detach().double()
    from xkan.symbolic import symbolic_formula
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expr, xs = symbolic_formula(m)
    f = sympy.lambdify(xs, expr[0], "numpy")
    val = torch.as_tensor(f(*[x[:, i].double().numpy() for i in range(n_in)]), dtype=torch.float64)
    return float((val - net).abs().max())


@pytest.mark.parametrize("width,arity", [([3, [1, 2], 1], 3), ([5, [1, 2], 1], 4), ([3, [1, 2], 1], [[], [3, 2], []])])
def test_formula_mult_arity(width, arity):
    assert formula_error(fully_symbolic(width, arity), width[0]) < 1e-4  # 0.2.8: 2e-4 .. 1.4 (#404)


# ================================================================ group 7: the author's decisions (D1-D17)


# ---------------------------------------------------------------- D1


# ---------------------------------------------------------------- D2, D3, D5


def test_d5_import_does_not_reseed():
    code = """
import sys, json, torch
torch.manual_seed(5); a = torch.rand(3).tolist()
torch.manual_seed(5)
sys.path.insert(0, {repo!r})
import xkan
b = torch.rand(3).tolist()
print('OUT ' + json.dumps({{'same': a == b}}))
"""
    assert run_py(code.format(repo=str(REV_DIR)))["same"]  # 0.2.8: kan/MLP.py seeded torch at import


# ---------------------------------------------------------------- D4


# ---------------------------------------------------------------- D7, D8


# ---------------------------------------------------------------- D14


# ---------------------------------------------------------------- D15, D16 options
def test_d15_exact_r2_independent_of_amplitude():
    from xkan.symbolic.fitting import fit_params, fit_params_batched
    x = torch.linspace(-1, 1, 100)
    out = []
    for s in (1., 1e-3, 1e-4):
        y = s * torch.sin(2 * x + 1)
        _, r2_old = fit_params(x, y, torch.sin, verbose=False)
        _, r2_new = fit_params(x, y, torch.sin, verbose=False, r2_eps=0.)
        _, r2_b = fit_params_batched(x[None], y[None], torch.sin, r2_eps=0.)
        out.append((float(r2_old), float(r2_new), float(r2_b[0])))
    assert out[2][0] < 0.5  # the default: r2 0.15 at amplitude 1e-4 (r412)
    assert all(abs(o[1] - 1) < 1e-4 and abs(o[2] - 1) < 1e-4 for o in out)


# ---------------------------------------------------------------- D10, D17 opt-ins


# ---------------------------------------------------------------- D6


# ---------------------------------------------------------------- D15 / D16 alternatives (r2_tau, widen_ranges)
def _choice(x, y, names, **kw):
    import numpy as np
    from xkan.symbolic.fitting import fit_params
    from xkan.symbolic.library import SYMBOLIC_LIB
    loss = lambda r2, c: 0.8 * c + 0.2 * np.log2(1 + 1e-5 - r2)  # suggest_symbolic's default selection loss
    res = {n: float(fit_params(x, y, SYMBOLIC_LIB[n][0], verbose=False, **kw)[1]) for n in names}
    return min(res, key=lambda k: loss(res[k], SYMBOLIC_LIB[k][2]))


def test_d15_tau_penalises_disguised_functions_at_any_amplitude():
    torch.manual_seed(0)
    x = torch.rand(1000) * 2 - 1
    for amp in (1., 1e-2, 1e-4):
        assert _choice(x, amp * x ** 2, ["x^2", "cos"], r2_tau=1e-8) == "x^2", amp  # exact r2: cos at 1e-2
    assert _choice(x, 1e-2 * x ** 2, ["x^2", "cos"], r2_eps=0.) == "cos"  # the failure the penalty addresses
    xs = torch.linspace(-1, 1, 100)
    lib = ["x", "x^2", "x^3", "exp", "sin", "tanh", "gaussian", "1/x", "abs"]
    for s in (1., 1e-3, 1e-4):
        assert _choice(xs, s * torch.sin(2 * xs + 1), lib, r2_tau=1e-8) == "sin", s  # default: x^2 below 1e-3 (#412)


def test_d16_widen_reaches_beyond_a_range_and_leaves_the_rest():
    from xkan.symbolic.fitting import fit_params, fit_params_batched
    from xkan.symbolic.library import SYMBOLIC_LIB
    torch.manual_seed(0)
    x = torch.rand(300) * 0.21 - 0.12  # a small-scale edge input (#529)
    y = torch.tanh(40 * x)
    p, r2 = fit_params(x, y, torch.tanh, verbose=False, widen_ranges=True)
    assert float(r2) > 0.9999 and abs(float(p[0]) + 40) < 0.5  # default: a = -10, r2 0.93
    pb, r2b = fit_params_batched(x[None], y[None], torch.tanh, widen_ranges=True)
    assert abs(float(r2b[0]) - float(r2)) < 1e-6
    for n in ("x", "x^2"):  # r2 does not depend on a's scale: no widening, the default sweep
        assert torch.equal(fit_params(x, y, SYMBOLIC_LIB[n][0], verbose=False)[0],
                           fit_params(x, y, SYMBOLIC_LIB[n][0], verbose=False, widen_ranges=True)[0])
    xi = torch.linspace(-1, 1, 200)
    yi = 2 * torch.sin(3 * xi + 0.5)  # interior optimum
    assert torch.equal(fit_params(xi, yi, torch.sin, verbose=False)[0], fit_params(xi, yi, torch.sin, verbose=False, widen_ranges=True)[0])
    lib = ["tanh", "gaussian", "arctan", "x", "x^2", "sin"]
    assert _choice(x, y, lib) == "x" and _choice(x, y, lib, widen_ranges=True) == "tanh"


# ---------------------------------------------------------------- selection='network', zero_gate, input_standardize
def gauss_model():
    import numpy as np
    from xkan import KANNetwork, create_dataset, fit
    from xkan.interp import prune
    f = lambda x: torch.exp(-x[:, [0]] ** 2 / 2) / np.sqrt(2 * np.pi)
    train, test = create_dataset(f, n_var=1, ranges=[-3, 3], train_num=500, test_num=500, seed=0)
    m = KANNetwork([1, 1, 1], grid=5, k=3, seed=0)
    fit(m, train, test, steps=30, lamb=2e-3)
    m = prune(m, train["x"])
    fit(m, train, test, steps=30)
    return m, train, test


def test_network_selection_with_gate_on_gauss():
    from xkan.symbolic import auto_symbolic
    m, train, test = gauss_model()
    rm = lambda q: float(torch.sqrt(torch.mean((q(test["x"]) - test["y"]) ** 2)))
    out = {}
    for label, kw in (("default", {"zero_gate": False}), ("network+gate", {"selection": "network", "zero_gate": True})):
        c = copy.deepcopy(m)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auto_symbolic(c, train["x"], verbose=0, batched=True, **kw)
        out[label] = rm(c)
    std = float(test["y"].std())
    assert out["default"] > 0.9 * std  # without the gate (0.2.8): a constant model, an edge removed
    assert out["network+gate"] < 0.9 * std


# ---------------------------------------------------------------- KANLib's one-forward grid update


# ---------------------------------------------------------------- refine(refit_mode=...)


NAN_RHS_CODE = """
import sys, json, torch
sys.path.insert(0, {repo!r})
from xkan.spline import solve_lstsq
torch.manual_seed(0)
A, B = torch.rand(3, 200, 13), torch.rand(3, 200, 2)
ref = solve_lstsq(A, B.clone())
B[1] = float('nan'); B[2, 5, 0] = float('inf')
c = solve_lstsq(A, B)
print('OUT ' + json.dumps([torch.equal(c[0], ref[0]), torch.equal(c[2][:, 1], ref[2][:, 1]), bool(c[1].isnan().all()),
                           bool(c[2][:, 0].isnan().all())]), flush=True)
"""


def test_solve_lstsq_non_finite_rhs():
    # a right-hand side with a non-finite value (a diverged model's curves) gets a NaN solution without the LAPACK call
    # (MKL's gelsy printed "Parameter 6 was incorrect on entry to SGELSY" for NaN); the others are solved as before
    r = subprocess.run([sys.executable, "-c", NAN_RHS_CODE.format(repo=str(REV_DIR))], capture_output=True, text=True,
                       env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
    line = [l for l in r.stdout.splitlines() if l.startswith("OUT ")]
    assert line, r.stderr[-3000:]
    assert json.loads(line[0][4:]) == [True, True, True, True]
    assert "MKL ERROR" not in r.stdout + r.stderr


@pytest.mark.parametrize("n", [2000, 27000])  # below / above the samples of one accumulation (chunked above)
def test_fit_params_on_the_input_device(n):
    from xkan.symbolic.fitting import fit_params
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = (torch.rand(n) * 2 - 1).to(dev)
    y = 1.5 * torch.sin(2 * x + 0.3) - 0.2 + 0.01 * torch.randn(n, device=dev)
    p, r2 = fit_params(x, y, torch.sin, verbose=False)  # 0.2.8's device='cpu' default: a device mismatch on CUDA
    assert p.device == x.device and float(r2) > 0.999
    fit = p[2] * torch.sin(p[0] * x + p[1]) + p[3]
    assert float((fit - (1.5 * torch.sin(2 * x + 0.3) - 0.2)).abs().max()) < 0.05


def test_singularity_avoiding_functions():
    from xkan.symbolic import library as U
    y_th = torch.tensor(10.)
    x = torch.cat([torch.randn(200) * 3, torch.tensor([0., 1., -1., 0.01, -0.01, 1e-3, 100.])])[:, None]
    for name in ("f_inv", "f_inv2", "f_inv3", "f_inv4", "f_inv5", "f_invsqrt", "f_log", "f_tan", "f_arctanh",
                 "f_arcsin", "f_arccos", "f_exp", "f_sqrt"):
        x_out, y_out = getattr(U, name)(x, y_th)
        assert torch.isfinite(y_out).all(), name
    # f_exp: NaN above log(max float) in 0.2.8, y_th now
    assert torch.equal(U.f_exp(torch.tensor([[89.], [100.]]), y_th)[1], torch.tensor([[10.], [10.]]))
    x_th = 1 / y_th ** 2  # f_sqrt continuous at its threshold
    v = U.f_sqrt(torch.stack([x_th * (1 - 1e-4), x_th * (1 + 1e-4)])[:, None], y_th)[1]
    assert torch.allclose(v[0], v[1], rtol=1e-3)
