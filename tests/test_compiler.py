"""compile_formula: a KANNetwork computing a sympy formula (its outputs against the formula, and the formula of the
network back), and the errors for formulas outside the library."""
import pytest
import sympy
import torch
from sympy import exp, pi, sin, sqrt, symbols

from xkan.symbolic import symbolic_formula
from xkan.symbolic.compiler import compile_formula

a, b, c = symbols("a b c")
EXPRESSIONS = [exp(sin(pi * a) + b ** 2), a * b + sin(c), sqrt(a ** 2 + b ** 2 + 1), (a + 1) * (b - 2) * c, a / (b + 3)]


@pytest.mark.parametrize("expr", EXPRESSIONS)
def test_the_network_computes_the_formula(expr):
    variables = sorted(expr.free_symbols, key=str)
    new = compile_formula(variables, expr)
    x = torch.rand(200, len(variables)) * 2 - 1
    f = sympy.lambdify(variables, expr)
    assert torch.allclose(new(x)[:, 0].double(), torch.tensor(f(*x.double().T)), rtol=1e-4, atol=1e-4)
    formula, _ = symbolic_formula(new, var=variables)
    g = sympy.lambdify(variables, formula[0])
    assert torch.allclose(torch.as_tensor(g(*x.double().T), dtype=torch.float64), torch.tensor(f(*x.double().T)),
                          rtol=1e-4, atol=1e-4)


def test_unsupported_formulas_raise():
    with pytest.raises(ValueError, match="exponent"):
        compile_formula([a], a ** 2.5)
    with pytest.raises(ValueError, match="two functions"):
        compile_formula([a], a ** 2 + sin(a))


def test_network_options_reach_the_compiled_network():
    # options of the network (here a linear base function and a seed) are those of the compiled network; the formula
    # it computes is the same
    expr = a * b + sin(c)
    plain, linear = compile_formula([a, b, c], expr), compile_formula([a, b, c], expr, base_fun="identity", seed=3)
    assert linear.config.layer["base_fun"] == "identity" and linear.config.seed == 3
    assert all(isinstance(layer.base_fun, torch.nn.Identity) for layer in linear.layers)
    x = torch.rand(100, 3) * 2 - 1
    assert torch.allclose(linear(x), plain(x), atol=1e-5)
    with pytest.raises(TypeError, match="from the formula"):
        compile_formula([a, b, c], expr, mult_arity=3)
