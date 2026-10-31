"""The library of symbolic functions an edge can be fixed to."""
from collections.abc import Mapping

import sympy
import torch

from ..layers.symbolic import SymbolicFunction

# singularity protection functions: (params, value) for an input x and a threshold y_th (a tensor)
#
# The singular branch is evaluated at a safe stand-in where it is masked out (_away), so that its gradient there is 0
# and not inf * 0 = NaN (pykan 0.2.8: NaN gradients at x = 0 for 1/x, sqrt, log, ..., and at x = +-1 for arctanh);
# the values are unchanged.
def _away(x, far, safe):
    return torch.where(far, x, torch.as_tensor(safe, dtype=x.dtype, device=x.device))

f_inv = lambda x, y_th: ((x_th := 1/y_th), y_th/x_th*x * (torch.abs(x) < x_th) + torch.nan_to_num(1/_away(x, torch.abs(x) >= x_th, x_th)) * (torch.abs(x) >= x_th))
f_inv2 = lambda x, y_th: ((x_th := 1/y_th**(1/2)), y_th * (torch.abs(x) < x_th) + torch.nan_to_num(1/_away(x, torch.abs(x) >= x_th, x_th)**2) * (torch.abs(x) >= x_th))
f_inv3 = lambda x, y_th: ((x_th := 1/y_th**(1/3)), y_th/x_th*x * (torch.abs(x) < x_th) + torch.nan_to_num(1/_away(x, torch.abs(x) >= x_th, x_th)**3) * (torch.abs(x) >= x_th))
f_inv4 = lambda x, y_th: ((x_th := 1/y_th**(1/4)), y_th * (torch.abs(x) < x_th) + torch.nan_to_num(1/_away(x, torch.abs(x) >= x_th, x_th)**4) * (torch.abs(x) >= x_th))
f_inv5 = lambda x, y_th: ((x_th := 1/y_th**(1/5)), y_th/x_th*x * (torch.abs(x) < x_th) + torch.nan_to_num(1/_away(x, torch.abs(x) >= x_th, x_th)**5) * (torch.abs(x) >= x_th))
# the linear branch is y_th * x, continuous with sign(x) sqrt|x| at |x| = x_th = 1/y_th^2 (pykan 0.2.8: x_th/y_th * x,
# which jumps from 1e-5 to 0.1 there for y_th = 10)
f_sqrt = lambda x, y_th: ((x_th := 1/y_th**2), y_th*x * (torch.abs(x) < x_th) + torch.nan_to_num(torch.sqrt(torch.abs(_away(x, torch.abs(x) >= x_th, x_th)))*torch.sign(x)) * (torch.abs(x) >= x_th))
# (params, value) as the others (pykan 0.2.8 returned the value only, and the layer took its second row)
f_power1d5 = lambda x, y_th: ((), torch.abs(x)**1.5)
f_invsqrt = lambda x, y_th: ((x_th := 1/y_th**2), y_th * (torch.abs(x) < x_th) + torch.nan_to_num(1/torch.sqrt(torch.abs(_away(x, torch.abs(x) >= x_th, x_th)))) * (torch.abs(x) >= x_th))
f_log = lambda x, y_th: ((x_th := torch.e**(-y_th)), - y_th * (torch.abs(x) < x_th) + torch.nan_to_num(torch.log(torch.abs(_away(x, torch.abs(x) >= x_th, 1.)))) * (torch.abs(x) >= x_th))
# params as one tuple (pykan 0.2.8 returned three elements, and the layer took delta as the value)
f_tan = lambda x, y_th: (((clip := x % torch.pi), (delta := torch.pi/2-torch.arctan(y_th))), - y_th/delta * (clip - torch.pi/2) * (torch.abs(clip - torch.pi/2) < delta) + torch.nan_to_num(torch.tan(clip)) * (torch.abs(clip - torch.pi/2) >= delta))
f_arctanh = lambda x, y_th: ((delta := 1-torch.tanh(y_th) + 1e-4), y_th * torch.sign(x) * (torch.abs(x) > 1 - delta) + torch.nan_to_num(torch.arctanh(_away(x, torch.abs(x) <= 1 - delta, 0.))) * (torch.abs(x) <= 1 - delta))
f_arcsin = lambda x, y_th: ((), torch.pi/2 * torch.sign(x) * (torch.abs(x) > 1) + torch.nan_to_num(torch.arcsin(x)) * (torch.abs(x) <= 1))
f_arccos = lambda x, y_th: ((), torch.pi/2 * (1-torch.sign(x)) * (torch.abs(x) > 1) + torch.nan_to_num(torch.arccos(x)) * (torch.abs(x) <= 1))
# exp of the clamped input (pykan 0.2.8: exp(x) = inf for x > 88 in float32, and inf * 0 = NaN)
f_exp = lambda x, y_th: ((x_th := torch.log(y_th)), y_th * (x > x_th) + torch.exp(torch.clamp(x, max=x_th)) * (x <= x_th))

SYMBOLIC_LIB = {'x': (lambda x: x, lambda x: x, 1, lambda x, y_th: ((), x)),
                 'x^2': (lambda x: x**2, lambda x: x**2, 2, lambda x, y_th: ((), x**2)),
                 'x^3': (lambda x: x**3, lambda x: x**3, 3, lambda x, y_th: ((), x**3)),
                 'x^4': (lambda x: x**4, lambda x: x**4, 3, lambda x, y_th: ((), x**4)),
                 'x^5': (lambda x: x**5, lambda x: x**5, 3, lambda x, y_th: ((), x**5)),
                 '1/x': (lambda x: 1/x, lambda x: 1/x, 2, f_inv),
                 '1/x^2': (lambda x: 1/x**2, lambda x: 1/x**2, 2, f_inv2),
                 '1/x^3': (lambda x: 1/x**3, lambda x: 1/x**3, 3, f_inv3),
                 '1/x^4': (lambda x: 1/x**4, lambda x: 1/x**4, 4, f_inv4),
                 '1/x^5': (lambda x: 1/x**5, lambda x: 1/x**5, 5, f_inv5),
                 'sqrt': (lambda x: torch.sqrt(x), lambda x: sympy.sqrt(x), 2, f_sqrt),
                 'x^0.5': (lambda x: torch.sqrt(x), lambda x: sympy.sqrt(x), 2, f_sqrt),
                 'x^1.5': (lambda x: torch.sqrt(x)**3, lambda x: sympy.sqrt(x)**3, 4, f_power1d5),
                 '1/sqrt(x)': (lambda x: 1/torch.sqrt(x), lambda x: 1/sympy.sqrt(x), 2, f_invsqrt),
                 '1/x^0.5': (lambda x: 1/torch.sqrt(x), lambda x: 1/sympy.sqrt(x), 2, f_invsqrt),
                 'exp': (lambda x: torch.exp(x), lambda x: sympy.exp(x), 2, f_exp),
                 'log': (lambda x: torch.log(x), lambda x: sympy.log(x), 2, f_log),
                 'abs': (lambda x: torch.abs(x), lambda x: sympy.Abs(x), 3, lambda x, y_th: ((), torch.abs(x))),
                 'sin': (lambda x: torch.sin(x), lambda x: sympy.sin(x), 2, lambda x, y_th: ((), torch.sin(x))),
                 'cos': (lambda x: torch.cos(x), lambda x: sympy.cos(x), 2, lambda x, y_th: ((), torch.cos(x))),
                 'tan': (lambda x: torch.tan(x), lambda x: sympy.tan(x), 3, f_tan),
                 'tanh': (lambda x: torch.tanh(x), lambda x: sympy.tanh(x), 3, lambda x, y_th: ((), torch.tanh(x))),
                 'sgn': (lambda x: torch.sign(x), lambda x: sympy.sign(x), 3, lambda x, y_th: ((), torch.sign(x))),
                 'arcsin': (lambda x: torch.arcsin(x), lambda x: sympy.asin(x), 4, f_arcsin),
                 'arccos': (lambda x: torch.arccos(x), lambda x: sympy.acos(x), 4, f_arccos),
                 'arctan': (lambda x: torch.arctan(x), lambda x: sympy.atan(x), 4, lambda x, y_th: ((), torch.arctan(x))),
                 'arctanh': (lambda x: torch.arctanh(x), lambda x: sympy.atanh(x), 4, f_arctanh),
                 '0': (lambda x: x*0, lambda x: x*0, 0, lambda x, y_th: ((), x*0)),
                 'gaussian': (lambda x: torch.exp(-x**2), lambda x: sympy.exp(-x**2), 3, lambda x, y_th: ((), torch.exp(-x**2))),
                 #'cosh': (lambda x: torch.cosh(x), lambda x: sympy.cosh(x), 5),
                 #'sigmoid': (lambda x: torch.sigmoid(x), sympy.Function('sigmoid'), 4),
                 #'relu': (lambda x: torch.relu(x), relu),
}


class PlainFunction:
    '''
    the singularity-avoiding form (x, y_th) -> (params, value) of a function without singularities; a class rather than
    a lambda, so that it pickles when the function does
    '''
    def __init__(self, fun):
        self.fun = fun

    def __call__(self, x, y_th):
        return (), self.fun(x)


class SymbolicLibrary(Mapping):
    """
    Named SymbolicFunctions (torch form, sympy form, complexity, singularity-avoiding form). library[name] gives the
    function; add(name, fn, sympy_fn, complexity, singular) adds one (singular default: no avoidance); subset(names)
    is a library of those functions.
    """

    def __init__(self, functions=()):
        self._functions = {}
        for f in (functions.values() if isinstance(functions, Mapping) else functions):
            self._functions[f.name] = f

    def __getitem__(self, name):
        if name not in self._functions:
            raise KeyError(f"no symbolic function {name!r}; the library has {sorted(self._functions)}")
        return self._functions[name]

    def __iter__(self):
        return iter(self._functions)

    def __len__(self):
        return len(self._functions)

    def __repr__(self):
        return f"SymbolicLibrary({list(self._functions)})"

    def add(self, name, fn, sympy_fn=None, complexity=1, singular=None):
        self._functions[name] = SymbolicFunction(name, fn, fn if sympy_fn is None else sympy_fn, complexity,
                                                 singular or (lambda x, y_th: ((), fn(x))))
        return self

    def subset(self, names):
        return SymbolicLibrary([self[n] for n in names])

    def resolve(self, lib):
        """a library from None (this one), a list of names, or a library"""
        if lib is None:
            return self
        if isinstance(lib, SymbolicLibrary):
            return lib
        return self.subset(list(lib))


def default_library():
    """a new library of pykan's functions (x, x^2, ..., sin, tanh, gaussian, 0, ...)"""
    return SymbolicLibrary(SymbolicFunction(name, fn, sym, c, singular) for name, (fn, sym, c, singular) in SYMBOLIC_LIB.items())
