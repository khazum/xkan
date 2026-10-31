"""Basis families of the edge functions: num + k basis values per input on the layer's knot grid. A new family is a
subclass of Basis registered with register_basis."""
from ..spline import B_batch, chebyshev_bases, rbf_bases, rswaf_bases


class Basis:
    """values(x, grid, k, num): the basis values of x (batch, in_dim) on grid, shape (batch, in_dim, num + k)."""
    name = ""

    def values(self, x, grid, k, num):
        raise NotImplementedError


class BSpline(Basis):
    """B-splines of order k on the knot grid (B_batch)."""
    name = "bspline"

    def values(self, x, grid, k, num):
        return B_batch(x, grid, k)


class RBF(Basis):
    """Gaussian radial basis functions evenly over the interior knot range (spline.rbf_bases)."""
    name = "rbf"

    def values(self, x, grid, k, num):
        return rbf_bases(x, grid, k, num)


class RSWAF(Basis):
    """FasterKAN's reflectional switch functions, sech^2 bumps where the Gaussians are (spline.rswaf_bases)."""
    name = "rswaf"

    def values(self, x, grid, k, num):
        return rswaf_bases(x, grid, k, num)


class Chebyshev(Basis):
    """Chebyshev polynomials of a tanh-squashed input (spline.chebyshev_bases)."""
    name = "chebyshev"

    def values(self, x, grid, k, num):
        return chebyshev_bases(x, grid, k, num)


BASES = {}


def register_basis(basis):
    """adds a Basis instance under its name; returns it"""
    if not isinstance(basis, Basis) or not basis.name:
        raise ValueError("register_basis takes a Basis instance with a name")
    BASES[basis.name] = basis
    return basis


def get_basis(name):
    if name not in BASES:
        raise ValueError(f"unknown basis {name!r}; one of {sorted(BASES)}")
    return BASES[name]


def basis_values(x, grid, k, basis="bspline", num=None):
    """the basis values (batch, in_dim, num + k) of x for a basis family; num from the grid by default"""
    num = grid.shape[1] - 1 - 2 * k if num is None else num
    return get_basis(basis).values(x, grid, k, num)


for _basis in (BSpline(), RBF(), RSWAF(), Chebyshev()):
    register_basis(_basis)
