"""Knot placements of the grid updates. A new placement is a subclass of KnotPlacement registered with
register_knot_placement."""
import torch

from ..spline import extend_grid, nonzero_quantile_knots, separate_knots, zero_aware_grid


class KnotPlacement:
    """
    interior(x, x_sorted, num, grid_eps, k, min_knot_gap): the num + 1 interior knots per input, (in_dim, num + 1);
    grid(...): the whole grid with k knots added on each side, (in_dim, num + 2k + 1). x are the samples
    (batch, in_dim), x_sorted the same sorted per input.
    """
    name = ""

    def interior(self, x, x_sorted, num, grid_eps, k, min_knot_gap):
        raise NotImplementedError

    def grid(self, x, x_sorted, num, grid_eps, k, min_knot_gap):
        return extend_grid(self.interior(x, x_sorted, num, grid_eps, k, min_knot_gap), k_extend=k)


def _quantile_interior(x_sorted, num, grid_eps):
    # quantiles of the samples blended with evenly spaced knots over their range (grid_eps: the share of the latter)
    batch = x_sorted.shape[0]
    ids = [int(batch / num * i) for i in range(num)] + [-1]
    adaptive = x_sorted[ids, :].permute(1, 0)
    h = (adaptive[:, [-1]] - adaptive[:, [0]]) / num
    uniform = adaptive[:, [0]] + h * torch.arange(num + 1)[None, :].to(x_sorted.device)
    return grid_eps * uniform + (1 - grid_eps) * adaptive


def _min_gap(knots, x_sorted, min_knot_gap):
    if min_knot_gap is None:
        return knots
    return separate_knots(knots, min_knot_gap * torch.clamp(x_sorted[-1] - x_sorted[0], min=0.1))


class Quantile(KnotPlacement):
    """knots at quantiles of the samples (pykan 0.2.8's placement)"""
    name = "quantile"

    def interior(self, x, x_sorted, num, grid_eps, k, min_knot_gap):
        return _min_gap(_quantile_interior(x_sorted, num, grid_eps), x_sorted, min_knot_gap)


class NonzeroQuantile(KnotPlacement):
    """for an input with exact zeros, non-zero values and no negative values: a knot at 0, the others at quantiles of
    the non-zero values (quantiles of all samples pile the knots on the zeros of sparse inputs)"""
    name = "nonzero_quantile"

    def interior(self, x, x_sorted, num, grid_eps, k, min_knot_gap):
        knots = nonzero_quantile_knots(x_sorted, num, grid_eps, _quantile_interior(x_sorted, num, grid_eps))
        return _min_gap(knots, x_sorted, min_knot_gap)


class ZeroAware(KnotPlacement):
    """spline.zero_aware_grid: the non-zero percentiles of 'nonzero_quantile' with a minimum knot separation and the
    outer knots spaced by the first and last intervals (min_knot_gap does not apply)"""
    name = "zero_aware"

    def interior(self, x, x_sorted, num, grid_eps, k, min_knot_gap):
        return zero_aware_grid(x, num, k, grid_eps)[:, k:-k]

    def grid(self, x, x_sorted, num, grid_eps, k, min_knot_gap):
        return zero_aware_grid(x, num, k, grid_eps)


KNOT_PLACEMENTS = {}


def register_knot_placement(placement):
    """adds a KnotPlacement instance under its name; returns it"""
    if not isinstance(placement, KnotPlacement) or not placement.name:
        raise ValueError("register_knot_placement takes a KnotPlacement instance with a name")
    KNOT_PLACEMENTS[placement.name] = placement
    return placement


def get_knot_placement(name):
    if name not in KNOT_PLACEMENTS:
        raise ValueError(f"unknown knot_placement {name!r}; one of {sorted(KNOT_PLACEMENTS)}")
    return KNOT_PLACEMENTS[name]


for _placement in (Quantile(), NonzeroQuantile(), ZeroAware()):
    register_knot_placement(_placement)
