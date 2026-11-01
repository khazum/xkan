"""The KAN layer: every edge (input i, output j) an activation function phi_ij(x_i) = scale_base_ij * b(x_i) +
scale_sp_ij * spline_ij(x_i); each output the sum (or mean) of its edges."""
from dataclasses import replace
from typing import NamedTuple

import numpy as np
import torch

from ..spline import curve2coef, extend_grid, refit_coef, refit_coef_bases, solve_lstsq
from ..utils import sparse_mask
from .bases import basis_values
from .config import KANLayerConfig, base_function
from .grids import get_knot_placement


class EdgeValues(NamedTuple):
    """the outputs and the per-edge values of a KAN layer for inputs x (batch, in_dim)"""
    y: torch.Tensor           # (batch, out_dim)
    preacts: torch.Tensor     # (batch, out_dim, in_dim): the input of each edge
    postacts: torch.Tensor    # (batch, out_dim, in_dim): the output of each edge (masked, DropKAN, aggregation factor)
    postspline: torch.Tensor  # (batch, out_dim, in_dim): the spline part of each edge (before its scale)


class KANLayer(torch.nn.Module):
    """
    A KAN layer of in_dim inputs and out_dim outputs.

    Args:
    -----
        in_dim, out_dim : int
        num : int
            the number of grid intervals G. Default: 5.
        k : int
            the order of the splines. Default: 3.
        device, dtype :
            of the parameters (default: torch's)
        **options :
            KANLayerConfig fields:
            basis : 'bspline' (default), 'rbf' (FastKAN's Gaussians), 'rswaf' (FasterKAN's sech^2) or 'chebyshev'
                (layers.bases; num + k coefficients per edge)
            base_fun : a name ('silu', default; 'identity', 'zero', 'gelu', 'relu', 'tanh') or a module / callable
            noise_scale, scale_base_mu, scale_base_sigma, scale_sp : the initial curves (noise at the interior
                knots, times noise_scale / num) and scales (scale_base ~ U(mu -/+ sigma) / sqrt(in_dim), scale_sp / sqrt(in_dim))
            scale_sp_fan_in : the initial scale_sp divided by sqrt(in_dim) (True, default, pykan 0.2.8) or not (False,
                pykan 0.2.7 and earlier: each curve starts at full strength, which deep, narrow networks such as the
                deep-formula tutorial train better from; a wide layer starts sqrt(in_dim) times larger)
            grid_eps : the share of evenly spaced knots in a grid update (0: quantiles of the samples, 1: uniform)
            grid_range : the initial grid. Default: (-1, 1).
            sp_trainable, sb_trainable : whether scale_sp / scale_base train
            sparse_init : a sparse initial mask
            knot_placement : 'quantile' (default), 'nonzero_quantile' or 'zero_aware' (layers.grids)
            min_knot_gap : the smallest gap between knots, a fraction of the samples' range (default None: none)
            refit_guard : an input whose refitted coefficients are non-finite or grow beyond refit_guard times their
                previous largest keeps its grid (default None: off)
            dropkan : DropKAN (arXiv 2407.13044), in training mode each edge's output dropped with this probability,
                the kept ones scaled by 1 / (1 - dropkan); 0 (default): off
            aggregation : 'sum' (default) or 'mean' (each edge times 1 / in_dim, the in_dim at construction; a pruned
                layer keeps its parent's factor)
            bias : a per-output bias (default False)
            out_of_grid : an input outside the grid's inner range [grid[:, k], grid[:, -k - 1]]: 'zero' (default,
                pykan's) leaves the spline at its incomplete basis there (zero beyond the outer knots) and the base
                path unbounded; 'soft_clamp' evaluates the whole edge at the input saturated within the first interval
                beyond the end (squash_to_grid: bounded outputs, a gradient that fades); 'clamp' at the input clamped
                to the range (constant beyond it: inputs that move past the grid between grid updates get no
                gradient, which can stall a deep network). The bounded modes refit in their own terms at grid updates
                and refine.

    Parameters: grid (in_dim, num + 2k + 1, frozen), coef (in_dim, out_dim, num + k), scale_base, scale_sp and mask
    (in_dim, out_dim; mask frozen), bias (out_dim) when on; buffer agg_scale for 'mean'.

    layer(x) returns the output (batch, out_dim), computed as one matrix product of the basis values of all inputs
    and the weighted coefficients (no per-edge tensor); layer(x, return_edges=True), or layer.edge_forward(x), returns
    the per-edge values as well (EdgeValues). Both go through the module call, so forward hooks see every forward.
    """

    def __init__(self, in_dim, out_dim, num=5, k=3, *, device=None, dtype=None, **options):
        super().__init__()
        self._build(KANLayerConfig(in_dim, out_dim, num, k, **options))
        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    @classmethod
    def from_config(cls, config, device=None, dtype=None):
        options = {f: getattr(config, f) for f in config.__dataclass_fields__ if f not in ("in_dim", "out_dim", "num", "k")}
        return cls(config.in_dim, config.out_dim, config.num, config.k, device=device, dtype=dtype, **options)

    def _build(self, cfg):
        self._config = cfg
        in_dim, out_dim, num, k = cfg.in_dim, cfg.out_dim, cfg.num, cfg.k
        self.in_dim, self.out_dim, self.num, self.k = in_dim, out_dim, num, k
        grid = torch.linspace(cfg.grid_range[0], cfg.grid_range[1], steps=num + 1)[None, :].expand(in_dim, num + 1)
        self.grid = torch.nn.Parameter(extend_grid(grid, k_extend=k)).requires_grad_(False)
        # the initial curves: noise at the interior knots, fitted by each basis (the random draws in pykan 0.2.8's order)
        noises = (torch.rand(num + 1, in_dim, out_dim) - 1 / 2) * cfg.noise_scale / num
        self.basis = cfg.basis
        if cfg.basis == "bspline":
            self.coef = torch.nn.Parameter(curve2coef(self.grid[:, k:-k].permute(1, 0), noises, self.grid, k))
        else:
            mat = basis_values(self.grid[:, k:-k].permute(1, 0), self.grid, k, cfg.basis, num).permute(1, 0, 2)
            self.coef = torch.nn.Parameter(solve_lstsq(mat, noises.permute(1, 0, 2)).permute(0, 2, 1).contiguous())
        mask = sparse_mask(in_dim, out_dim) if cfg.sparse_init else torch.ones(in_dim, out_dim)
        self.mask = torch.nn.Parameter(mask).requires_grad_(False)
        self.scale_base = torch.nn.Parameter(cfg.scale_base_mu * 1 / np.sqrt(in_dim) +
                                             cfg.scale_base_sigma * (torch.rand(in_dim, out_dim) * 2 - 1) * 1 / np.sqrt(in_dim)
                                             ).requires_grad_(cfg.sb_trainable)
        fan_in = 1 / np.sqrt(in_dim) if cfg.scale_sp_fan_in else 1.
        self.scale_sp = torch.nn.Parameter(torch.ones(in_dim, out_dim) * cfg.scale_sp * fan_in * self.mask
                                           ).requires_grad_(cfg.sp_trainable)
        self.base_fun = base_function(cfg.base_fun)
        self.grid_eps = cfg.grid_eps
        self.refit_guard = cfg.refit_guard
        self.knot_placement = cfg.knot_placement
        self.min_knot_gap = cfg.min_knot_gap
        self.dropkan = float(cfg.dropkan)
        self.aggregation = cfg.aggregation
        if cfg.aggregation == "mean":
            # a buffer, so that pruning (get_subset) and checkpoints keep the factor of the layer as constructed
            self.register_buffer("agg_scale", torch.tensor(1.0 / in_dim))
        else:
            self.agg_scale = 1.0
        self.bias = torch.nn.Parameter(torch.zeros(out_dim)) if cfg.bias else None
        self.out_of_grid = cfg.out_of_grid
        self._dropkan_generator = None  # set by a training loop for its training forwards (xkan.train.Trainer)

    @property
    def config(self):
        """the layer's options as they are now (dimensions and grid-update options included)"""
        return replace(self._config, in_dim=self.in_dim, out_dim=self.out_dim, basis=self.basis, grid_eps=self.grid_eps,
                       refit_guard=self.refit_guard, knot_placement=self.knot_placement, min_knot_gap=self.min_knot_gap,
                       dropkan=self.dropkan, aggregation=self.aggregation, bias=self.bias is not None,
                       out_of_grid=self.out_of_grid)

    @property
    def device(self):
        return self.grid.device

    def basis_values(self, x, grid=None):
        """the basis values (batch, in_dim, num + k) of x on grid (default the layer's)"""
        return basis_values(x, self.grid if grid is None else grid, self.k, self.basis, self.num)

    def clamp_to_grid(self, x, grid=None):
        """x (batch, in_dim) clamped per input to the inner range of grid (default the layer's)"""
        g = self.grid if grid is None else grid
        return torch.clamp(x, g[:, self.k].to(x.dtype), g[:, -self.k - 1].to(x.dtype))

    def squash_to_grid(self, x, grid=None):
        """x (batch, in_dim) inside the inner range of grid (default the layer's) as it is; beyond an end e, e + h *
        tanh((x - e) / h), h the width of the first interval beyond e: the input saturates within that interval, its
        derivative 1 at e and decaying beyond (order 0: the hard clamp)"""
        g = self.grid if grid is None else grid
        k = self.k
        if k == 0:
            return self.clamp_to_grid(x, g)
        lo, hi = g[:, k].to(x.dtype), g[:, -k - 1].to(x.dtype)
        h_lo = (g[:, k] - g[:, k - 1]).clamp(min=1e-6).to(x.dtype)
        h_hi = (g[:, -k] - g[:, -k - 1]).clamp(min=1e-6).to(x.dtype)
        above = hi + h_hi * torch.tanh((x - hi) / h_hi)
        below = lo - h_lo * torch.tanh((lo - x) / h_lo)
        return torch.where(x > hi, above, torch.where(x < lo, below, x))

    def _bounded(self, x, grid=None):
        """x (batch, in_dim) as the edges evaluate it under out_of_grid, on grid (default the layer's)"""
        if self.out_of_grid == "zero":
            return x
        g = self.grid if grid is None else grid
        return self.squash_to_grid(x, g) if self.out_of_grid == "soft_clamp" else self.clamp_to_grid(x, g)

    def _spline_basis(self, x, grid=None):
        """the basis values (batch, in_dim, num + k) the splines of the edges take for inputs x under out_of_grid, on
        grid (default the layer's)"""
        g = self.grid if grid is None else grid
        return basis_values(self._bounded(x, g), g, self.k, self.basis, self.num)

    def _base_values(self, x, grid=None):
        """the base path's values (batch, in_dim) b(.) for inputs x under out_of_grid, on grid (default the layer's)"""
        return self.base_fun(self._bounded(x, grid))

    def _dropkan_active(self):
        return self.dropkan > 0. and self.training

    def forward(self, x, return_edges=False):
        """the output (batch, out_dim) for inputs x (batch, in_dim); with return_edges, EdgeValues"""
        if return_edges or self._dropkan_active():
            edges = self._edge_values(x)  # (DropKAN masks individual edges)
            return edges if return_edges else edges.y
        x = self._bounded(x)
        basis = self.basis_values(x).flatten(start_dim=1)  # (batch, in_dim * (num + k))
        w = (self.coef * (self.scale_sp * self.mask)[:, :, None]).permute(1, 0, 2).flatten(start_dim=1)  # (out, in * (num + k))
        y = torch.nn.functional.linear(basis, w) + self.base_fun(x) @ (self.scale_base * self.mask)
        if self.aggregation == "mean":
            y = y * self.agg_scale
        if self.bias is not None:
            y = y + self.bias
        return y

    def edge_forward(self, x):
        """the output and the per-edge values for inputs x (batch, in_dim): EdgeValues (the module call, hooks included)"""
        return self(x, return_edges=True)

    def _edge_values(self, x):
        batch = x.shape[0]
        preacts = x[:, None, :].clone().expand(batch, self.out_dim, self.in_dim)
        base = self._base_values(x)  # (batch, in_dim)
        y = torch.einsum("ijk,jlk->ijl", self._spline_basis(x), self.coef)  # (batch, in_dim, out_dim)
        postspline = y.clone().permute(0, 2, 1)
        y = self.scale_base[None, :, :] * base[:, :, None] + self.scale_sp[None, :, :] * y
        y = self.mask[None, :, :] * y
        if self._dropkan_active():
            # each edge's output dropped per sample, the kept ones scaled; the draws from the global generator, or from
            # _dropkan_generator when set (the same masks for every evaluation of a training step)
            g = self._dropkan_generator
            u = torch.rand_like(y) if g is None else torch.rand(y.shape, generator=g, device=y.device, dtype=y.dtype)
            y = y * ((u >= self.dropkan).to(y.dtype) / (1.0 - self.dropkan))
        if self.aggregation == "mean":
            y = y * self.agg_scale
        postacts = y.clone().permute(0, 2, 1)
        y = torch.sum(y, dim=1)
        if self.bias is not None:
            y = y + self.bias
        return EdgeValues(y, preacts, postacts, postspline)

    # ------------------------------------------------------------------------------------------------- grid updates

    def _refit_from(self, x, grid, coef, k, basis, new_grid):
        # coefficients on new_grid (this layer's basis and order) of the curves (grid, coef, k, basis) at x, by least
        # squares: refit_coef for B-splines on both sides, else refit_coef_bases (float64)
        if basis == "bspline" and self.basis == "bspline":
            return refit_coef(x, grid, coef, k, new_grid, self.k)
        return refit_coef_bases(basis_values(x, grid, k, basis), basis_values(x, new_grid, self.k, self.basis, self.num), coef)

    def _refit_mapped(self, x_pos, old_basis, old_base, old_coef):
        # under a bounded out_of_grid, the coefficients on this layer's grid of the old curves as the old edges
        # evaluate them at x_pos (their basis values old_basis, base path old_base), the new coefficients fitted to
        # them as this layer's edges evaluate x_pos
        new_basis = self._spline_basis(x_pos)
        coef = refit_coef_bases(old_basis, new_basis, old_coef)
        # where the base path's value moves with the grid, its change, fitted on the new basis, goes into each edge's
        # spline in the ratio of the edge's base and spline scales (edges with a spline scale below 1e-3 keep their
        # spline)
        delta = (old_base - self._base_values(x_pos))[:, :, None]  # (batch, in_dim, 1)
        d_coef = refit_coef_bases(delta, new_basis, torch.ones_like(old_coef[:, :1, :1]))  # (in_dim, 1, n)
        sp = self.scale_sp.data
        ratio = torch.where(sp.abs() >= 1e-3, self.scale_base.data / torch.where(sp.abs() >= 1e-3, sp, 1.), 0.)
        return coef + ratio[:, :, None] * d_coef

    @staticmethod
    def _with_uniform_points(x_pos, n):
        # the samples plus n evenly spaced points over their range: tied samples (zero-inflated or discrete inputs) pin
        # a refitted curve at a few distinct points only, the even points keep it close between them as well
        u = torch.linspace(0, 1, n, device=x_pos.device, dtype=x_pos.dtype)[:, None]
        return torch.cat([x_pos, x_pos[[0]] + (x_pos[[-1]] - x_pos[[0]]) * u], dim=0)

    def update_grid_from_samples(self, x, mode="sample", grid_eps=None):
        """
        new knots from the samples x (batch, in_dim) (the layer's knot_placement), the curves refitted on them

        mode: the points of the refit, 'sample' (default), the samples; 'sample+uniform', the samples plus evenly
        spaced points over their range; 'grid', the knots of a grid with twice the intervals. grid_eps: for this update
        (default the layer's). An input with non-finite samples keeps its grid and coefficients; with refit_guard, so
        does an input whose refit is non-finite or grows beyond refit_guard times its previous largest coefficient.
        """
        if mode not in ("sample", "sample+uniform", "grid"):
            raise ValueError(f"unknown mode {mode!r}; 'sample', 'sample+uniform' or 'grid'")
        grid_eps = self.grid_eps if grid_eps is None else float(grid_eps)
        batch = x.shape[0]
        bad = ~torch.isfinite(x).all(dim=0)
        if bad.any():
            # placeholder samples evenly over the old interior range keep the batched refit finite (restored below)
            span = self.grid.data[bad][:, [self.k, -self.k - 1]].to(x.dtype)
            u = torch.linspace(0, 1, batch, device=x.device, dtype=x.dtype)[:, None]
            x = x.clone()
            x[:, bad] = span[:, 0] + (span[:, 1] - span[:, 0]) * u
        x_pos = torch.sort(x, dim=0)[0]
        num = self.grid.shape[1] - 1 - 2 * self.k
        placement = get_knot_placement(self.knot_placement)
        new_grid = placement.grid(x, x_pos, num, grid_eps, self.k, self.min_knot_gap)
        if mode == "grid":
            x_pos = placement.interior(x, x_pos, 2 * num, grid_eps, self.k, self.min_knot_gap).permute(1, 0)
        elif mode == "sample+uniform":
            x_pos = self._with_uniform_points(x_pos, 4 * (num + self.k))
        old_grid, old_coef = self.grid.data.clone(), self.coef.data.clone()
        if self.out_of_grid != "zero":
            old_basis, old_base = self._spline_basis(x_pos, old_grid), self._base_values(x_pos, old_grid)
        self.grid.data = new_grid
        if self.out_of_grid == "zero":
            self.coef.data = self._refit_from(x_pos, old_grid, old_coef, self.k, self.basis, self.grid.data)
        else:
            self.coef.data = self._refit_mapped(x_pos, old_basis, old_base, old_coef)
        if self.refit_guard is not None:
            old_max = old_coef.abs().amax(dim=(1, 2))
            new_max = self.coef.data.abs().amax(dim=(1, 2))
            bad = bad | ~torch.isfinite(self.coef.data).all(dim=2).all(dim=1) | ~torch.isfinite(self.grid.data).all(dim=1) \
                | (new_max > self.refit_guard * torch.clamp(old_max, min=1.0))
        if bad.any():
            self.grid.data[bad] = old_grid[bad]
            self.coef.data[bad] = old_coef[bad]

    def initialize_grid_from_parent(self, parent, x, mode="sample"):
        """
        this layer's grid from a parent layer's (usually coarser): the parent's interior knots interpolated linearly to
        this layer's number of intervals; the coefficients fitted to the parent's curves at x (mode as in
        update_grid_from_samples). Under a bounded out_of_grid the parent's curves are taken as its edges evaluate
        them, and this layer takes the parent's scales (the base path's change goes into the splines in their ratio).
        """
        x_pos = torch.sort(x, dim=0)[0]
        num = self.grid.shape[1] - 1 - 2 * self.k

        def interpolated(n):
            # the parent's interior knots as a piecewise-linear function of [-1, 1], at n + 1 even points (a k = 1
            # helper layer, in the parent's dtype; its random initial values unused, drawn on a forked stream)
            knots = parent.grid[:, parent.k:-parent.k]
            with torch.random.fork_rng(devices=[]):
                sp2 = KANLayer(1, self.in_dim, num=knots.shape[1] - 1, k=1, scale_base_mu=0.0, scale_base_sigma=0.0,
                               out_of_grid="zero").to(device=x.device, dtype=parent.grid.dtype)
            sp2.coef.data = curve2coef(sp2.grid[:, sp2.k:-sp2.k].permute(1, 0).expand(-1, self.in_dim),
                                       knots.permute(1, 0).unsqueeze(dim=2), sp2.grid[:, :], k=1).permute(1, 0, 2)
            percentile = torch.linspace(-1, 1, n + 1).to(device=self.device, dtype=sp2.coef.dtype)
            return sp2.edge_forward(percentile.unsqueeze(dim=1)).y.permute(1, 0)

        grid = interpolated(num)
        if mode == "grid":
            x_pos = interpolated(2 * num).permute(1, 0)
        elif mode == "sample+uniform":
            x_pos = self._with_uniform_points(x_pos, 4 * (self.num + self.k))
        elif mode != "sample":
            raise ValueError(f"unknown refit mode {mode!r}; 'sample', 'sample+uniform' or 'grid'")
        self.grid.data = extend_grid(grid, k_extend=self.k)
        if self.out_of_grid == "zero" and parent.out_of_grid == "zero":
            self.coef.data = self._refit_from(x_pos, parent.grid, parent.coef, parent.k, parent.basis, self.grid.data)
        else:
            with torch.no_grad():
                self.scale_base.data, self.scale_sp.data = parent.scale_base.data.clone(), parent.scale_sp.data.clone()
                self.coef.data = self._refit_mapped(x_pos, parent._spline_basis(x_pos), parent._base_values(x_pos),
                                                    parent.coef.data)

    # ------------------------------------------------------------------------------------------------- structure

    def get_subset(self, in_id, out_id):
        """a layer with the inputs in_id and outputs out_id of this one (same options and curves)"""
        with torch.random.fork_rng(devices=[]):  # its random initial values are overwritten
            spb = KANLayer.from_config(replace(self.config, in_dim=len(in_id), out_dim=len(out_id), base_fun=self.base_fun))
        spb.grid.data = self.grid[in_id]
        spb.coef.data = self.coef[in_id][:, out_id]
        spb.scale_base.data = self.scale_base[in_id][:, out_id]
        spb.scale_sp.data = self.scale_sp[in_id][:, out_id]
        spb.mask.data = self.mask[in_id][:, out_id]
        spb.scale_base.requires_grad_(self.scale_base.requires_grad)
        spb.scale_sp.requires_grad_(self.scale_sp.requires_grad)
        if self.bias is not None:
            spb.bias.data = self.bias.data[out_id].clone()
        if self.aggregation == "mean":
            spb.agg_scale = self.agg_scale.clone()  # the parent's factor
        spb.train(self.training)
        return spb

    def reindexed(self, in_map, out_map):
        """a layer whose input i is this layer's input in_map[i] and output j its output out_map[j] (same options and
        curves); None in a map: a new input / output, disconnected (mask 0)"""
        with torch.random.fork_rng(devices=[]):  # the new edges' random initial values are masked
            new = KANLayer.from_config(replace(self.config, in_dim=len(in_map), out_dim=len(out_map), base_fun=self.base_fun),
                                       device=self.grid.device, dtype=self.coef.dtype)
        ii = [k for k, i in enumerate(in_map) if i is not None]
        jj = [k for k, j in enumerate(out_map) if j is not None]
        src_i = torch.tensor([in_map[k] for k in ii], dtype=torch.long, device=self.grid.device)
        src_j = torch.tensor([out_map[k] for k in jj], dtype=torch.long, device=self.grid.device)
        rows = torch.tensor(ii, dtype=torch.long, device=self.grid.device)[:, None]
        cols = torch.tensor(jj, dtype=torch.long, device=self.grid.device)[None, :]
        with torch.no_grad():
            new.mask.zero_()
            new.grid[rows[:, 0]] = self.grid[src_i]
            for name in ("coef", "scale_base", "scale_sp", "mask"):
                getattr(new, name)[rows, cols] = getattr(self, name)[src_i][:, src_j]
            if self.bias is not None:
                new.bias.zero_()
                new.bias[cols[0]] = self.bias[src_j]
        new.scale_base.requires_grad_(self.scale_base.requires_grad)
        new.scale_sp.requires_grad_(self.scale_sp.requires_grad)
        if self.aggregation == "mean":
            new.agg_scale = self.agg_scale.clone()  # the parent's factor
        new.train(self.training)
        return new

    def swap(self, i1, i2, mode="in"):
        """swaps inputs (mode 'in') or outputs ('out') i1 and i2"""
        with torch.no_grad():
            def swap_(data, mode):
                if mode == "in":
                    data[i1], data[i2] = data[i2].clone(), data[i1].clone()
                else:
                    data[:, i1], data[:, i2] = data[:, i2].clone(), data[:, i1].clone()

            if mode == "in":
                swap_(self.grid.data, "in")
            for p in (self.coef, self.scale_base, self.scale_sp, self.mask):
                swap_(p.data, mode)
            if mode == "out" and self.bias is not None:
                self.bias.data[i1], self.bias.data[i2] = self.bias.data[i2].clone(), self.bias.data[i1].clone()
