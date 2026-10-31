"""Fitting symbolic functions to edges: the affine parameters (a, b, c, d) of c * f(a * x + b) + d by a grid search over
(a, b) and least squares for (c, d); one edge (fit_params) or many at once (fit_params_batched)."""
import sympy
import torch

__all__ = ["fit_params", "fit_params_batched", "REAL_SYMPY"]


def _r2(numerator, sxx, syy, r2_eps, r2_tau=None, n=1):
    '''
    r2 of the fits: numerator = sxy^2, sxx and syy the sums of squares of fun(a*x+b) and y about their means
    '''
    if r2_tau is not None:
        denominator = syy * (sxx + n * r2_tau)
    elif r2_eps > 0:
        return numerator/(sxx*syy+r2_eps)
    else:
        denominator = sxx * syy
    positive = denominator > 0
    return torch.where(positive, numerator / torch.where(positive, denominator, torch.ones_like(denominator)), torch.zeros_like(numerator))


_WIDEN_STEPS = 5

# The grid searches evaluate fun(a * x + b) on blocks of rows of a of about this many values, so that a block's
# temporaries stay in the CPU's cache (each sum still runs over the samples in order; the block size changes the
# summation's rounding by about 1e-7). None: the whole grid at once, the summation of xkan before 2026-10-30.
SWEEP_BLOCK_ELEMENTS = 2**20


def _sweep_rows(samples, grid_number):
    # rows of a per block for samples x rows x grid_number values
    if SWEEP_BLOCK_ELEMENTS is None:
        return grid_number
    return max(1, min(grid_number, SWEEP_BLOCK_ELEMENTS // max(1, samples * grid_number)))


def _widened(r2_old, r2_new):
    '''
    whether a widened sweep's best r2 pays: more than 1e-3 of the remaining 1 - r2 (and more than 1e-9)
    '''
    return r2_new - r2_old > torch.clamp(1e-3 * (1 - r2_old), min=1e-9)


def fit_params(x, y, fun, a_range=(-10,10), b_range=(-10,10), grid_number=101, iteration=3, verbose=True, device='cpu', r2_eps=1e-4,
               r2_tau=None, widen_ranges=False):
    '''
    fit a, b, c, d such that
    
    .. math::
        |y-(cf(ax+b)+d)|^2
        
    is minimized. Both x and y are 1D array. Sweep a and b, find the best fitted model.
    
    Args:
    -----
        x : 1D array
            x values
        y : 1D array
            y values
        fun : function
            symbolic function
        a_range : tuple
            sweeping range of a
        b_range : tuple
            sweeping range of b
        grid_num : int
            number of steps along a and b
        iteration : int
            number of zooming in
        verbose : bool
            print extra information if True
        device : str
            device
        r2_eps : float
            added to the denominator of r2 (1e-4, pykan 0.2.8's value, by default); it makes r2 depend on the
            amplitude of y. 0: the exact r2 (0 for a constant fun or y)
        r2_tau : float or None
            if given, r2 is the exact r2 times var_f / (var_f + r2_tau), var_f the variance of fun(a*x+b) over the
            samples: a candidate used where it barely varies (e.g. cos(a x + b) with a tiny a, a quadratic in
            disguise) is penalised, the same for every amplitude of y (r2_eps's penalty depended on it)
        widen_ranges : bool
            if True, when the first sweep's best a or b lies on the boundary of its range, the range is extended on
            that side by its width (up to 5 times) as long as that raises the best r2 materially (by more than 1e-3
            of 1 - r2); otherwise, and for interior optima, the sweep is the default one

    Returns:
    --------
        a_best : float
            best fitted a
        b_best : float
            best fitted b
        c_best : float
            best fitted c
        d_best : float
            best fitted d
        r2_best : float
            best r2 (coefficient of determination)
    
    Example
    -------
    >>> num = 100
    >>> x = torch.linspace(-1,1,steps=num)
    >>> noises = torch.normal(0,1,(num,)) * 0.02
    >>> y = 5.0*torch.sin(3.0*x + 2.0) + 0.7 + noises
    >>> fit_params(x, y, torch.sin)
    r2 is 0.9999727010726929
    (tensor([2.9982, 1.9996, 5.0053, 0.7011]), tensor(1.0000))
    '''
    # fit a, b, c, d such that y=c*fun(a*x+b)+d; both x and y are 1D array.
    # sweep a and b, choose the best fitted model
    # on the device of x (pykan 0.2.8 built the (a, b) grid on `device`, 'cpu' by default, and failed for CUDA x)
    device = x.device
    # samples x grid_number^2 values per sweep; above max_elements, in chunks of samples (the same sums, accumulated)
    max_elements = 2**28
    chunk = max(1, max_elements // (grid_number * grid_number))
    @torch.no_grad()
    def sweep(a_range, b_range):
        a_ = torch.linspace(a_range[0], a_range[1], steps=grid_number, device=device)
        b_ = torch.linspace(b_range[0], b_range[1], steps=grid_number, device=device)
        a_grid, b_grid = torch.meshgrid(a_, b_, indexing='ij')
        y_mean = torch.mean(y, dim=[0], keepdim=True)
        if x.shape[0] <= chunk:
            y_c = (y - y_mean)[:, None, None]
            syy = torch.sum(y_c**2, dim=0)
            numerator, sxx = [], []
            for a_rows in torch.split(a_, _sweep_rows(x.shape[0], grid_number)):
                post_fun = fun(a_rows[None, :, None] * x[:, None, None] + b_[None, None, :])
                post_fun = post_fun - torch.mean(post_fun, dim=[0], keepdim=True)
                numerator.append(torch.sum(post_fun * y_c, dim=0)**2)
                sxx.append(torch.sum(post_fun * post_fun, dim=0))
            numerator, sxx = torch.cat(numerator), torch.cat(sxx)
        else:
            parts = [(x[s:s+chunk], y[s:s+chunk]) for s in range(0, x.shape[0], chunk)]
            x_mean = sum(torch.sum(fun(a_grid[None,:,:] * xs[:,None,None] + b_grid[None,:,:]), dim=0) for xs, _ys in parts)[None] / x.shape[0]
            cross, sq = 0., 0.
            for xs, ys in parts:
                pf = fun(a_grid[None,:,:] * xs[:,None,None] + b_grid[None,:,:]) - x_mean
                cross = cross + torch.sum(pf*(ys-y_mean)[:,None,None], dim=0)
                sq = sq + torch.sum(pf**2, dim=0)
            numerator = cross**2
            sxx, syy = sq, torch.sum((y - y_mean)**2, dim=0)
        r2 = _r2(numerator, sxx, syy, r2_eps, r2_tau, x.shape[0])
        return a_, b_, torch.nan_to_num(r2)

    def best(r2):
        best_id = torch.argmax(r2)
        return torch.div(best_id, grid_number, rounding_mode='floor'), best_id % grid_number

    if widen_ranges:
        # D16: on a boundary hit, a range extended on that side, kept only if it pays (pykan 0.2.8 narrowed into the
        # boundary cell, so a best a outside a_range was never reached, #529)
        _a, _b, r2 = sweep(a_range, b_range)
        for _w in range(_WIDEN_STEPS):
            a_id, b_id = best(r2)
            edges = (a_id == 0, a_id == grid_number - 1, b_id == 0, b_id == grid_number - 1)
            if not any(edges):
                break
            wa, wb = float(a_range[1]) - float(a_range[0]), float(b_range[1]) - float(b_range[0])
            new_a = (float(a_range[0]) - wa * edges[0], float(a_range[1]) + wa * edges[1])
            new_b = (float(b_range[0]) - wb * edges[2], float(b_range[1]) + wb * edges[3])
            _a, _b, r2_new = sweep(new_a, new_b)
            if not _widened(r2[a_id, b_id], r2_new.max()):
                break
            a_range, b_range, r2 = new_a, new_b, r2_new

    for _ in range(iteration):
        a_, b_, r2 = sweep(a_range, b_range)


        a_id, b_id = best(r2)
        
        
        if a_id == 0 or a_id == grid_number - 1 or b_id == 0 or b_id == grid_number - 1:
            if _ == 0 and verbose==True:
                print('Best value at boundary.')
            if a_id == 0:
                a_range = [a_[0], a_[1]]
            if a_id == grid_number - 1:
                a_range = [a_[-2], a_[-1]]
            if b_id == 0:
                b_range = [b_[0], b_[1]]
            if b_id == grid_number - 1:
                b_range = [b_[-2], b_[-1]]
            
        else:
            a_range = [a_[a_id-1], a_[a_id+1]]
            b_range = [b_[b_id-1], b_[b_id+1]]
            
    a_best = a_[a_id]
    b_best = b_[b_id]
    post_fun = fun(a_best * x + b_best)
    r2_best = r2[a_id, b_id]
    
    if verbose == True:
        print(f"r2 is {r2_best}")
        if r2_best < 0.9:
            print('r2 is not very high, please double check if you are choosing the correct symbolic function.')

    post_fun = torch.nan_to_num(post_fun)
    # ordinary least squares of y on f with an intercept in float64, as fit_params_batched (pykan 0.2.8: scikit-learn's
    # LinearRegression, on the CPU); a constant f gets slope 0
    f, yd = post_fun.detach().double(), y.detach().double()
    f_c = f - f.mean()
    sxx = (f_c ** 2).sum()
    c_best = (f_c * (yd - yd.mean())).sum() / sxx if float(sxx) > 0 else torch.zeros_like(sxx)
    d_best = yd.mean() - c_best * f.mean()
    return torch.stack([a_best, b_best, c_best.to(a_best.dtype), d_best.to(a_best.dtype)]), r2_best


def _linspace_rows(lo, hi, steps):
    '''torch.linspace(lo[e], hi[e], steps) for every row e, computed as torch.linspace computes it (from the start in
    the first half, from the end in the second), so the grid points are the same'''
    idx = torch.arange(steps, device=lo.device, dtype=lo.dtype)
    step = (hi - lo) / (steps - 1)
    from_start = lo[:, None] + step[:, None] * idx[None, :]
    from_end = hi[:, None] - step[:, None] * (steps - 1 - idx)[None, :]
    return torch.where(idx[None, :] < steps // 2, from_start, from_end)


def fit_params_batched(x, y, fun, a_range=(-10,10), b_range=(-10,10), grid_number=101, iteration=3, max_elements=2**27, r2_eps=1e-4,
                       r2_tau=None, widen_ranges=False):
    '''
    fit_params for many curves at once: for every row e, a, b, c, d such that |y[e] - (c*fun(a*x[e]+b)+d)|^2 is
    minimized, by the same sweep of a and b (zooming in iteration times) and the same linear fit of c and d

    Args:
    -----
        x : 2D torch.tensor
            x values, shape (number of curves, number of samples)
        y : 2D torch.tensor
            y values, shape (number of curves, number of samples)
        fun : function
            symbolic function (elementwise)
        a_range, b_range : tuple
            sweeping ranges of a and b: numbers (the same for all curves) or 1D tensors (one per curve)
        r2_tau : float or None
            if given, r2 is the exact r2 times var_f / (var_f + r2_tau), var_f the variance of fun(a*x+b) over the
            samples: a candidate used where it barely varies (e.g. cos(a x + b) with a tiny a, a quadratic in
            disguise) is penalised, the same for every amplitude of y (r2_eps's penalty depended on it)
        widen_ranges : bool
            if True, when the first sweep's best a or b lies on the boundary of its range, the range is extended on
            that side by its width (up to 5 times) as long as that raises the best r2 materially (by more than 1e-3
            of 1 - r2); otherwise, and for interior optima, the sweep is the default one
        r2_eps : float
            added to the denominator of r2 (1e-4, pykan 0.2.8's value, by default); it makes r2 depend on the
            amplitude of y. 0: the exact r2 (0 for a constant fun or y)
        grid_number : int
            number of steps along a and b
        iteration : int
            number of zooming in
        max_elements : int
            curves are swept in chunks of at most this many values (curves x samples x grid_number^2)

    Returns:
    --------
        params : 2D torch.tensor
            a, b, c, d per curve, shape (number of curves, 4)
        r2 : 1D torch.tensor
            r2 per curve

    Example
    -------
    >>> x = torch.linspace(-1,1,steps=100)[None,:].expand(3,-1)
    >>> y = torch.stack([5.0*torch.sin(3.0*x[0] + 2.0) + 0.7, torch.sin(x[1]), 2*torch.sin(-x[2])])
    >>> params, r2 = fit_params_batched(x, y, torch.sin)
    '''
    n_curve, n = x.shape
    chunk = max(1, int(max_elements // max(1, n * grid_number ** 2)))
    out_params, out_r2 = [], []
    for e0 in range(0, n_curve, chunk):
        xs, ys = x[e0:e0+chunk], y[e0:e0+chunk]
        m = xs.shape[0]
        per_curve = lambda v: torch.broadcast_to(torch.as_tensor(v, device=x.device, dtype=x.dtype), (n_curve,))[e0:e0+chunk].clone()
        a_lo, a_hi = per_curve(a_range[0]), per_curve(a_range[1])
        b_lo, b_hi = per_curve(b_range[0]), per_curve(b_range[1])
        rows = torch.arange(m, device=x.device)
        y_c = ys - torch.mean(ys, dim=1, keepdim=True)
        rows_per_block = _sweep_rows(m * n, grid_number)

        @torch.no_grad()
        def sweep(a_lo, a_hi, b_lo, b_hi):
            a_ = _linspace_rows(a_lo, a_hi, grid_number)  # (m, grid_number)
            b_ = _linspace_rows(b_lo, b_hi, grid_number)
            numerator, sxx = [], []
            for s in range(0, grid_number, rows_per_block):
                post_fun = fun(a_[:, None, s:s + rows_per_block, None] * xs[:, :, None, None] + b_[:, None, None, :])  # (m, n, rows, G)
                post_fun = post_fun - torch.mean(post_fun, dim=1, keepdim=True)
                numerator.append(torch.sum(post_fun * y_c[:, :, None, None], dim=1) ** 2)
                sxx.append(torch.sum(post_fun * post_fun, dim=1))
                del post_fun
            numerator, sxx = torch.cat(numerator, dim=1), torch.cat(sxx, dim=1)
            syy = torch.sum(y_c ** 2, dim=1)[:, None, None]
            return a_, b_, torch.nan_to_num(_r2(numerator, sxx, syy, r2_eps, r2_tau, n))  # (m, G, G)

        def best(r2):
            best_id = torch.argmax(r2.reshape(m, -1), dim=1)
            return torch.div(best_id, grid_number, rounding_mode='floor'), best_id % grid_number

        if widen_ranges:
            # D16, per curve as in fit_params: a range extended on the side of a boundary hit while that pays
            _a, _b, r2 = sweep(a_lo, a_hi, b_lo, b_hi)
            active = torch.ones(m, dtype=torch.bool, device=x.device)
            for _w in range(_WIDEN_STEPS):
                a_id, b_id = best(r2)
                e_a0, e_a1, e_b0, e_b1 = a_id == 0, a_id == grid_number - 1, b_id == 0, b_id == grid_number - 1
                edge = (e_a0 | e_a1 | e_b0 | e_b1) & active
                if not edge.any():
                    break
                wa, wb = a_hi - a_lo, b_hi - b_lo
                na_lo, na_hi = torch.where(edge & e_a0, a_lo - wa, a_lo), torch.where(edge & e_a1, a_hi + wa, a_hi)
                nb_lo, nb_hi = torch.where(edge & e_b0, b_lo - wb, b_lo), torch.where(edge & e_b1, b_hi + wb, b_hi)
                _a, _b, r2_new = sweep(na_lo, na_hi, nb_lo, nb_hi)
                accept = edge & _widened(r2[rows, a_id, b_id], r2_new.reshape(m, -1).max(dim=1).values)
                a_lo, a_hi = torch.where(accept, na_lo, a_lo), torch.where(accept, na_hi, a_hi)
                b_lo, b_hi = torch.where(accept, nb_lo, b_lo), torch.where(accept, nb_hi, b_hi)
                r2 = torch.where(accept[:, None, None], r2_new, r2)
                active = accept

        for _ in range(iteration):
            a_, b_, r2 = sweep(a_lo, a_hi, b_lo, b_hi)
            a_id, b_id = best(r2)
            # as fit_params: at the boundary only the parameter at the boundary moves (to the edge interval);
            # otherwise both zoom in to the neighbours of the best point
            a0, a1, b0, b1 = a_id == 0, a_id == grid_number - 1, b_id == 0, b_id == grid_number - 1
            boundary = a0 | a1 | b0 | b1
            inner_a = (a_id - 1).clamp(0, grid_number - 1), (a_id + 1).clamp(0, grid_number - 1)
            inner_b = (b_id - 1).clamp(0, grid_number - 1), (b_id + 1).clamp(0, grid_number - 1)
            new_a_lo = torch.where(boundary, torch.where(a0, a_[:, 0], torch.where(a1, a_[:, -2], a_lo)), a_[rows, inner_a[0]])
            new_a_hi = torch.where(boundary, torch.where(a0, a_[:, 1], torch.where(a1, a_[:, -1], a_hi)), a_[rows, inner_a[1]])
            new_b_lo = torch.where(boundary, torch.where(b0, b_[:, 0], torch.where(b1, b_[:, -2], b_lo)), b_[rows, inner_b[0]])
            new_b_hi = torch.where(boundary, torch.where(b0, b_[:, 1], torch.where(b1, b_[:, -1], b_hi)), b_[rows, inner_b[1]])
            a_lo, a_hi, b_lo, b_hi = new_a_lo, new_a_hi, new_b_lo, new_b_hi
        a_best, b_best = a_[rows, a_id], b_[rows, b_id]
        r2_best = r2[rows, a_id, b_id]
        f = torch.nan_to_num(fun(a_best[:, None] * xs + b_best[:, None])).double()
        # ordinary least squares of y on f with an intercept (sklearn's LinearRegression in fit_params); a constant f
        # gets slope 0
        f_c = f - f.mean(dim=1, keepdim=True)
        sxx = (f_c ** 2).sum(dim=1)
        c_best = torch.where(sxx > 0, (f_c * y_c.double()).sum(dim=1) / torch.where(sxx > 0, sxx, torch.ones_like(sxx)), torch.zeros_like(sxx))
        d_best = ys.double().mean(dim=1) - c_best * f.mean(dim=1)
        out_params.append(torch.stack([a_best, b_best, c_best.to(x.dtype), d_best.to(x.dtype)], dim=1))
        out_r2.append(r2_best)
    return torch.cat(out_params), torch.cat(out_r2)


def _symbolic_fit_inputs(x, a_range, input_standardize, a_range_relative):
    '''
    the samples and the a range of the symbolic fit of an edge, x of shape (n,), or of a batch of edges, (edges, n):
    with input_standardize the samples standardized by their 1% and 99% quantiles (center, spread; the fitted affine
    parameters go back through _symbolic_unstandardize), else with a_range_relative True the a range divided by the
    largest |x|, with 'widen' divided by it only when it is below 1 (widened for small inputs, never narrowed). For
    one edge the range and center / spread are Python floats, as fix_symbolic always computed them.

    Returns:
    --------
        x_fit, a_range, center, spread (center and spread None without input_standardize)
    '''
    one = x.dim() == 1
    if input_standardize:
        q = torch.quantile(x, torch.tensor([0.01, 0.99], dtype=x.dtype, device=x.device), dim=-1)
        center, spread = (q[0] + q[1]) / 2, ((q[1] - q[0]) / 2).clamp(min=1e-6)
        if one:
            center, spread = float(center), float(spread)
            return (x - center) / spread, a_range, center, spread
        return (x - center[:, None]) / spread[:, None], a_range, center, spread
    if a_range_relative not in (True, False, "widen"):
        raise ValueError(f"a_range_relative must be True, False or 'widen', got {a_range_relative!r}")
    if a_range_relative:
        scale = x.abs().amax(dim=-1).clamp(min=1e-6)
        if a_range_relative == "widen":
            scale = scale.clamp(max=1.)
        scale = float(scale) if one else scale
        return x, (a_range[0] / scale, a_range[1] / scale), None, None
    return x, a_range, None, None


def _symbolic_unstandardize(a, b, center, spread):
    '''
    the affine parameters of f(a x + b) = f(a' (x - center) / spread + b') from a', b' (fitted on standardized samples)
    '''
    return a / spread, b - a * center / spread


# real-valued sympy forms of the library functions that the singularity-avoiding forward evaluates for negative
# arguments (utils.f_sqrt, f_invsqrt, f_log, f_power1d5), for symbolic_formula(real=True)
REAL_SYMPY = {
    'sqrt': lambda x: sympy.sign(x) * sympy.sqrt(sympy.Abs(x)),
    'x^0.5': lambda x: sympy.sign(x) * sympy.sqrt(sympy.Abs(x)),
    '1/sqrt(x)': lambda x: 1 / sympy.sqrt(sympy.Abs(x)),
    '1/x^0.5': lambda x: 1 / sympy.sqrt(sympy.Abs(x)),
    'log': lambda x: sympy.log(sympy.Abs(x)),
    'x^1.5': lambda x: sympy.Abs(x) ** sympy.Rational(3, 2),
}
