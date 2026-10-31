import torch


def B_batch(x, grid, k=0):
    '''
    evaludate x on B-spline bases
    
    Args:
    -----
        x : 2D torch.tensor
            inputs, shape (number of splines, number of samples)
        grid : 2D torch.tensor
            grids, shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines; the grid must already be extended by k points on both ends
            (extend_grid)
    
    Returns:
    --------
        spline values : 3D torch.tensor
            shape (batch, in_dim, G+k). G: the number of grid intervals, k: spline order.
      
    Example
    -------
    >>> from xkan.spline import B_batch
    >>> x = torch.rand(100,2)
    >>> grid = torch.linspace(-1,1,steps=11)[None, :].expand(2, 11)
    >>> B_batch(x, grid, k=3).shape
    '''
    
    x = x.unsqueeze(dim=2)
    grid = grid.unsqueeze(dim=0)
    
    if k == 0:
        value = (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])
    else:
        B_km1 = B_batch(x[:,:,0], grid=grid[0], k=k - 1)
        
        # a zero-width knot span (coincident knots, e.g. a grid updated from tied or constant samples) belongs to a
        # basis of order k-1 that is zero everywhere, so its term is 0 (the usual convention 0/0 = 0). Dividing by 1
        # there instead of 0 gives that term without the NaN, which made the gradient NaN and, through the sum,
        # zeroed the other, valid term of the basis.
        left = grid[:, :, k:-1] - grid[:, :, :-(k + 1)]
        right = grid[:, :, k + 1:] - grid[:, :, 1:(-k)]
        left = torch.where(left == 0, torch.ones_like(left), left)
        right = torch.where(right == 0, torch.ones_like(right), right)
        value = (x - grid[:, :, :-(k + 1)]) / left * B_km1[:, :, :-1] + (grid[:, :, k + 1:] - x) / right * B_km1[:, :, 1:]
    
    # in case grid is degenerate
    value = torch.nan_to_num(value)
    return value



def coef2curve(x_eval, grid, coef, k):
    '''
    converting B-spline coefficients to B-spline curves. Evaluate x on B-spline curves (summing up B_batch results over B-spline basis).
    
    Args:
    -----
        x_eval : 2D torch.tensor
            shape (batch, in_dim)
        grid : 2D torch.tensor
            shape (in_dim, G+2k). G: the number of grid intervals; k: spline order.
        coef : 3D torch.tensor
            shape (in_dim, out_dim, G+k)
        k : int
            the piecewise polynomial order of splines.
        
    Returns:
    --------
        y_eval : 3D torch.tensor
            shape (batch, in_dim, out_dim)
        
    '''
    
    b_splines = B_batch(x_eval, grid, k=k)
    y_eval = torch.einsum('ijk,jlk->ijl', b_splines, coef.to(b_splines.device))
    
    return y_eval


def curve2coef(x_eval, y_eval, grid, k):
    '''
    converting B-spline curves to B-spline coefficients using least squares.
    
    Args:
    -----
        x_eval : 2D torch.tensor
            shape (batch, in_dim)
        y_eval : 3D torch.tensor
            shape (batch, in_dim, out_dim)
        grid : 2D torch.tensor
            shape (in_dim, grid+2*k)
        k : int
            spline order
            
    Returns:
    --------
        coef : 3D torch.tensor
            shape (in_dim, out_dim, G+k)
    '''
    # one design matrix per input, shared by all its outputs (solved with out_dim right-hand sides): the same solutions
    # as expanding it to (in_dim, out_dim, batch, n_coef), without out_dim copies of it
    mat = B_batch(x_eval, grid, k).permute(1,0,2)  # (in_dim, batch, n_coef)
    y_eval = y_eval.permute(1,0,2)  # (in_dim, batch, out_dim)
    # contiguous: the coefficients become a parameter, and LBFGS flattens its gradient with view(-1)
    coef = solve_lstsq(mat, y_eval).permute(0,2,1).contiguous()  # (in_dim, out_dim, n_coef)
    return coef


def solve_lstsq(mat, y_eval, singular=None):
    '''
    least-squares solutions of the systems mat @ coef = y_eval, one per input

    Args:
    -----
        mat : 3D torch.tensor
            design matrices, shape (in_dim, batch, n_coef)
        y_eval : 3D torch.tensor
            right-hand sides, shape (in_dim, batch, out_dim)
        singular : None or 1D bool torch.tensor
            the systems to re-solve with the minimum-norm SVD solution (see resolve_rank_deficient); None, found
            here. Passing it saves recomputing it when the same design matrices are solved for several right-hand sides.

    Returns:
    --------
        coef : 3D torch.tensor
            shape (in_dim, n_coef, out_dim)
    '''
    finite = torch.isfinite(y_eval)
    if not bool(finite.all()):
        # a right-hand side with a non-finite value (the curves of a diverged model) has a NaN solution; it is not
        # passed to LAPACK, whose CPU driver rejects NaN with an error message (MKL: "Parameter 6 was incorrect on
        # entry to SGELSY"). The other right-hand sides are solved as before.
        bad = ~finite.all(dim=1, keepdim=True)  # (in_dim, 1, out_dim)
        return solve_lstsq(mat, torch.where(finite, y_eval, torch.zeros_like(y_eval)), singular).masked_fill(bad, float('nan'))
    try:
        coef = torch.linalg.lstsq(mat, y_eval).solution
        coef = resolve_rank_deficient(mat, y_eval, coef, singular=singular)
    except Exception:
        # the solver can fail (e.g. LAPACK on a rank-deficient system); fall back to the minimum-norm least-squares
        # solution instead of leaving coef undefined
        coef = min_norm_solution(mat, y_eval)
        _REFIT_COUNTS["systems"] += mat.shape[0]
        _REFIT_COUNTS["resolved"] += mat.shape[0]
    return coef


# Least-squares systems (one per input) solved by solve_lstsq since import or the last reset, and those re-solved by the
# minimum-norm SVD solution (rank-deficient or non-finite): a diagnostic of how often refits meet singular systems.
_REFIT_COUNTS = {"systems": 0, "resolved": 0}


def refit_counts(reset=False):
    '''
    the least-squares systems solved (one per input of a fit or refit) and those re-solved as rank-deficient or
    non-finite, since import or the last reset

    Args:
    -----
        reset : bool
            set both counts to 0 after reading them

    Returns:
    --------
        dict with "systems" and "resolved"
    '''
    counts = dict(_REFIT_COUNTS)
    if reset:
        _REFIT_COUNTS.update(systems=0, resolved=0)
    return counts


# Largest number of basis values (rows x inputs x (old + new coefficients)) a grid refit evaluates and solves at once:
# a refit of a wide layer solves its inputs in blocks (independent systems), so its memory stays bounded. With fewer
# values everything is solved at once.
REFIT_MAX_ELEMENTS = 2 ** 28


def refit_coef(x_eval, grid, coef, k, new_grid, new_k=None, max_elements=None):
    '''
    coefficients on new_grid of the spline curves (grid, coef, k), fitted by least squares at x_eval

    The fit is linear in the curves: the new coefficients are T @ coef, with T (in_dim, G'+k', G+k) the least-squares
    fit of the old basis functions themselves on the new ones. That is G+k right-hand sides per input instead of one
    per output curve: memory O(batch * in_dim * (G+k)) instead of O(batch * in_dim * out_dim) (pykan 0.2.8 and the
    first revisions fitted the curves; the same solution up to rounding, the rank decisions being those of the new
    design matrix). Inputs are taken in blocks of at most max_elements basis values (default REFIT_MAX_ELEMENTS).

    Args:
    -----
        x_eval : 2D torch.tensor
            shape (batch, in_dim)
        grid : 2D torch.tensor
            grid of the curves, shape (in_dim, G+2k+1)
        coef : 3D torch.tensor
            coefficients of the curves, shape (in_dim, out_dim, G+k)
        k : int
            spline order of the curves
        new_grid : 2D torch.tensor
            the new grid, shape (in_dim, G'+2k'+1)
        new_k : None or int
            spline order on the new grid; None, k
        max_elements : None or int
            block size in basis values; None, REFIT_MAX_ELEMENTS

    Returns:
    --------
        coef : 3D torch.tensor
            shape (in_dim, out_dim, G'+k')
    '''
    new_k = k if new_k is None else new_k
    max_elements = REFIT_MAX_ELEMENTS if max_elements is None else max_elements
    batch, in_dim = x_eval.shape
    n_old, n_new = grid.shape[1] - 1 - k, new_grid.shape[1] - 1 - new_k
    rows = max(1, int(max_elements // max(1, batch * (n_old + n_new))))
    blocks = []
    for i0 in range(0, in_dim, rows):
        i1 = min(in_dim, i0 + rows)
        x = x_eval[:, i0:i1]
        mat = B_batch(x, new_grid[i0:i1], new_k).permute(1, 0, 2)  # (inputs, batch, n_new)
        cutoff = torch.finfo(mat.dtype).eps if mat.is_cuda else rank_cutoff(mat)
        old = B_batch(x, grid[i0:i1], k).permute(1, 0, 2)  # (inputs, batch, n_old)
        transfer = solve_lstsq(mat, old, singular=numerically_singular(mat, cutoff))  # (inputs, n_new, n_old)
        blocks.append(torch.einsum('inm,iom->ion', transfer, coef[i0:i1].to(transfer.dtype)))
    return torch.cat(blocks, dim=0).contiguous()


def _float64_device(t):
    '''
    device for the float64 linear algebra on t: its own, or the CPU where float64 is unsupported (e.g. MPS)
    '''
    return t.device if t.device.type in ('cpu', 'cuda') else torch.device('cpu')


def rank_cutoff(mat):
    '''
    relative singular-value cutoff of the minimum-norm SVD solution: the working precision times the larger dimension
    of the system, as torch.linalg.lstsq's default rcond (the truncation of gelsy, its CPU driver)
    '''
    return torch.finfo(mat.dtype).eps * max(mat.shape[-2:])


def numerically_singular(mat, cutoff=None):
    '''
    design matrices whose smallest singular value is at most cutoff times the largest (default rank_cutoff), from the
    singular values of their R factor in double precision

    Args:
    -----
        mat : 3D torch.tensor
            shape (in_dim, batch, n_coef)
        cutoff : None or float
            relative cutoff; None, rank_cutoff(mat)

    Returns:
    --------
        1D bool torch.tensor, shape (in_dim,)
    '''
    if cutoff is None:
        cutoff = rank_cutoff(mat)
    r = torch.linalg.qr(mat.to(_float64_device(mat)).double(), mode='r').R
    sv = torch.linalg.svdvals(r)
    return (sv[..., -1] <= cutoff * sv[..., 0]).to(mat.device)


def min_norm_solution(mat, y_eval):
    '''
    minimum-norm least-squares solution from the SVD in double precision, singular values below rank_cutoff dropped,
    on the device of mat

    Args:
    -----
        mat : 3D torch.tensor
            shape (in_dim, batch, n_coef)
        y_eval : 3D torch.tensor
            shape (in_dim, batch, out_dim)

    Returns:
    --------
        coef : 3D torch.tensor
            shape (in_dim, n_coef, out_dim)
    '''
    dev = _float64_device(mat)
    coef = torch.linalg.pinv(mat.to(dev).double(), rtol=rank_cutoff(mat)) @ y_eval.to(dev).double()
    return coef.to(device=mat.device, dtype=mat.dtype)


def resolve_rank_deficient(mat, y_eval, coef, singular=None):
    '''
    re-solve the rank-deficient systems of a least-squares solution with the minimum-norm SVD solution

    A B-spline design matrix is rank-deficient when bases have no support among the samples (e.g. a grid updated from
    tied samples), and ill-conditioned when they have little (e.g. sparse inputs). The LAPACK drivers that lstsq uses
    by default handle these differently:
    - CUDA's gels (QR without pivoting) assumes full rank: on a system singular to working precision it returns NaN
      or finite values of order 1e20, so the spline later turns into NaN (issues #233, #498); on a merely
      ill-conditioned system it is accurate. On CUDA, the systems singular to working precision (cutoff eps) are
      re-solved.
    - The CPU's gelsy truncates singular values below rank_cutoff, and some builds misjudge the rank of the systems it
      truncates (torch 2.10's MKL cut rank-deficient and ill-conditioned systems to rank 1, so the refit no longer
      reproduced the curves at the samples). On the CPU, every system below rank_cutoff is re-solved.
    The re-solve is the minimum-norm solution from the SVD at rank_cutoff, as gelsy intends; any non-finite solution
    is re-solved as well. Systems singular to working precision get the same solution on every device.

    Args:
    -----
        mat : 3D torch.tensor
            shape (in_dim, batch, n_coef)
        y_eval : 3D torch.tensor
            shape (in_dim, batch, out_dim)
        coef : 3D torch.tensor
            the lstsq solution, shape (in_dim, n_coef, out_dim)
        singular : None or 1D bool torch.tensor
            the systems below the device's cutoff, if already known

    Returns:
    --------
        coef : 3D torch.tensor
            shape (in_dim, n_coef, out_dim)
    '''
    if singular is None:
        cutoff = torch.finfo(mat.dtype).eps if mat.is_cuda else rank_cutoff(mat)
        singular = numerically_singular(mat, cutoff)
    bad = singular | ~torch.isfinite(coef).all(dim=2).all(dim=1)
    idx = torch.nonzero(bad).flatten()  # synchronises once, as bad.any() did
    _REFIT_COUNTS["systems"] += bad.numel()
    _REFIT_COUNTS["resolved"] += idx.numel()
    if idx.numel():
        coef = coef.clone()
        coef[idx] = min_norm_solution(mat[idx], y_eval[idx])
    return coef


def nonzero_quantile_knots(x_pos, num_interval, grid_eps, grid):
    '''
    knots of a grid update that respect exact zeros: for an input with zeros, non-zero values and no negative values,
    one knot at 0 and num_interval knots at quantiles of its non-zero values (blended with uniform knots over them by
    grid_eps); the other inputs keep the knots in grid

    Args:
    -----
        x_pos : 2D torch.tensor
            samples sorted along dim 0, shape (batch, in_dim)
        num_interval : int
            number of grid intervals
        grid_eps : float
            weight of the uniform knots
        grid : 2D torch.tensor
            the knots of the other inputs, shape (in_dim, num_interval+1)

    Returns:
    --------
        2D torch.tensor, shape (in_dim, num_interval+1)
    '''
    batch, in_dim = x_pos.shape
    n_zero = (x_pos == 0).sum(dim=0)
    n_nz = batch - n_zero
    use = (n_zero > 0) & (n_nz > 0) & (x_pos[0] >= 0)  # sorted: the zeros come first
    # num_interval knots over the non-zero values; a single one goes to the largest value, so that the interval
    # [0, knot] spans them
    if num_interval > 1:
        ranks = torch.linspace(0, 1, num_interval, device=x_pos.device, dtype=x_pos.dtype)
    else:
        ranks = torch.ones(1, device=x_pos.device, dtype=x_pos.dtype)
    idx = n_zero[:, None] + torch.round(ranks[None, :] * (n_nz - 1).clamp(min=0)[:, None].to(x_pos.dtype)).long()
    adaptive = torch.gather(x_pos.permute(1,0), 1, idx.clamp(0, batch - 1))
    uniform = adaptive[:, :1] + (x_pos[-1][:, None] - adaptive[:, :1]) * ranks[None, :]
    knots = torch.cat([torch.zeros(in_dim, 1, device=x_pos.device, dtype=x_pos.dtype),
                       grid_eps * uniform + (1 - grid_eps) * adaptive], dim=1)
    return torch.where(use[:, None], knots, grid)


def separate_knots(grid, gap):
    '''
    knots at least gap apart: each knot is moved right to at least the previous knot plus gap

    Args:
    -----
        grid : 2D torch.tensor
            knots, shape (in_dim, n)
        gap : 1D torch.tensor or float
            smallest gap, per input (shape (in_dim,)) or for all

    Returns:
    --------
        2D torch.tensor, shape (in_dim, n)
    '''
    gap = torch.as_tensor(gap, dtype=grid.dtype, device=grid.device).expand(grid.shape[0])
    cols = [grid[:, 0]]
    for j in range(1, grid.shape[1]):
        cols.append(torch.maximum(grid[:, j], cols[-1] + gap))
    return torch.stack(cols, dim=1)


def knot_range(grid, k, num):
    '''
    per input, the interior knot range [grid[:, k], grid[:, num + k]] of a grid of num intervals and order k
    '''
    return grid[:, k], grid[:, num + k]


def rbf_bases(x, grid, k, num):
    '''
    Gaussian radial basis functions: num + k centres evenly spaced over the interior knot range of each input, width
    equal to the centre spacing (at least 1e-3). Returns (batch, in_dim, num + k), the shape of B_batch, so that the
    coefficients and the grid updates of the B-spline layer apply.
    '''
    n = num + k
    lo, hi = knot_range(grid, k, num)
    t = torch.linspace(0.0, 1.0, n, device=grid.device, dtype=grid.dtype)
    centres = lo.unsqueeze(1) + (hi - lo).unsqueeze(1) * t.unsqueeze(0)  # (in_dim, n)
    width = ((hi - lo) / max(1, n - 1)).clamp(min=1e-3).unsqueeze(1)  # (in_dim, 1)
    return torch.exp(-(((x.unsqueeze(-1) - centres.unsqueeze(0)) / width.unsqueeze(0)) ** 2))


def rswaf_bases(x, grid, k, num):
    '''
    Reflectional switch activation functions (FasterKAN), 1 - tanh((x - c) / h)^2 = sech((x - c) / h)^2: num + k centres
    c and width h as rbf_bases (evenly over the interior knot range, h the centre spacing). Like the Gaussians they
    approximate cubic B-splines, and cost one tanh per value. Returns (batch, in_dim, num + k).
    '''
    n = num + k
    lo, hi = knot_range(grid, k, num)
    t = torch.linspace(0.0, 1.0, n, device=grid.device, dtype=grid.dtype)
    centres = lo.unsqueeze(1) + (hi - lo).unsqueeze(1) * t.unsqueeze(0)  # (in_dim, n)
    width = ((hi - lo) / max(1, n - 1)).clamp(min=1e-3).unsqueeze(1)  # (in_dim, 1)
    return 1.0 - torch.tanh((x.unsqueeze(-1) - centres.unsqueeze(0)) / width.unsqueeze(0)) ** 2


def chebyshev_bases(x, grid, k, num):
    '''
    Chebyshev polynomials T_0 .. T_{num + k - 1} of tanh((x - mid) / half), the input squashed onto (-1, 1) by the
    interior knot range of its dimension (half at least 1e-3). Returns (batch, in_dim, num + k).
    '''
    n = num + k
    lo, hi = knot_range(grid, k, num)
    mid, half = (hi + lo) / 2.0, ((hi - lo) / 2.0).clamp(min=1e-3)
    u = torch.tanh((x - mid.unsqueeze(0)) / half.unsqueeze(0))
    polys = [torch.ones_like(u), u]
    for _ in range(2, n):
        polys.append(2.0 * u * polys[-1] - polys[-2])
    return torch.stack(polys[:n], dim=-1)


def refit_coef_bases(bases_old, bases_new, coef, max_elements=None):
    '''
    coefficients for the basis values bases_new (batch, in_dim, n_new) of the curves bases_old (batch, in_dim, n_old)
    with coefficients coef (in_dim, out_dim, n_old), by least squares: the refit of refit_coef for any basis family,
    through the fit T of the old basis functions on the new ones (see refit_coef), in blocks of inputs of at most
    max_elements basis values (default REFIT_MAX_ELEMENTS).

    Solved in float64 (on the CPU for devices without it), the result in coef's dtype: the design matrices of the
    global bases are worse conditioned than the B-splines', and in float32 the solver's rank cutoff (singular values
    below eps * max(batch, n_new) of the largest) drops part of the curves (15 Chebyshev polynomials at 1000 samples:
    2e-3 of the curve; 3e-7 in float64).
    '''
    max_elements = REFIT_MAX_ELEMENTS if max_elements is None else max_elements
    batch, in_dim, n_old = bases_old.shape
    n_new = bases_new.shape[2]
    dev = _float64_device(coef)
    rows = max(1, int(max_elements // max(1, batch * (n_old + n_new))))
    blocks = []
    for i0 in range(0, in_dim, rows):
        i1 = min(in_dim, i0 + rows)
        mat = bases_new[:, i0:i1].permute(1, 0, 2).to(dev).double()  # (inputs, batch, n_new)
        old = bases_old[:, i0:i1].permute(1, 0, 2).to(dev).double()  # (inputs, batch, n_old)
        transfer = solve_lstsq(mat, old)  # (inputs, n_new, n_old)
        blocks.append(torch.einsum('inm,iom->ion', transfer, coef[i0:i1].to(dev).double()))
    return torch.cat(blocks, dim=0).to(device=coef.device, dtype=coef.dtype)


def zero_aware_grid(x, num, k, grid_eps=0.02):
    '''
    knot grid (in_dim, num + 2k + 1) from samples x (batch, in_dim) for zero-inflated inputs:
    percentiles blended with a uniform grid (grid_eps); for a column with zeros, non-zero values and no negatives,
    a knot at 0 and the others at percentiles of the non-zero values; a minimum knot separation of 1e-3 of the
    column's range (at least 1.001e-4); the k outer knots on each side spaced by the first / last interval
    '''
    if x.dim() != 2:
        raise ValueError(f"zero_aware_grid requires a 2D tensor of shape (batch, in_dim), got shape {tuple(x.shape)}")
    if x.shape[0] == 0:
        raise ValueError("zero_aware_grid cannot compute quantiles for an empty batch (batch_size=0).")
    device, dtype = x.device, x.dtype
    batch, in_dim = x.shape
    G = int(num)
    x_sorted, _ = torch.sort(x, dim=0)
    col_min = x_sorted[0]
    col_max = x_sorted[-1]
    delta_min = torch.clamp(1e-3 * (col_max - col_min), min=1.001e-4)
    n_zero = (x == 0).sum(dim=0)
    n_nz = batch - n_zero
    use_nz = (n_zero > 0) & (n_nz > 0) & (col_min >= 0)
    # with one interior knot (G == 1) it sits at the column maximum, so that the interior spans [0, max]
    if G > 1:
        ranks_g = torch.linspace(0.0, 1.0, G, device=device, dtype=dtype)
    else:
        ranks_g = torch.ones(1, device=device, dtype=dtype)
    idx_nz = n_zero.unsqueeze(1) + torch.round(ranks_g.unsqueeze(0) * (n_nz - 1).clamp(min=0).unsqueeze(1).to(dtype)).long()
    idx_nz = idx_nz.clamp(0, batch - 1)
    adaptive_nz = torch.gather(x_sorted.t(), 1, idx_nz)
    uniform_nz = adaptive_nz[:, :1] + (col_max.unsqueeze(1) - adaptive_nz[:, :1]) * ranks_g.unsqueeze(0)
    blended_nz = grid_eps * uniform_nz + (1.0 - grid_eps) * adaptive_nz
    interior_nz = torch.cat([torch.zeros(in_dim, 1, device=device, dtype=dtype), blended_nz], dim=1)
    ranks_all = torch.linspace(0.0, 1.0, G + 1, device=device, dtype=dtype)
    idx_all = torch.round(ranks_all * (batch - 1)).long().clamp(0, batch - 1)
    adaptive_all = x_sorted[idx_all].t()
    uniform_all = col_min.unsqueeze(1) + (col_max - col_min).unsqueeze(1) * ranks_all.unsqueeze(0)
    interior_all = grid_eps * uniform_all + (1.0 - grid_eps) * adaptive_all
    interior_all[:, 0] = col_min
    interior = torch.where(use_nz.unsqueeze(1), interior_nz, interior_all)
    for j in range(1, G + 1):
        interior[:, j] = torch.maximum(interior[:, j], interior[:, j - 1] + delta_min)
    step_left = torch.clamp(interior[:, 1] - interior[:, 0], min=1.001e-4).unsqueeze(1)
    step_right = torch.clamp(interior[:, -1] - interior[:, -2], min=1.001e-4).unsqueeze(1)
    k_steps = torch.arange(k, 0, -1, device=device, dtype=dtype).unsqueeze(0)
    left_knots = interior[:, :1] - step_left * k_steps
    right_knots = interior[:, -1:] + step_right * k_steps.flip(dims=[1])
    full_grid = torch.cat([left_knots, interior, right_knots], dim=1)
    for j in range(1, full_grid.shape[1]):
        full_grid[:, j] = torch.maximum(full_grid[:, j], full_grid[:, j - 1] + 1.001e-4)
    return full_grid


def extend_grid(grid, k_extend=0):
    '''
    extend grid
    '''
    h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

    for i in range(k_extend):
        grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
        grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)

    return grid