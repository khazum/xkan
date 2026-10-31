"""Least-squares fits of spline coefficients (curve2coef, refit_coef, solve_lstsq): one design matrix per input,
minimum-norm solutions of rank-deficient systems on every device, a failing LAPACK call, ill-conditioned systems, refits
in blocks of inputs; B-spline bases at coincident knots."""
import pytest
import torch

from xkan import spline as REV

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def problem(batch=200, in_dim=7, out_dim=5, num=5, k=3, dtype=torch.float64, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    grid = REV.extend_grid(torch.linspace(-1, 1, num + 1, dtype=dtype)[None, :].expand(in_dim, num + 1), k_extend=k)
    x = torch.rand(batch, in_dim, generator=g, dtype=dtype) * 2 - 1
    y = torch.randn(batch, in_dim, out_dim, generator=g, dtype=dtype)
    return x.to(device), y.to(device), grid.to(device), k


def tied_problem(device):
    """a grid refit from tied samples: 30 of 32 samples at 0, so most bases have no support (rank-deficient)"""
    torch.manual_seed(0)
    in_dim, out_dim, k = 20, 4, 3
    x = torch.zeros(32, in_dim)
    x[:2] = torch.rand(2, in_dim)
    grid = REV.extend_grid(torch.linspace(-1, 1, 6)[None, :].expand(in_dim, 6), k_extend=k)
    y = torch.randn(32, in_dim, out_dim)
    return x.to(device), y.to(device), grid.to(device), k


def exact(x, y, grid, k):
    """the minimum-norm least-squares solution in double precision, (in_dim, out_dim, n_coef)"""
    mat = REV.B_batch(x, grid, k).permute(1, 0, 2)
    cut = torch.finfo(mat.dtype).eps * max(mat.shape[-2:])
    return (torch.linalg.pinv(mat.double(), rtol=cut) @ y.permute(1, 0, 2).double()).permute(0, 2, 1)


def residual(x, y, grid, k, coef):
    """largest squared residual over the inputs of a solution of curve2coef's systems"""
    mat = REV.B_batch(x, grid, k).permute(1, 0, 2).double()
    fit = mat @ coef.double().permute(0, 2, 1)
    return float(((fit - y.permute(1, 0, 2).double()) ** 2).sum(dim=(1, 2)).max())


@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-10), (torch.float32, 1e-4)])
def test_curve2coef_is_the_least_squares_solution(dtype, tol):
    x, y, grid, k = problem(dtype=dtype)
    coef = REV.curve2coef(x, y, grid, k)
    assert coef.shape == (7, 5, 8) and coef.dtype == dtype
    assert float((coef.double() - exact(x, y, grid, k)).abs().max()) < tol


def test_lstsq_failure_falls_back(monkeypatch):
    # a failing solver no longer leaves the coefficients undefined (pykan 0.2.8: UnboundLocalError)
    x, y, grid, k = problem()

    def boom(*args, **kwargs):
        raise RuntimeError("simulated LAPACK failure")
    monkeypatch.setattr(torch.linalg, "lstsq", boom)
    coef = REV.curve2coef(x, y, grid, k)
    monkeypatch.undo()
    assert torch.isfinite(coef).all()
    assert float((coef - REV.curve2coef(x, y, grid, k)).abs().max()) < 1e-8


def test_rank_deficient_cpu_is_finite():
    x, y, grid, k = tied_problem("cpu")
    assert REV.numerically_singular(REV.B_batch(x, grid, k).permute(1, 0, 2)).all()
    coef = REV.curve2coef(x, y, grid, k)
    assert torch.isfinite(coef).all() and float(coef.abs().max()) < 1e3


@pytest.mark.parametrize("device", DEVICES)
def test_refit_counts(device):
    # every input's system is counted; the tied ones as re-solved, the well-posed ones not
    REV.refit_counts(reset=True)
    x, y, grid, k = tied_problem(device)
    REV.curve2coef(x, y, grid, k)
    assert REV.refit_counts() == {"systems": 20, "resolved": 20}
    x, y, grid, k = problem(device=device)
    REV.curve2coef(x, y, grid, k)
    assert REV.refit_counts(reset=True) == {"systems": 27, "resolved": 20}
    assert REV.refit_counts() == {"systems": 0, "resolved": 0}


@pytest.mark.parametrize("device", DEVICES)
def test_rank_deficient_solution_is_least_squares(device):
    # a tied refit whose targets are curves of another grid: the solution reaches the exact least-squares residual
    x, _, grid, k = tied_problem(device)
    old_grid = REV.extend_grid(torch.linspace(-1, 1, 4)[None, :].expand(x.shape[1], 4), k_extend=2).to(device)
    old_coef = torch.randn(x.shape[1], 4, 5, generator=torch.Generator().manual_seed(1)).to(device) * 0.25
    y = REV.coef2curve(x, old_grid, old_coef, 2)
    best = residual(x, y, grid, k, exact(x, y, grid, k))
    assert residual(x, y, grid, k, REV.curve2coef(x, y, grid, k)) <= best + 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_rank_deficient_cuda_matches_cpu():
    # CUDA's gels returns NaN or 1e20 on rank-deficient systems; they are re-solved by the SVD
    x, y, grid, k = tied_problem("cuda")
    rev = REV.curve2coef(x, y, grid, k)
    cpu = REV.curve2coef(x.cpu(), y.cpu(), grid.cpu(), k)
    assert torch.isfinite(rev).all() and float((rev.cpu() - cpu).abs().max()) < 1e-3
    # a well-posed CUDA problem keeps the CUDA solution
    x2, y2, grid2, k2 = problem(dtype=torch.float32, device="cuda")
    assert float((REV.curve2coef(x2, y2, grid2, k2).double() - exact(x2, y2, grid2, k2)).abs().max()) < 1e-4


def sparse_problem(device, n=2000, in_dim=40, frac=0.03, k=3, num=5):
    """an ill-conditioned but full-rank refit: sparse log-normal inputs (3% non-zero) on knots at the quantiles of the
    non-zero values, targets that are curves of that grid (an exact fit exists)"""
    g = torch.Generator().manual_seed(0)
    nz = torch.rand(n, in_dim, generator=g) < frac
    nz[:2] = True
    x = torch.exp(torch.randn(n, in_dim, generator=g) * 0.7) * nz
    nzq = torch.stack([torch.quantile(x[nz[:, j], j], torch.linspace(0, 1, num)) for j in range(in_dim)])
    grid = REV.extend_grid(torch.cat([torch.zeros(in_dim, 1), nzq], 1), k_extend=k)
    y = REV.coef2curve(x, grid, torch.randn(in_dim, 4, num + k, generator=g) * 0.25, k)
    return x.to(device), y.to(device), grid.to(device), k


@pytest.mark.parametrize("device", DEVICES)
def test_ill_conditioned_systems(device):
    # CUDA keeps gels' accurate solution of an ill-conditioned full-rank system; the CPU gets the truncated SVD
    # solution gelsy intends (some gelsy builds misjudge the rank of such systems)
    x, y, grid, k = sparse_problem(device)
    mat = REV.B_batch(x, grid, k).permute(1, 0, 2)
    cut = REV.rank_cutoff(mat)
    sv = torch.linalg.svdvals(mat.double())
    ratio = sv[:, -1] / sv[:, 0]
    assert ((ratio > torch.finfo(torch.float32).eps) & (ratio <= cut)).float().mean() >= 0.5  # most inputs in the band
    coef = REV.curve2coef(x, y, grid, k)
    if device == "cuda":
        assert residual(x, y, grid, k, coef) < 1e-8
    else:
        svd = (torch.linalg.pinv(mat.double(), rtol=cut) @ y.permute(1, 0, 2).double()).permute(0, 2, 1)
        assert residual(x, y, grid, k, coef) <= residual(x, y, grid, k, svd) + 1e-8


@pytest.mark.parametrize("device", DEVICES)
def test_refit_blocks_match_one_solve(device):
    torch.manual_seed(0)
    x = torch.randn(300, 7, device=device)
    grid = REV.extend_grid(torch.linspace(-2, 2, 6)[None, :].expand(7, -1), k_extend=3).to(device)
    new = REV.extend_grid(torch.sort(torch.randn(7, 6), dim=1)[0], k_extend=3).to(device)
    coef = torch.randn(7, 11, 8, device=device)
    one = REV.refit_coef(x, grid, coef, 3, new)
    blocks = REV.refit_coef(x, grid, coef, 3, new, max_elements=300 * 16)  # blocks of one input
    assert torch.allclose(one, blocks, atol=1e-5)
    ref = exact(x.cpu().double(), REV.coef2curve(x, grid, coef, 3).cpu().double(), new.cpu().double(), 3)
    assert torch.allclose(one.cpu().double(), ref, atol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_b_batch_coincident_knots(device):
    # a repeated knot: finite gradients and a partition of unity (pykan 0.2.8: NaN gradients, a hole at the knot)
    grid = torch.tensor([[-3., -2., -1., 0., 0., 1., 2., 3., 4.]], device=device)
    x = torch.linspace(0.0, 0.99, 12, device=device)[:, None].requires_grad_(True)  # inside the interior [0, 1)
    b = REV.B_batch(x, grid, 3)
    b.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.allclose(b.sum(-1), torch.ones(12, 1, device=device), atol=1e-5)


def test_b_batch_partition_of_unity_and_support():
    grid = REV.extend_grid(torch.sort(torch.randn(3, 7), dim=1)[0], k_extend=3)
    lo, hi = grid[:, 3], grid[:, -4]
    x = lo + (hi - lo) * torch.rand(500, 1)
    b = REV.B_batch(x, grid, 3)
    assert b.shape == (500, 3, 9) and (b >= -1e-7).all()
    assert torch.allclose(b.sum(-1), torch.ones(500, 3), atol=1e-5)
    far = REV.B_batch(grid[:, -1][None, :] + 1.0, grid, 3)  # beyond the extended grid: no support
    assert torch.equal(far, torch.zeros_like(far))


def test_solve_lstsq_shapes_and_blocks_of_singular():
    torch.manual_seed(0)
    mat = torch.rand(4, 50, 6)
    mat[1, :, 5] = mat[1, :, 4]  # one rank-deficient system
    y = torch.randn(4, 50, 3)
    c = REV.solve_lstsq(mat, y)
    assert c.shape == (4, 6, 3) and torch.isfinite(c).all()
    ref = torch.linalg.pinv(mat.double(), rtol=REV.rank_cutoff(mat)) @ y.double()
    assert torch.allclose(c[1].double(), ref[1], atol=1e-4)  # the minimum-norm solution of the singular system
    singular = REV.numerically_singular(mat)
    assert singular.tolist() == [False, True, False, False]
    assert torch.allclose(REV.solve_lstsq(mat, y, singular=singular), c, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_refit_memory_does_not_grow_with_outputs():
    # the refit solves for the old basis functions (G+k right-hand sides per input), not for each output's curve:
    # 4000 rows x 64 inputs x 4000 outputs of curve values would take 4 GB
    torch.manual_seed(0)
    x = torch.rand(4000, 64, device="cuda") * 2 - 1
    grid = REV.extend_grid(torch.linspace(-1, 1, 6)[None, :].expand(64, 6), k_extend=3).cuda()
    new = REV.extend_grid(torch.sort(torch.rand(64, 6) * 2 - 1, dim=1)[0], k_extend=3).cuda()
    peaks = []
    for out_dim in (8, 4000):
        coef = torch.randn(64, out_dim, 8, device="cuda")
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
        REV.refit_coef(x, grid, coef, 3, new)
        peaks.append(torch.cuda.max_memory_allocated() - base)
    assert peaks[1] < peaks[0] + 4 * coef.numel() * coef.element_size(), [p / 2**20 for p in peaks]
