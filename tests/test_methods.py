"""Helpers of xkan.utils: jacobians and hessians of functions, datasets from data, sparse masks, augmented inputs,
rounding of expressions."""
import pytest
import sympy
import torch


from xkan import utils as U


def test_batch_jacobian_and_hessian_of_a_function():
    x = torch.randn(20, 2)
    f = lambda z: z[:, [0]] ** 2 * z[:, [1]]
    jac = U.batch_jacobian(f, x)
    assert torch.allclose(jac, torch.stack([2 * x[:, 0] * x[:, 1], x[:, 0] ** 2], 1), atol=1e-6)
    hes = U.batch_hessian(f, x)
    assert torch.allclose(hes[:, 0, 0], 2 * x[:, 1], atol=1e-6) and torch.allclose(hes[:, 0, 1], 2 * x[:, 0], atol=1e-6)
    with pytest.raises(ValueError, match="mode"):
        U.batch_jacobian(f, x, mode="matrix")


def test_create_dataset_from_data():
    x, y = torch.rand(100, 3), torch.rand(100, 1)
    train, test = U.create_dataset_from_data(x, y, train_ratio=0.75)
    assert train["x"].shape == (75, 3) and test["x"].shape == (25, 3)
    rows = torch.cat([train["x"], test["x"]])
    assert torch.equal(torch.sort(rows[:, 0])[0], torch.sort(x[:, 0])[0])  # a split of the rows, nothing lost
    labels = dict(zip(map(tuple, x.tolist()), y[:, 0].tolist()))
    assert all(labels[tuple(r)] == float(l) for r, l in zip(train["x"].tolist(), train["y"][:, 0]))


@pytest.mark.parametrize("in_dim,out_dim", [(3, 5), (5, 3), (4, 4)])
def test_sparse_mask(in_dim, out_dim):
    mask = U.sparse_mask(in_dim, out_dim)
    assert mask.shape == (in_dim, out_dim)
    assert (mask.sum(0) >= 1).all() and (mask.sum(1) >= 1).all()  # every node connected


def test_augment_input():
    a, b = sympy.symbols("a b")
    x = torch.rand(10, 2, dtype=torch.float64)
    out = U.augment_input([a, b], [a + b, a * b], x)
    assert out.shape == (10, 4)
    assert torch.allclose(out[:, 0], x[:, 0] + x[:, 1]) and torch.allclose(out[:, 1], x[:, 0] * x[:, 1])
    assert torch.equal(out[:, 2:], x)
    # data dicts and create_dataset's (train, test): the inputs augmented, the labels kept, the given dicts unchanged
    train, test = U.create_dataset(lambda z: z[:, [0]] * z[:, [1]], n_var=2, train_num=20, test_num=5)
    aug_train, aug_test = U.augment_input([a, b], [a / b], (train, test))
    assert aug_train["x"].shape == (20, 3) and aug_test["x"].shape == (5, 3) and aug_train["y"] is train["y"]
    assert train["x"].shape == (20, 2)
    assert torch.allclose(aug_train["x"][:, 0], (train["x"][:, 0] / train["x"][:, 1]).float())
    with pytest.raises(TypeError):
        U.augment_input([a, b], [a / b], [x])


def test_ex_round():
    a, b = sympy.symbols("a b")
    ex = U.ex_round(3.14534242 * sympy.exp(sympy.sin(sympy.pi * a) + b ** 2) - 2.32345402, 2)
    assert sorted(float(f) for f in ex.atoms(sympy.Float)) == pytest.approx([-2.32, 3.15], abs=1e-3)
    # values that round to one unit of the last digit are that value, not a low-precision one printed '0.e-2'
    small = U.ex_round(-0.0095 * a + 0.3 + 0.004 * b, 2)
    assert str(small) == "0.3 - 0.01*a"
    assert float(small.coeff(a)) == pytest.approx(-0.01, abs=2e-5)  # pykan's: -0.009766
