"""Behaviour that issues of pykan 0.2.8 concerned, on the networks and tools of 1.0: dtypes and devices, copies,
pickles and functional calls, input checks, training modes and schedules, random streams, pruning and refining edge
cases, derivatives, symbolic edges at singularities and non-finite activations."""
import copy
import io
import pickle
import warnings

import numpy as np
import pytest
import torch

from xkan import CompositeLoss, KANNetwork, Trainer, create_dataset, fit, record_activations
from xkan.interp import expand_depth, prune, prune_edge, prune_node, refine
from xkan.losses import MSE
from xkan.nn import update_grids
from xkan.symbolic import auto_symbolic, fix_symbolic
from xkan.train import GridUpdate
from xkan.utils import get_derivative

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def data(n=200, d=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, d, generator=g) * 2 - 1
    return x, torch.sin(torch.pi * x[:, :1]) + x[:, 1:2] ** 2


def trained(width=(2, 5, 1), steps=30, lamb=1e-2):
    x, y = data()
    net = KANNetwork(list(width), grid=5, k=3, seed=1)
    fit(net, {"x": x, "y": y}, steps=steps, opt="adam", lr=2e-2, lamb=lamb)
    return net, x, y


# ------------------------------------------------------------------------------------------------- dtypes, devices

@pytest.mark.parametrize("device", DEVICES)
def test_device_follows_the_parameters(device):
    net = KANNetwork([2, 3, 1]).to(device)
    assert net.device.type == device and net.symbolic[0].mask.device.type == device
    x = torch.rand(10, 2, device=device)
    assert net(x).device.type == device
    update_grids(net, x)
    assert net.layers[0].grid.device.type == device


def test_input_cast_and_derived_networks_keep_the_dtype():
    net = KANNetwork([2, 4, 1], seed=1).double()
    x = torch.rand(50, 2)  # float32 input into a float64 network
    assert net(x).dtype == torch.float64
    for derived in (prune_node(net, x, keep=[[0, 2]]), refine(net, 6, x), expand_depth(net), prune_edge(net, x)):
        assert derived.dtype == torch.float64 and derived(x).dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_checkpoint_loads_on_the_cpu(tmp_path):
    from xkan.io import load_checkpoint, save_checkpoint
    net = KANNetwork([2, 3, 1], seed=1, device="cuda")
    save_checkpoint(net, tmp_path)
    back = load_checkpoint(tmp_path, map_location="cpu")
    x = torch.rand(10, 2)
    assert back.device.type == "cpu" and torch.allclose(back(x), net(x.cuda()).cpu(), atol=1e-6)


# ------------------------------------------------------------------------------------------------- copies, pickles

def test_copies_after_training_and_inside_recording():
    net, x, y = trained()
    with record_activations(net):
        net(x).sum().backward()  # a forward with a graph: nothing of it stays on the network
        clone = copy.deepcopy(net)
    assert torch.equal(clone(x), net(x))
    fix_symbolic(clone, 0, 0, 0, "sin", x)  # the copy is a working network
    assert not net.symbolic[0].active


def test_whole_network_pickles():
    net, x, _ = trained()
    fix_symbolic(net, 0, 1, 2, "sin", x)
    buf = io.BytesIO()
    torch.save(net, buf)
    buf.seek(0)
    back = torch.load(buf, weights_only=False)
    assert torch.equal(back(x), net(x)) and back.symbolic[0].names == net.symbolic[0].names
    assert torch.equal(pickle.loads(pickle.dumps(net))(x), net(x))


def test_functional_call_uses_the_given_parameters():
    net, x, _ = trained()
    params = {k: v.detach().clone() for k, v in net.named_parameters()}
    params["node_bias.1"] = params["node_bias.1"] + 1.
    out = torch.func.functional_call(net, params, (x,))
    assert torch.allclose(out, net(x) + 1., atol=1e-6)


def test_constructor_leaves_the_callers_width():
    width = [2, [3, 1], 1]
    KANNetwork(width)
    assert width == [2, [3, 1], 1]


# ------------------------------------------------------------------------------------------------- training

def test_trainer_restores_modes_after_a_fit_and_an_error():
    net, x, y = trained(steps=1)
    net.train()
    Trainer(net, CompositeLoss({"y": MSE()})).fit({"x": x, "y": y}, steps=2)
    assert net.training
    net.eval()

    def bad(outputs, batch, model, rec):
        raise RuntimeError("stop")
    with pytest.raises(RuntimeError, match="stop"):
        Trainer(net, bad).fit({"x": x, "y": y}, steps=1)
    assert not net.training


def test_batch_larger_than_the_data_is_the_full_batch():
    x, y = data(50)
    a, b = KANNetwork([2, 3, 1], seed=1), KANNetwork([2, 3, 1], seed=1)
    Trainer(a, CompositeLoss({"y": MSE()}), batch_size=500).fit({"x": x, "y": y}, steps=3)
    Trainer(b, CompositeLoss({"y": MSE()})).fit({"x": x, "y": y}, steps=3)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


def test_grid_update_schedule_edge_cases():
    assert GridUpdate(num=100, stop=10).every == 1  # more updates than steps (0.2.8: ZeroDivisionError)
    assert GridUpdate(num=10, stop=50).every == 5
    with pytest.raises(ValueError):
        GridUpdate(num=0)
    calls = []
    net = KANNetwork([2, 3, 1], seed=1)
    x, y = data(60)

    class Spy(GridUpdate):
        def on_step_start(self, trainer, step):
            if not (step % self.every or step >= self.stop or step < self.start):
                calls.append(step)
            super().on_step_start(trainer, step)
    Trainer(net, CompositeLoss({"y": MSE()}), callbacks=[Spy(3, 9, start=1)]).fit({"x": x, "y": y}, steps=12)
    assert calls == [3, 6]


def test_validation_in_chunks_within_a_memory_budget():
    x, y = data(10000, d=2)
    net = KANNetwork([2, 2000, 1], grid=3, seed=1)  # 2000 * 6 + 1 values per row: chunks of 5592 rows
    trainer = Trainer(net, CompositeLoss({"y": MSE()}))
    assert trainer._eval_chunk(10000) == 2**26 // 12001
    assert torch.allclose(trainer.evaluate({"x": x, "y": y})["y"], torch.mean((net(x) - y) ** 2), rtol=1e-5)


def test_training_draws_from_its_own_generator():
    x, y = data()
    torch.manual_seed(5)
    np.random.seed(5)
    ref_t, ref_n = torch.rand(3), np.random.rand(3)
    torch.manual_seed(5)
    np.random.seed(5)
    fit(KANNetwork([2, 3, 1], seed=1), {"x": x, "y": y}, steps=3, opt="adam", batch=16)
    assert torch.equal(torch.rand(3), ref_t) and np.array_equal(np.random.rand(3), ref_n)


def test_create_dataset_ranges_and_constant_labels():
    train, test = create_dataset(lambda z: z[:, [0]] * 0 + 3., n_var=2, ranges=[[0, 1], [5, 6]], train_num=40,
                                 test_num=10, normalize_input=True, normalize_label=True)
    assert train["x"].shape == (40, 2) and test["y"].shape == (10, 1)
    assert torch.isfinite(train["y"]).all() and torch.equal(train["y"], torch.zeros(40, 1))  # centred, not divided by 0
    raw, _ = create_dataset(lambda z: z[:, [0]], n_var=2, ranges=[[0, 1], [5, 6]], train_num=40, seed=3)
    assert raw["x"][:, 1].min() >= 5 and raw["x"][:, 0].max() <= 1


def test_constant_hidden_node_trains_finite():
    x, y = data()
    net = KANNetwork([2, 3, 1], seed=1)
    with torch.no_grad():
        net.layers[0].mask[:, 1] = 0.  # a node fed by no edge: constant, zero spread
    h = fit(net, {"x": x, "y": y}, steps=5, lamb=1e-2)
    assert np.isfinite(h["train/total"]).all() and all(torch.isfinite(p).all() for p in net.parameters())


@pytest.mark.parametrize("metric", ["edge_forward_spline_n", "edge_forward_sum", "edge_forward_spline_u",
                                    "edge_backward", "node_backward"])
def test_fit_with_each_regulariser(metric):
    x, y = data()
    net = KANNetwork([2, 3, 1], seed=1)
    h = fit(net, {"x": x, "y": y}, steps=6, lamb=1e-2, reg_metric=metric)
    assert h["train/total"][-1] < h["train/total"][0] and np.isfinite(h["train/reg/kan"]).all()


def test_fit_with_singularity_avoiding_symbolic_edges():
    x, y = data()
    net = KANNetwork([2, 2, 1], seed=1)
    fix_symbolic(net, 0, 0, 0, "1/x", fit=False)  # singular at 0, inside the data
    h = fit(net, {"x": x, "y": y}, steps=4, forward_kwargs=dict(singularity_avoiding=True, y_th=10.))
    assert np.isfinite(h["train/total"]).all() and all(torch.isfinite(layer.grid).all() for layer in net.layers)


# ------------------------------------------------------------------------------------------------- structure

def test_no_dead_nodes_after_prune_and_the_function_kept():
    net, x, _ = trained(width=(2, 6, 1), steps=60)
    p = prune(net, x)
    assert p.width_in[1] < net.width_in[1]
    for i in range(p.width_in[1]):  # every kept node has an incoming and an outgoing edge
        assert p.layers[0].mask[:, i].any() and p.layers[1].mask[i].any()
    assert float(torch.sqrt(torch.mean((p(x) - net(x)) ** 2))) < 0.1 * float(net(x).std())


def test_prune_edge_then_prune_node_folds_each_constant_once():
    net, x, _ = trained(width=(2, 6, 1), steps=60)
    a = prune_node(prune_edge(net, x, 5e-2), x, 5e-2)
    b = prune_node(prune_edge(net, x, 5e-2, fold_constants=False), x, 5e-2, fold_constants=False)
    y = net(x)
    assert torch.mean((a(x) - y) ** 2) < torch.mean((b(x) - y) ** 2)


def test_refined_network_shares_no_tensor_and_keeps_the_mean_factor():
    net = KANNetwork([3, 4, 1], seed=1, aggregation="mean")
    x = torch.rand(100, 3)
    p = prune_node(net, x, keep=[[0, 2]])
    r = refine(p, 7, x)
    ours = {t.data_ptr() for t in list(r.parameters()) + list(r.buffers())}
    theirs = {t.data_ptr() for t in list(p.parameters()) + list(p.buffers())}
    assert not ours & theirs
    assert float(r.layers[0].agg_scale) == float(net.layers[0].agg_scale) == pytest.approx(1 / 3)
    with torch.no_grad():
        r.layers[0].coef.add_(1.)
    assert not torch.equal(r.layers[0].coef[..., :3], p.layers[0].coef[..., :3])


def test_networks_of_different_seeds_stay_different_after_refine():
    x = torch.rand(100, 2)
    a, b = refine(KANNetwork([2, 3, 1], seed=1), 6, x), refine(KANNetwork([2, 3, 1], seed=2), 6, x)
    assert not torch.equal(a.layers[0].coef, b.layers[0].coef)


def test_expanded_layers_keep_the_options():
    net = KANNetwork([2, 3, 1], seed=1, basis="rbf", base_fun="identity", grid_eps=0.5)
    d = expand_depth(net)
    assert d.layers[-1].basis == "rbf" and d.layers[-1].grid_eps == 0.5 and d.config.layer["basis"] == "rbf"


# ------------------------------------------------------------------------------------------------- derivatives

def test_get_derivative_jacobian_is_the_gradient_and_the_hessian_symmetric():
    net, x, y = trained(width=(2, 2, 1), steps=5, lamb=0.)
    x, y = x[:20], y[:20]
    jac = get_derivative(net, x, y, derivative="jacobian")
    loss = torch.mean((net(x) - y) ** 2)
    params = [p for p in net.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    flat = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(grads, params)])
    trainable = torch.cat([torch.full((p.numel(),), p.requires_grad) for p in net.parameters()])
    assert torch.allclose(jac[0][trainable], flat, rtol=1e-4, atol=1e-6)
    hess = get_derivative(net, x, y, derivative="hessian")[0]
    assert hess.shape == (jac.shape[1], jac.shape[1]) and torch.allclose(hess, hess.T, atol=1e-5)
    reg = get_derivative(net, x, y, derivative="jacobian", loss_mode="all", lamb=1e-2)
    assert reg.shape == jac.shape and not torch.equal(reg, jac)


# ------------------------------------------------------------------------------------------------- symbolic edges

def test_masked_symbolic_edges_are_not_evaluated():
    net = KANNetwork([2, 1], seed=1)
    fix_symbolic(net, 0, 0, 0, "log", fit=False)  # log of negative inputs: NaN
    net.symbolic[0].set_mask(0, 0, 0.)
    x = torch.rand(10, 2) * 2 - 1
    assert torch.isfinite(net(x)).all()


def test_forward_y_th_default_and_singularity_avoiding():
    net = KANNetwork([1, 1], seed=1)
    fix_symbolic(net, 0, 0, 0, "1/x", fit=False)
    x = torch.tensor([[0.], [1e-5], [0.5]])
    assert not torch.isfinite(net(x)).all()
    avoided = net(x, singularity_avoiding=True)
    assert torch.isfinite(avoided).all() and float(avoided.abs().max()) <= 1000. + 1.
    assert float(net(x, singularity_avoiding=True, y_th=10.).abs().max()) <= 11.


def test_non_finite_activations_are_named():
    net = KANNetwork([2, 2, 1], seed=1)
    with torch.no_grad():
        net.layers[0].coef[0, 1] = float("nan")
    x = torch.rand(20, 2)
    with pytest.raises(ValueError, match=r"\(0,0,1\)"):
        fix_symbolic(net, 0, 0, 1, "sin", x)
    with pytest.raises(ValueError, match="non-finite"):
        auto_symbolic(net, x, lib=["x", "sin"], verbose=0, batched=True)


def test_network_selection_keeps_the_other_edges():
    net, x, _ = trained()
    before = copy.deepcopy(net)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fixed = auto_symbolic(net, x, lib=["x", "x^2", "sin", "0"], verbose=0, batched=True, selection="network")
    assert set(fixed) == {(l, i, j) for l in range(net.depth) for i in range(net.width_in[l])
                          for j in range(net.width_out[l + 1])}
    for a, b in zip(net.layers, before.layers):  # only the masks of the fixed edges change
        assert torch.equal(a.coef, b.coef) and torch.equal(a.grid, b.grid)


def test_input_standardize_reaches_offset_inputs():
    torch.manual_seed(0)
    x = torch.rand(300, 1) * 0.2 + 5.  # far from 0: a relative a-range cannot place sin's argument
    net = KANNetwork([1, 1], grid=10, k=3, seed=1, grid_range=(5., 5.2))
    from xkan.spline import curve2coef
    layer = net.layers[0]
    with torch.no_grad():
        layer.scale_base.zero_()
        layer.scale_sp.fill_(1.)
        layer.coef.copy_(curve2coef(x, torch.sin(20 * (x - 5.1))[:, :, None], layer.grid, layer.k))
    r2 = fix_symbolic(net, 0, 0, 0, "sin", x, input_standardize=True)
    assert float(r2) > 0.99


def test_input_names_follow_the_kept_columns():
    import sympy
    from xkan.interp import prune_input
    from xkan.symbolic import symbolic_formula
    net = KANNetwork([3, 1], seed=1)
    for i in range(3):
        fix_symbolic(net, 0, i, 0, "x", fit=False)
    p = prune_input(net, torch.rand(20, 3), keep=[0, 2], fold_constants=False)
    assert p.input_names() == ["x_1", "x_3"] and p.input_names(["a", "b", "c"]) == ["a", "c"]
    assert p.input_names(["u", "v"]) == ["u", "v"]
    with pytest.raises(ValueError):
        p.input_names(["a"])
    (f,), variables = symbolic_formula(p)
    x1, x3 = sympy.symbols("x_1 x_3")
    assert variables == [x1, x3] and f.free_symbols == {x1, x3}
    (g,), _ = symbolic_formula(p, var=["a", "b", "c"], normalizer=([0., 9., 1.], [1., 9., 2.]))
    a, c = sympy.symbols("a c")
    assert g.free_symbols == {a, c} and abs(float(g.subs({a: 0., c: 1.}))) < 1e-9  # c normalised by column 2's
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from xkan.viz import plot
    fig = plot(p, torch.rand(20, 3), in_vars=["a", "b", "c"])
    assert {t.get_text() for t in fig.get_axes()[0].texts} == {"a", "c"}


def test_lbfgs_history_is_the_loss_at_each_steps_start():
    x, y = data()
    net = KANNetwork([2, 3, 1], seed=1)
    starts = []

    from xkan.train import Callback

    class Before(Callback):
        def on_step_start(self, trainer, step):
            with torch.no_grad():
                starts.append(float(torch.mean((trainer.model(x) - y) ** 2)))
    h = Trainer(net, CompositeLoss({"y": MSE()}), "lbfgs", callbacks=[Before()]).fit({"x": x, "y": y}, steps=4)
    assert np.allclose(h["train/y"], starts, rtol=1e-5)


def test_library_repr_and_plot_tree_variables():
    from xkan.symbolic import default_library
    assert repr(default_library().subset(["x", "sin"])) == "SymbolicLibrary(['x', 'sin'])"
    pytest.importorskip("sklearn")
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import sympy
    from xkan.contrib.hypothesis import plot_tree
    x = torch.rand(100, 3) * 2 - 1
    plot_tree(lambda z: z[:, [0]] + z[:, [1]] * z[:, [2]], x, in_var=sympy.symbols("a b c"))
    plot_tree(lambda z: z[:, [0]] + z[:, [1]] * z[:, [2]], x, in_var=["a", "b", "c"])
