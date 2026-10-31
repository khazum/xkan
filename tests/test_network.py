"""KANNetwork, MultiHead, activation recording and the grid update of a module tree: initial parameters from the seed
(the layers' draws in order), outputs against a reference forward written out here (per-edge sums, symbolic edges,
affine transforms, multiplication nodes), recorded activations, grid updates against the layer-by-layer loop;
configurations, state dicts, copies and compilation."""
import copy

import pytest
import torch

from xkan import KANLayer, KANNetwork, KANNetworkConfig, MultiHead, record_activations
from xkan.layers import SymbolicFunction, SymbolicLayer
from xkan.nn import update_grids
from xkan.symbolic.library import SYMBOLIC_LIB

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

CASES = {
    "plain": dict(width=[2, 5, 1], grid=5, k=3),
    "deep": dict(width=[4, 8, 8, 2], grid=3, k=3),
    "mult": dict(width=[3, [4, 2], 1], grid=5, k=3),
    "mult3": dict(width=[3, [2, 2], [1, 1], 1], grid=4, k=2, mult_arity=3),
    "hetero": dict(width=[3, [2, 2], 1], grid=5, k=3, mult_arity=[[], [2, 3], []]),
    "per_layer": dict(width=[3, 4, 2], grid=[5, 3], k=[3, 2], base_fun="identity"),
    "rbf": dict(width=[2, 4, 1], grid=5, k=3, basis="rbf", grid_range=(-2, 2)),
}


def net(case, seed=3, device="cpu"):
    return KANNetwork(**CASES[case], seed=seed, device=device)


def fix(new, l, i, j, name, affine=(0.7, 0.1, 1.3, -0.2)):
    # a symbolic edge: numeric edge off, symbolic on, the given affine parameters
    with torch.no_grad():
        new.layers[l].mask[i, j] = 0.
    fn, sym, c, singular = SYMBOLIC_LIB[name]
    new.symbolic[l].set_function(j, i, SymbolicFunction(name, fn, sym, c, singular), affine)
    new.symbolic[l].set_mask(j, i, 1.)


def reference_forward(new, x, singularity_avoiding=False, y_th=1000.):
    """the network's function written out: per layer the per-edge numeric values and the symbolic edges summed into
    subnodes, the subnode affine transform, the products of the multiplication nodes, the node affine transform"""
    cfg = new.config
    x = x[:, new.input_id]
    for l in range(new.depth):
        y = new.layers[l].edge_forward(x).postacts.sum(dim=2)  # (batch, out)
        sym = new.symbolic[l]
        for j in range(sym.out_dim):
            for i in range(sym.in_dim):
                if sym.mask[j, i] != 0:
                    a, b, c, d = sym.affine[j, i]
                    f = sym.functions[j][i]
                    v = f.singular(a * x[:, i] + b, torch.tensor(y_th))[1] if singularity_avoiding else f.fn(a * x[:, i] + b)
                    y[:, j] = y[:, j] + sym.mask[j, i] * (c * v + d)
        y = new.subnode_scale[l] * y + new.subnode_bias[l]
        n_sum = cfg.width[l + 1][0]
        nodes, start = [y[:, k] for k in range(n_sum)], n_sum
        for a in cfg.arities(l + 1):
            p = y[:, start]
            for t in range(1, a):
                p = p * y[:, start + t]
            nodes.append(p)
            start += a
        x = new.node_scale[l] * torch.stack(nodes, dim=1) + new.node_bias[l]
    return x


def inputs(n, d, device="cpu", seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, d, generator=g) * 2 - 1).to(device)


@pytest.mark.parametrize("case", sorted(CASES))
def test_initial_parameters_follow_the_seed(case):
    # the layers' random initial values drawn in order from a generator seeded with the seed (pykan's draws); the
    # affine transforms identities; no symbolic edge
    new = net(case)
    torch.manual_seed(3)
    for l, layer in enumerate(new.layers):
        ref = KANLayer.from_config(new.config.layer_config(l))
        assert all(torch.equal(a, b) for a, b in zip(ref.state_dict().values(), layer.state_dict().values()))
    for l in range(new.depth):
        assert torch.equal(new.node_scale[l], torch.ones_like(new.node_scale[l])) and not new.node_bias[l].any()
        assert not new.symbolic[l].active and not new.symbolic[l].mask.any()
    assert not torch.equal(net(case, seed=4).layers[0].coef, new.layers[0].coef)


def test_the_global_streams_are_kept():
    torch.manual_seed(5)
    ref = torch.rand(3)
    torch.manual_seed(5)
    KANNetwork([3, 4, 1], seed=11)
    assert torch.equal(torch.rand(3), ref)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", sorted(CASES))
def test_outputs_are_the_reference_forward(case, device):
    new = net(case, device=device)
    x = inputs(64, new.width_in[0], device)
    ref = reference_forward(new, x)
    with record_activations(new):  # the recorded forward: the layers' per-edge values, as the reference
        assert torch.equal(new(x), ref)
    assert torch.allclose(new(x), ref, atol=1e-6)  # the fused forward within rounding


@pytest.mark.parametrize("device", DEVICES)
def test_symbolic_edges_and_singularity_avoiding(device):
    new = net("deep", device=device)
    fix(new, 0, 1, 2, "sin")
    fix(new, 1, 3, 0, "1/x", affine=(1.0, 0.0, 0.5, 0.0))
    fix(new, 2, 0, 1, "x^2")
    x = inputs(80, 4, device)
    for kw in (dict(), dict(singularity_avoiding=True), dict(singularity_avoiding=True, y_th=5.)):
        assert torch.allclose(new(x, **kw), reference_forward(new, x, **kw), rtol=1e-4, atol=1e-5)
    assert [s.active for s in new.symbolic] == [True, True, True]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", ["deep", "mult", "hetero"])
def test_recorded_activations(case, device):
    new = net(case, device=device)
    if case == "deep":
        fix(new, 1, 2, 3, "sin")
    x = inputs(70, new.width_in[0], device)
    with record_activations(new) as rec:
        y = new(x)
    r = rec.record
    assert torch.equal(r.acts[-1], y.detach()) and torch.equal(r.inputs[0], x)
    for l in range(new.depth):
        edges = new.layers[l].edge_forward(r.inputs[l])
        assert torch.equal(r.edge_numeric[l], edges.postacts) and torch.equal(r.edge_spline[l], edges.postspline)
        _, sym = new.symbolic[l](r.inputs[l]) if new.symbolic[l].active else (None, 0.)
        assert torch.equal(r.edge_outputs[l], edges.postacts + sym)
        assert torch.allclose(r.subnode[l], r.edge_outputs[l].sum(dim=2), atol=1e-6)
        assert torch.equal(r.premult[l], new.subnode_scale[l] * r.subnode[l] + new.subnode_bias[l])
        assert torch.equal(r.inputs[l + 1] if l + 1 < new.depth else r.outputs[l], r.outputs[l])
        assert torch.equal(r.edge_inputs(l), x[:, None, :].expand_as(r.edge_outputs[l]) if l == 0 else r.edge_inputs(l))
        assert torch.equal(r.edge_std(l), torch.std(r.edge_outputs[l], dim=0))
        assert torch.equal(r.subnode_std(l), torch.std(r.subnode[l], dim=0))
        assert torch.allclose(r.edge_scale(l, numeric=True), torch.std(r.edge_numeric[l], dim=0) /
                              (torch.std(r.edge_inputs(l), dim=0) + 0.1), rtol=1e-6)


def test_recording_keeps_the_graph_and_is_scoped():
    new = net("plain")
    x = inputs(30, 2)
    with record_activations(new) as outer:
        new(x)
        with record_activations(new) as inner:
            new(x * 0.5)
        assert len(new._recorders) == 1
        new(x)
    assert not new._recorders and "_recorders" not in new.__dict__
    # a forward is recorded by every open scope: the inner one saw only x * 0.5, the outer one last saw x
    assert torch.equal(inner.record.inputs[0], x * 0.5) and torch.equal(outer.record.inputs[0], x)
    reg = sum(outer.record.edge_scale(l, numeric=True).sum() for l in range(new.depth))
    reg.backward()
    assert new.layers[0].coef.grad is not None and new.layers[0].coef.grad.abs().sum() > 0
    with pytest.raises(KeyError):
        record_activations(new).__getitem__(new)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", ["deep", "mult", "rbf"])
def test_grid_update_is_the_layer_by_layer_update(case, device):
    # one forward with every layer updated from its input = updating the layers in order, each from a fresh forward
    a, b = net(case, device=device), net(case, device=device)
    x = inputs(500, a.width_in[0], device) * 1.7
    update_grids(a, x)
    for l in range(b.depth):
        with torch.no_grad(), record_activations(b) as rec:
            b(x)
        b.layers[l].update_grid_from_samples(rec.record.inputs[l])
    for p, q in zip(a.layers, b.layers):
        assert torch.allclose(p.grid, q.grid, atol=1e-6) and torch.allclose(p.coef, q.coef, atol=1e-5)
    assert torch.equal(a.layers[0].grid, b.layers[0].grid) and torch.equal(a.layers[0].coef, b.layers[0].coef)


def test_grid_update_records_nothing():
    new = net("plain")
    x = inputs(100, 2)
    with record_activations(new) as rec:
        update_grids(new, x)
    assert not rec.records


def test_multihead_shared_trunk():
    torch.manual_seed(0)
    trunk = KANNetwork([6, 4], seed=1)
    model = MultiHead(trunk, {"cls": KANNetwork([4, 3], seed=2), "recon": torch.nn.Linear(4, 6)})
    x = torch.rand(20, 6)
    out = model(x)
    assert set(out) == {"cls", "recon"} and out["cls"].shape == (20, 3) and out["recon"].shape == (20, 6)
    assert torch.equal(out["cls"], model.heads["cls"](trunk(x)))
    with record_activations(model) as rec:
        model(x)
    assert set(rec.records) == {trunk, model.heads["cls"]}
    with pytest.raises(ValueError, match="index by network"):
        rec.record
    update_grids(model, x)  # every KAN layer of trunk and heads
    with pytest.raises(ValueError):
        MultiHead(trunk, {})


def test_config_round_trip_and_validation():
    for case, kw in CASES.items():
        net = KANNetwork(**kw, seed=4)
        cfg = KANNetworkConfig.from_dict(net.config.to_dict())
        assert cfg == net.config, case
        again = KANNetwork.from_config(cfg)
        assert all(torch.equal(a, b) for a, b in zip(net.state_dict().values(), again.state_dict().values()))
    assert KANNetwork([2, 1]).config.layer["noise_scale"] == 0.3
    with pytest.raises(ValueError):
        KANNetwork([2])
    with pytest.raises(ValueError):
        KANNetwork([[2, 1], 1])
    with pytest.raises(ValueError):
        KANNetwork([3, [2, 2], 1], mult_arity=[[], [2], []])
    with pytest.raises(ValueError):
        KANNetwork([3, 2, 1], grid=[3])
    with pytest.raises(TypeError):
        KANNetwork([3, 1], nonsense=1)
    with pytest.raises(ValueError):
        KANNetwork([3, 1], basis="nonsense")


def test_state_dict_round_trip_refreshes_symbolic_edges():
    new = net("deep")
    fix(new, 1, 2, 3, "sin")
    other = KANNetwork(**CASES["deep"], seed=9)
    other.symbolic[1].functions[3][2] = new.symbolic[1].functions[3][2]
    other.load_state_dict(new.state_dict())
    assert [s.active for s in other.symbolic] == [False, True, False]
    x = inputs(40, 4)
    assert torch.equal(other(x), new(x))


def test_input_columns_checked():
    net = KANNetwork([3, 2])
    with pytest.raises(ValueError, match="3 columns"):
        net(torch.rand(4, 5))


def test_copies_are_independent_and_drop_compiled_forwards():
    new = net("plain")
    new.compile()
    clone = copy.deepcopy(new)
    assert "_compiled_call_impl" not in clone.__dict__
    with torch.no_grad():
        clone.layers[0].coef.add_(1.)
    x = inputs(10, 2)
    assert not torch.equal(clone(x), new(x))
    assert clone.layers[0].coef.data_ptr() != new.layers[0].coef.data_ptr()


@pytest.mark.parametrize("case", ["plain", "mult", "hetero"])
def test_compiles_to_one_graph(case):
    new = net(case)
    x = inputs(32, new.width_in[0])
    torch._dynamo.reset()
    compiled = torch.compile(new, backend="aot_eager", fullgraph=True)
    assert torch.allclose(compiled(x), new(x), atol=1e-6)
    compiled(x).square().sum().backward()
    assert new.layers[0].coef.grad is not None


def test_symbolic_layer_subset_swap_and_refresh():
    layer = SymbolicLayer(3, 2)
    fn, sym, c, singular = SYMBOLIC_LIB["sin"]
    layer.set_function(1, 2, SymbolicFunction("sin", fn, sym, c, singular))
    layer.set_mask(1, 2, 1.)
    assert layer.active and layer.names[1][2] == "sin"
    sub = layer.subset([2, 0], [1])
    assert sub.names == [["sin", "0"]] and sub.active and torch.equal(sub.mask, torch.tensor([[1., 0.]]))
    layer.swap(0, 2)
    assert layer.names[1][0] == "sin" and layer.mask[1, 0] == 1. and layer.mask[1, 2] == 0.
    with torch.no_grad():
        layer.mask.zero_()
    assert layer.active
    layer.refresh()
    assert not layer.active


def test_from_layers_shares_the_layers():
    # a network of existing layers: their parameters are the network's (training it trains them), the forward is
    # theirs, and the network tools run on it
    from xkan import KANNetwork, fit
    from xkan.interp import attribute
    from xkan.layers import KANLayer
    from xkan.symbolic import auto_symbolic, symbolic_formula
    torch.manual_seed(0)
    a, b = KANLayer(3, 4, num=5, k=3), KANLayer(4, 1, num=4, k=2, base_fun="identity")
    net = KANNetwork.from_layers([a, b])
    assert net.width == [[3, 0], [4, 0], [1, 0]] and net.config.grid == (5, 4) and net.config.k == (3, 2)
    assert net.layers[0] is a and net.layers[1] is b
    x = torch.rand(64, 3) * 2 - 1
    assert torch.equal(net(x), b(a(x)))
    y = torch.sin(x[:, :1]) + x[:, 1:2] ** 2
    before = a.coef.detach().clone()
    fit(net, {"x": x, "y": y}, steps=3, update_grid=False)
    assert not torch.equal(a.coef, before)  # the layer itself was trained
    att = attribute(net, x)
    assert att.edge[0].shape == (4, 3) and att.edge[1].shape == (1, 4)
    auto_symbolic(net, x, lib=["x", "x^2", "sin"], verbose=0)
    assert symbolic_formula(net)[0][0].free_symbols
    with pytest.raises(ValueError, match="outputs"):
        KANNetwork.from_layers([KANLayer(3, 4), KANLayer(5, 1)])
    with pytest.raises(ValueError, match="at least one"):
        KANNetwork.from_layers([])
