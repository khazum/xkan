"""Pruning networks with many outputs: the outputs weighed 1 each ('sum'), a path kept for every output, pruning to a
validation budget, folding into layer biases; the per-output sparsity regulariser."""
import contextlib

import pytest
import torch

from xkan import KANNetwork, fit
from xkan.interp import (attribute, disconnected_outputs, n_edges, prune, prune_edge, prune_input, prune_node,
                         prune_to_budget)
from xkan.losses import MSE, CompositeLoss, OutputSparsity
from xkan.train import Trainer


def task(outputs=60, seed=0, rows=400):
    # outputs that each depend on one or two of 4 inputs
    g = torch.Generator().manual_seed(seed)
    a = torch.randint(0, 4, (outputs,), generator=g)
    b = torch.randint(0, 4, (outputs,), generator=g)
    w = torch.rand(outputs, generator=g) + 0.5
    x = torch.rand(rows, 4, generator=g) * 2 - 1
    return x, w * torch.sin(2 * x[:, a]) + 0.3 * x[:, b] ** 2


def wide(outputs=60, seed=0, steps=40, **options):
    # through 6 hidden nodes
    x, y = task(outputs, seed)
    net = KANNetwork([4, 6, outputs], grid=4, k=3, seed=seed, **options)
    fit(net, {"x": x, "y": y}, steps=steps, opt="adam", lr=2e-2)
    return net, x, y


def test_sum_weighting_is_mean_times_outputs():
    net, x, _ = wide()
    s, m = attribute(net, x, per_output=False), attribute(net, x, per_output=False, output_weighting="mean")
    for a, b in zip(s.edge, m.edge):
        assert torch.allclose(a, b * net.width_in[-1], rtol=1e-5, atol=1e-7)
    one = KANNetwork([2, 3, 1], grid=3, k=3, seed=0)
    xs = torch.rand(50, 2) * 2 - 1
    assert all(torch.equal(a, b) for a, b in zip(attribute(one, xs, per_output=False).edge,
                                                 attribute(one, xs, per_output=False, output_weighting="mean").edge))


def test_mean_weighting_prunes_every_wide_output_and_sum_keeps_them():
    net, x, _ = wide()
    old = prune_edge(net, x, output_weighting="mean", keep_paths=False)
    assert int((old.layers[1].mask != 0).sum()) == 0 and len(disconnected_outputs(old)) == net.width_in[-1]
    new = prune_edge(net, x)
    assert disconnected_outputs(new) == [] and 0 < n_edges(new) < n_edges(net)


@pytest.mark.parametrize("fn", ["edge", "node", "input", "prune"])
def test_every_output_keeps_a_path(fn):
    net, x, _ = wide()
    tools = {"edge": lambda **kw: prune_edge(net, x, float("inf"), **kw),
             "node": lambda **kw: prune_node(net, x, float("inf"), **kw),
             "input": lambda **kw: prune_input(net, x, float("inf"), **kw),
             "prune": lambda **kw: prune(net, x, float("inf"), float("inf"), **kw)}
    with pytest.warns(UserWarning) if fn != "edge" else contextlib.nullcontext():
        kept = tools[fn]()
    assert disconnected_outputs(kept) == []
    if fn == "edge":
        # one edge into every output, and the hidden nodes they come from reached from an input
        assert torch.equal((kept.layers[1].mask != 0).sum(dim=0), torch.ones(net.width_in[-1], dtype=torch.long))
        assert int((prune_edge(net, x, float("inf"), keep_paths=False).layers[1].mask != 0).sum()) == 0
    assert torch.isfinite(kept(x)).all()


def test_paths_follow_the_highest_scores():
    net, x, _ = wide()
    scores = attribute(net, x, per_output=False).edge
    kept = prune_edge(net, x, float("inf"))
    best = torch.argmax(scores[1], dim=1)  # (outputs,): the highest-scoring hidden node of each output
    assert torch.equal(torch.argmax(kept.layers[1].mask, dim=0), best)
    for h in best.unique().tolist():  # each hidden node used keeps its highest-scoring input edge
        assert torch.equal(torch.where(kept.layers[0].mask[:, h] != 0)[0], torch.argmax(scores[0][h])[None])


def test_fold_into_the_layer_bias():
    net, x, _ = wide(bias=True)
    p = prune_edge(net, x, 0.2)
    assert torch.equal(p.subnode_bias[1], net.subnode_bias[1]) and not torch.equal(p.layers[1].bias, net.layers[1].bias)
    unfolded = prune_edge(net, x, 0.2, fold_constants=False)
    err = lambda m: float(((m(x) - net(x)) ** 2).mean())  # noqa: E731
    assert err(p) < err(unfolded)


def test_prune_to_budget():
    net, x, y = wide(steps=80)
    xv, yv = task(rows=600)
    xv, yv = xv[400:], yv[400:]  # held out
    evaluate = lambda m: float(((m(xv) - yv) ** 2).mean())  # noqa: E731
    tuned = []

    def fine_tune(m):
        tuned.append(1)
        Trainer(m, CompositeLoss({"y": MSE()}), "adam", 5e-3, seed=0).fit({"x": x, "y": y}, steps=10)

    start = n_edges(net)
    report = prune_to_budget(net, x, evaluate, budget=0.1, fine_tune=fine_tune)
    assert report["edges"] == n_edges(net) < start and report["disconnected"] == []
    assert report["loss"] <= report["limit"] and evaluate(net) == pytest.approx(report["loss"], rel=1e-6)
    assert len(tuned) == len(report["rounds"]) and any(not r["accepted"] for r in report["rounds"])
    assert report["rounds"][-1]["step"] < 1 / 64 * 2


def test_output_sparsity_is_scale_free_per_output():
    net = KANNetwork([5, 3], grid=3, k=3, seed=0)
    reg = OutputSparsity()
    v = float(reg.value(net))
    assert 1 <= v <= 5 ** 0.5
    layer = net.layers[0]
    with torch.no_grad():
        for p in (layer.scale_base, layer.scale_sp):
            p[:, 1] *= 7.
    assert float(reg.value(net)) == pytest.approx(v, rel=1e-5)
    with torch.no_grad():
        layer.mask[1:, 0] = 0.
    assert float(reg.value(net)) < v
    plain = OutputSparsity(normalize=False)
    assert float(plain.value(net)) > float(OutputSparsity(normalize=False).value(KANNetwork([5, 3], grid=3, k=3, seed=0)))


def test_output_sparsity_concentrates_the_outputs():
    def ratio(weight):
        net, x, y = wide(steps=1)
        reg = [OutputSparsity(weight, layers=[net.layers[1]])] if weight else []
        Trainer(net, CompositeLoss({"y": MSE()}, reg), "adam", 2e-2, seed=0).fit({"x": x, "y": y}, steps=150)
        return float(OutputSparsity(layers=[net.layers[1]]).value(net))
    assert ratio(1e-1) < 0.8 * ratio(0.)


def test_a_symbolic_edge_does_not_keep_its_numeric_edge():
    # where a symbolic and a numeric edge share a position (compile_formula, then perturb), the numeric one goes by
    # its own score: keep_paths adds only the edges a path needs
    from xkan.interp import perturb
    from xkan.symbolic import compile_formula
    import sympy
    a, b = sympy.symbols("a b")
    net = compile_formula([a, b], sympy.sin(a) + b ** 2)
    perturb(net, mode="all")
    x = torch.rand(300, 2) * 2 - 1
    for th in (0.5, 5.0):  # one numeric edge removed, both
        kept, plain = prune_edge(net, x, th), prune_edge(net, x, th, keep_paths=False)
        assert disconnected_outputs(plain) == [] and n_edges(plain) < n_edges(net)
        assert all(torch.equal(p.mask, q.mask) for p, q in zip(kept.layers, plain.layers))
