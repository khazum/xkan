"""Attribution and structural tools on KANNetworks: consistent scores; pruned networks that keep the kept nodes' and
edges' parameters and (with constant folding) the function; input pruning; refined grids that keep the function;
permutations and expansions that keep it exactly or within rounding; edits and perturbations of the masks; the
networks given unchanged."""
import copy

import pytest
import torch

from xkan import KANNetwork, fit
from xkan.interp import (attribute, auto_swap, connection_cost, expand_depth, expand_width, feature_score, module,
                        n_edges, perturb, prune, prune_edge, prune_input, prune_node, refine, remove_edge, remove_node,
                        swap)
from xkan.symbolic import fix_symbolic

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def trained(width=(3, 6, 4, 1), device="cpu", steps=40):
    torch.manual_seed(0)
    x = (torch.rand(300, width[0]) * 2 - 1).to(device)
    y = torch.sin(torch.pi * x[:, :1]) + x[:, 1:2] ** 2
    net = KANNetwork(list(width), grid=4, k=3, seed=1, device=device)
    fit(net, {"x": x, "y": y}, steps=steps, opt="adam", lr=2e-2, lamb=1e-2)
    return net, x


def rms(a, b):
    return float(torch.sqrt(torch.mean((a - b) ** 2)))


def same_state(a, b):
    return a.state_dict().keys() == b.state_dict().keys() and all(
        torch.equal(u, v) for u, v in zip(a.state_dict().values(), b.state_dict().values()))


@pytest.mark.parametrize("device", DEVICES)
def test_attribution_scores_are_consistent(device):
    net, x = trained(device=device)
    tracks, single, summed = attribute(net, x), attribute(net, x, per_output=False), attribute(net, x, output_weighting="sum")
    for a, b, c in zip(tracks.node, single.node, summed.node):
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-7) and torch.allclose(c, a * net.width_in[-1], rtol=1e-6)
    for a, b in zip(tracks.edge, single.edge):
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-7)
    assert torch.equal(feature_score(net, x), single.node[0]) and (single.node[0] >= 0).all()
    l = net.depth - 1  # a node's score: the sum of the scores of its outgoing edges
    assert torch.allclose(single.node[l], single.edge[l].sum(dim=0), rtol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_prune_node_keeps_the_kept_nodes_and_folds_the_rest(device):
    net, x = trained(device=device)
    before = copy.deepcopy(net.state_dict())
    scores = attribute(net, x, per_output=False).node
    th = float(scores[1].sort().values[1])  # the 2 weakest nodes of layer 1 go (and layer 2's below that score)
    k1, k2 = (torch.where(scores[l] > th)[0].tolist() for l in (1, 2))
    p = prune_node(net, x, th)
    assert p.width[1] == [len(k1), 0] and len(k1) == net.width_in[1] - 2 and p.width[2] == [len(k2), 0]
    assert torch.equal(p.layers[0].coef, net.layers[0].coef[:, k1])
    assert torch.equal(p.layers[1].coef, net.layers[1].coef[k1][:, k2]) and torch.equal(p.layers[2].coef, net.layers[2].coef[k2])
    folded, unfolded = rms(p(x), net(x)), rms(prune_node(net, x, th, fold_constants=False)(x), net(x))
    assert folded < unfolded and folded < 0.1 * float(net(x).std())
    assert all(torch.equal(v, net.state_dict()[k]) for k, v in before.items())  # the given network unchanged


@pytest.mark.parametrize("device", DEVICES)
def test_prune_edge_and_prune(device):
    net, x = trained(device=device)
    scores = attribute(net, x, per_output=False).edge
    th = float(torch.quantile(scores[0].flatten().cpu(), 0.3))
    e = prune_edge(net, x, th)
    assert torch.equal(e.layers[0].mask, ((scores[0] > th).T * net.layers[0].mask).to(e.layers[0].mask.dtype))
    assert n_edges(e) < n_edges(net) and rms(e(x), net(x)) < rms(prune_edge(net, x, th, fold_constants=False)(x), net(x))
    node_th = float(attribute(net, x, per_output=False).node[1].sort().values[1])
    full = prune(net, x, node_th, th)
    by_hand = prune_node(prune_edge(prune_node(net, x, node_th), x, th), x, node_th)
    assert same_state(full, by_hand) and full.width == by_hand.width
    assert prune(net, x, node_th, th, prune_dead_nodes=False).width == prune_node(net, x, node_th).width


def test_prune_node_keeps_given_nodes_and_multiplication_nodes():
    net, x = trained(width=(3, [4, 2], 1))
    p = prune_node(net, x, keep=[[0, 2, 4]], fold_constants=False)
    assert p.width == [[3, 0], [2, 1], [1, 0]] and p.width_out[1] == 4
    assert torch.equal(p.layers[0].coef, net.layers[0].coef[:, [0, 2, 4, 5]])  # node 4 (the first product): subnodes 4, 5
    assert torch.equal(p.layers[1].coef, net.layers[1].coef[[0, 2, 4]])
    with pytest.warns(UserWarning, match="most important"):
        one = prune_node(net, x, threshold=1e9)
    assert one.width_in[1] == 1


def test_prune_input_selects_columns_and_folds():
    net, x = trained()
    p = prune_input(net, x, keep=[0, 2], fold_constants=False)
    assert p.input_id.tolist() == [0, 2] and p.in_features == 3 and p.width[0] == [2, 0]
    assert torch.equal(p.layers[0].coef, net.layers[0].coef[[0, 2]])
    assert torch.allclose(prune_input(net, x, keep=[0, 1, 2])(x), net(x))
    folded = prune_input(net, x, keep=[0, 2])  # the removed input's mean contribution kept
    y = net(x)
    assert rms(folded(x), y) < rms(p(x), y)
    again = prune_input(p, x, keep=[1])
    assert again.input_id.tolist() == [2] and again(x).shape == (300, 1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["sample", "grid", "sample+uniform"])
def test_refine_keeps_the_function(mode, device):
    net, x = trained(device=device)
    r = refine(net, 8, x, refit_mode=mode)
    assert r.config.grid == 8 and r.layers[0].num == 8 and r.layers[0].coef.shape[-1] == 11
    assert rms(r(x), net(x)) < 0.05 * float(net(x).std())
    for a, b in zip(r.layers, net.layers):
        assert torch.equal(a.mask, b.mask) and torch.equal(a.scale_sp, b.scale_sp) and a.coef.data_ptr() != b.coef.data_ptr()


def test_swap_keeps_the_function_and_swaps_back():
    net, x = trained()
    s = swap(net, 1, 0, 4)
    assert torch.allclose(s(x), net(x), atol=1e-6)
    assert torch.equal(s.layers[0].coef[:, 0], net.layers[0].coef[:, 4]) and torch.equal(s.layers[1].coef[4], net.layers[1].coef[0])
    assert same_state(swap(s, 1, 0, 4), net)
    with pytest.raises(ValueError):
        swap(KANNetwork([2, [2, 1], 1]), 1, 0, 2)


def test_expansions_keep_the_function_and_curves():
    net, x = trained()
    y = net(x)
    d = expand_depth(net)
    assert d.depth == net.depth + 1 and torch.allclose(d(x), y, atol=1e-6)
    w = expand_width(net, 1, 2)
    assert w.width[1] == [8, 0] and torch.allclose(w(x), y, atol=1e-6)
    assert torch.equal(w.layers[0].coef[:, 2:], net.layers[0].coef) and torch.equal(w.layers[1].coef[2:], net.layers[1].coef)
    m = expand_width(net, 2, 1, sum_bool=False, mult_arity=3)
    assert m.width[2] == [4, 1] and m.width_out[2] == 7 and torch.allclose(m(x), y, atol=1e-6)


def test_auto_swap_does_not_raise_the_cost():
    net, x = trained()
    before = float(connection_cost(net, x))
    after = float(connection_cost(auto_swap(net, x), x))
    assert after <= before + 1e-6


def test_edits():
    net, x = trained()
    fix_symbolic(net, 0, 1, 2, "sin", fit=False)
    remove_edge(net, 0, 1, 2)
    assert net.layers[0].mask[1, 2] == 0 and net.symbolic[0].mask[2, 1] == 0 and not net.symbolic[0].active
    remove_node(net, 1, 3)
    assert not net.layers[0].mask[:, 3].any() and not net.layers[1].mask[3].any()
    m = copy.deepcopy(net)
    module(m, 0, "[0,1]->[2,4]")
    mask = net.layers[0].mask.clone()
    mask[2, 2] = mask[2, 4] = 0.  # input 2 (outside) to the module's nodes
    mask[0, [0, 1, 3, 5]] = mask[1, [0, 1, 3, 5]] = 0.  # the module's inputs to the other nodes
    assert torch.equal(m.layers[0].mask, mask)


@pytest.mark.parametrize("mode", ["non-intrusive", "all", "minimal"])
def test_perturb(mode):
    net = KANNetwork([2, 2, 1], seed=1)
    for (l, i, j) in ((0, 0, 0), (1, 0, 0)):  # one symbolic path: input 0 -> node 0 -> output
        fix_symbolic(net, l, i, j, "x", fit=False)
    for l in range(net.depth):
        with torch.no_grad():
            net.layers[l].mask.zero_()
    perturb(net, mag=0.5, mode=mode)
    m0, m1 = net.layers[0].mask, net.layers[1].mask
    # layer 0, edge (input i, output j): input 0 and node 0 are active (a symbolic edge), input 1 and node 1 not
    expected0 = {"all": [[.5, .5], [.5, .5]], "non-intrusive": [[0., .5], [0., .5]], "minimal": [[.5, 0.], [0., 0.]]}[mode]
    assert m0.tolist() == expected0
    if mode == "non-intrusive":  # the last layer: numeric edges on with zero scales
        assert m1.tolist() == [[1.], [1.]] and not net.layers[1].scale_base.any() and not net.layers[1].scale_sp.any()
    else:
        assert m1.tolist() == {"all": [[.5], [.5]], "minimal": [[.5], [0.]]}[mode]
