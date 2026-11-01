"""Structural changes of a KAN network: pruning nodes, edges and inputs, refining grids, adding layers and nodes,
permuting nodes, and the small edits (removing edges and nodes, modules, perturbations).

Tools that change the shape of a network (prune*, refine, expand_*, swap, auto_swap) return a new network and leave
the given one unchanged; edits of values (remove_edge, remove_node, module, perturb) work in place. Tools that need
activations take the data explicitly (a forward without autograd)."""
import copy
import warnings
from dataclasses import replace

import torch

from ..layers import KANLayer, SymbolicLayer
from ..nn.network import KANNetwork
from .attribution import network_attribution
from .recording import recorded


def _parts(net):
    return ([p.detach() for p in net.subnode_scale], [p.detach() for p in net.subnode_bias],
            [p.detach() for p in net.node_scale], [p.detach() for p in net.node_bias])


def _config_of(net, width, arities, layers, in_features=None):
    cfg = net.config
    homogeneous = isinstance(cfg.mult_arity, int) and all(a == cfg.mult_arity for row in arities for a in row)
    nums, ks = [layer.num for layer in layers], [layer.k for layer in layers]
    return replace(cfg, width=tuple(tuple(w) for w in width),
                   mult_arity=cfg.mult_arity if homogeneous else tuple(tuple(r) for r in arities),
                   grid=nums[0] if len(set(nums)) == 1 else tuple(nums), k=ks[0] if len(set(ks)) == 1 else tuple(ks),
                   in_features=net.config.in_features if in_features is None else in_features)


def _subnodes(cfg, l, node):
    """the subnodes (outputs of layer l - 1) of node `node` of node layer l"""
    n_sum = cfg.width[l][0]
    if node < n_sum:
        return [node]
    arities = cfg.arities(l)
    start = n_sum + sum(arities[:node - n_sum])
    return list(range(start, start + arities[node - n_sum]))


def reindex_nodes(net, l, node_map, new_arities=()):
    """
    A network whose node i of node layer l is net's node node_map[i] (None: a new node, disconnected); new nodes
    are addition nodes, or multiplication nodes of the arities new_arities (in order). l = 0 re-indexes the inputs
    (input_id follows). Addition nodes must precede multiplication nodes.
    """
    cfg = net.config
    width = [list(w) for w in cfg.width]
    arities = [list(cfg.arities(m)) for m in range(len(width))]
    n_sum_old = cfg.width[l][0]
    new_arities = list(new_arities)
    node_sum, node_ar, sub_map = [], [], []
    for node in node_map:
        if node is None:
            a = new_arities.pop(0) if new_arities else None
            node_sum.append(a is None)
            node_ar.append(a)
            sub_map.extend([None] * (1 if a is None else a))
        else:
            node_sum.append(node < n_sum_old)
            node_ar.append(None if node < n_sum_old else cfg.arities(l)[node - n_sum_old])
            sub_map.extend(_subnodes(cfg, l, node) if l else [node])
    n_sum = sum(node_sum)
    if any(node_sum[n_sum:]):
        raise ValueError("addition nodes must precede multiplication nodes")
    width[l] = [n_sum, len(node_map) - n_sum]
    arities[l] = [a for a in node_ar if a is not None]
    layers, symbolic = list(net.layers), list(net.symbolic)
    sub_scale, sub_bias, node_scale, node_bias = (list(p) for p in _parts(net))
    if l > 0:
        all_in = list(range(layers[l - 1].in_dim))
        layers[l - 1] = layers[l - 1].reindexed(all_in, sub_map)
        symbolic[l - 1] = symbolic[l - 1].reindexed(all_in, sub_map)

        def take(v, mapping, default):
            out = torch.full((len(mapping),), default, dtype=v.dtype, device=v.device)
            kept = [k for k, i in enumerate(mapping) if i is not None]
            out[kept] = v[[mapping[k] for k in kept]]
            return out
        sub_scale[l - 1], sub_bias[l - 1] = take(sub_scale[l - 1], sub_map, 1.), take(sub_bias[l - 1], sub_map, 0.)
        node_scale[l - 1], node_bias[l - 1] = take(node_scale[l - 1], node_map, 1.), take(node_bias[l - 1], node_map, 0.)
    if l < net.depth:
        all_out = list(range(layers[l].out_dim))
        layers[l] = layers[l].reindexed(node_map, all_out)
        symbolic[l] = symbolic[l].reindexed(node_map, all_out)
    input_id = net.input_id
    if l == 0:
        if None in node_map:
            raise ValueError("inputs can be selected or reordered, not added")
        input_id = net.input_id[torch.tensor(node_map, dtype=torch.long, device=net.input_id.device)]
    new_cfg = _config_of(net, width, arities, layers, in_features=net.in_features if l == 0 else None)
    return KANNetwork._assemble(new_cfg, layers, symbolic, (sub_scale, sub_bias, node_scale, node_bias), input_id,
                                net.training)


# ----------------------------------------------------------------------------------------------------- attribution

def attribute(net, data, *, layer=None, out_score=None, per_output=True, output_weighting="sum", **forward_kwargs):
    """the Attribution (xkan.interp.attribution) of net from a forward of data"""
    rec, _ = recorded(net, data, **forward_kwargs)
    return network_attribution(net, rec, layer=layer, out_score=out_score, per_output=per_output,
                               output_weighting=output_weighting)


def feature_score(net, data, **kw):
    """(width_in[0],): the attribution scores of the inputs"""
    return attribute(net, data, per_output=False, **kw).node[0]


def feature_interaction(net, data, l, neuron_th=1e-2, feature_th=1e-2, *, output_weighting="sum"):
    """{tuple of inputs: count} over the nodes of node layer l whose score exceeds neuron_th: the inputs above
    feature_th times the node's largest input score"""
    rec, _ = recorded(net, data)
    scores = network_attribution(net, rec, layer=l, output_weighting=output_weighting).node_all[0]  # (width_in[l], width_in[0])
    groups = {}
    for score in scores:
        if torch.max(score) > neuron_th:
            features = tuple(torch.where(score > torch.max(score) * feature_th)[0].tolist())
            groups[features] = groups.get(features, 0) + 1
    return groups


# ----------------------------------------------------------------------------------------------------- paths

def _edges_present(net):
    """per layer (in, out) bool on the CPU: a numeric (mask != 0) or symbolic edge"""
    return [((layer.mask != 0) | (sym.mask.T != 0)).cpu() for layer, sym in zip(net.layers, net.symbolic)]


def _connected(cfg, edges, nodes):
    """per node layer (width_in[l],) bool: reached from a kept input through kept nodes and edges (a multiplication
    node through any of its subnodes)"""
    conn = [nodes[0].clone()]
    for l in range(cfg.depth):
        sub = (edges[l] & conn[l][:, None]).any(dim=0)
        n_sum = cfg.width[l + 1][0]
        node = [sub[:n_sum]] + [sub[_subnodes(cfg, l + 1, n)].any()[None] for n in range(n_sum, cfg.width_in[l + 1])]
        conn.append(torch.cat(node) & nodes[l + 1])
    return conn


def _reconnect(cfg, scores, exists, edges, nodes, conn, l, n):
    # gives node n of node layer l a path: its highest-scoring existing edge whose source is reached or can be given
    # a path itself (the same way); edges and nodes updated, True when found
    subs = _subnodes(cfg, l, n)
    cand = exists[l - 1][:, subs]  # (in, subnodes)
    sc = torch.where(cand, scores[l - 1][subs, :].T, torch.tensor(-float("inf")))
    for flat in torch.argsort(sc.flatten(), descending=True)[:int(cand.sum())].tolist():
        i, s = divmod(flat, len(subs))
        if l - 1 == 0 or conn[l - 1][i] or _reconnect(cfg, scores, exists, edges, nodes, conn, l - 1, i):
            edges[l - 1][i, subs[s]] = True
            nodes[l - 1][i] = True
            return True
    return False


def _keep_paths(cfg, scores, exists, edges, nodes):
    """
    Gives every output that had a path (through exists) a path again: greedily from the output back, its
    highest-scoring edge from a node that is reached or can be reached the same way; edges and nodes (per layer / node
    layer, bool on the CPU) updated in place. Returns the outputs left without a path (they had none).
    """
    scores = [s.detach().float().cpu() for s in scores]
    had = _connected(cfg, exists, [torch.ones(w, dtype=torch.bool) for w in cfg.width_in])[-1]
    conn = _connected(cfg, edges, nodes)
    for j in torch.where(had & ~conn[-1])[0].tolist():
        if _reconnect(cfg, scores, exists, edges, nodes, conn, cfg.depth, j):
            conn = _connected(cfg, edges, nodes)
    return torch.where(~conn[-1])[0].tolist()


def disconnected_outputs(net):
    """the outputs of net that no path of numeric or symbolic edges joins to an input: constants"""
    cfg = net.config
    conn = _connected(cfg, _edges_present(net), [torch.ones(w, dtype=torch.bool) for w in cfg.width_in])
    return torch.where(~conn[-1])[0].tolist()


def _fold(net, l, mean_out):
    # constant edge outputs (out,) of layer l into the layer's bias when it has one, else the subnode bias
    layer = net.layers[l]
    if layer.bias is not None:
        layer.bias.data.add_(mean_out.to(layer.bias.dtype))
    else:
        net.subnode_bias[l].data.add_(net.subnode_scale[l] * mean_out.to(net.subnode_bias[l].dtype))


# ----------------------------------------------------------------------------------------------------- pruning

def prune_node(net, data, threshold=1e-2, *, keep=None, output_weighting="sum", fold_constants=True, keep_paths=True):
    """
    A network without the hidden nodes whose attribution score is at most threshold (keep: per hidden node layer
    the nodes to keep instead). fold_constants: the mean outputs (over data) of the removed nodes' edges move into
    the biases they fed (the layer's bias, or the subnode's), so the pruned network's output does not shift by them.
    A layer keeps at least its most important node; with keep_paths (and no keep) every output that had a path from
    the inputs keeps its best one (disconnected_outputs lists any left constant).
    output_weighting : 'sum' (default; every output weighs 1, so an edge into an output scores its share of that
        output whatever the number of outputs) or 'mean' (pykan's: every output weighs 1 / outputs)
    """
    rec, _ = recorded(net, data)
    attribution = network_attribution(net, rec, per_output=False, output_weighting=output_weighting)
    scores = attribution.node
    masks = []
    for l in range(1, net.depth):
        if keep is not None:
            mask = torch.zeros(net.width_in[l], dtype=torch.bool)
            mask[list(keep[l - 1])] = True
        else:
            mask = (scores[l] > threshold).cpu()
            if not mask.any():
                mask[int(torch.argmax(scores[l]))] = True
                warnings.warn(f"prune_node: no node of layer {l} is above threshold {threshold}; its most important "
                              "node is kept")
        masks.append(mask)
    if keep is None and keep_paths:
        ones = lambda w: torch.ones(w, dtype=torch.bool)  # noqa: E731
        edges = _edges_present(net)
        _keep_paths(net.config, attribution.edge, edges, [e.clone() for e in edges],
                    [ones(net.width_in[0])] + masks + [ones(net.width_in[-1])])
    kept = [torch.where(mask)[0].tolist() for mask in masks]
    new = copy.deepcopy(net)
    if fold_constants:
        with torch.no_grad():
            for l in range(1, net.depth):
                removed = torch.ones(net.width_in[l], dtype=torch.bool, device=net.device)
                removed[kept[l - 1]] = False
                if removed.any():
                    _fold(new, l, rec.edge_outputs[l][:, :, removed].mean(dim=0).sum(dim=1))
    for l in range(1, net.depth):
        new = reindex_nodes(new, l, kept[l - 1])
    return new


def _mask_edges(net, rec, scores, keep, *, fold_constants=True, keep_paths=True):
    # masks, in place, the numeric edges of net outside keep (per layer (in, out) bool), first giving back to every
    # output that had a path its best one (keep_paths); the removed edges' mean outputs folded (fold_constants)
    if keep_paths:
        present = _edges_present(net)
        symbolic = [(sym.mask.T != 0).cpu() for sym in net.symbolic]
        before = [k.cpu() | sym for k, sym in zip(keep, symbolic)]  # the paths left: kept numeric or symbolic edges
        edges = [b.clone() for b in before]
        _keep_paths(net.config, scores, present, edges, [torch.ones(w, dtype=torch.bool) for w in net.config.width_in])
        keep = [k.cpu() | (e & ~b) for k, e, b in zip(keep, edges, before)]  # plus the edges given back
    with torch.no_grad():
        for l, layer in enumerate(net.layers):
            old_mask = layer.mask
            new_mask = (keep[l].to(old_mask.device) * old_mask).to(old_mask.dtype)
            removed = (old_mask != 0) & (new_mask == 0)  # (in, out)
            if fold_constants and removed.any():
                _fold(net, l, (rec.edge_numeric[l].mean(dim=0).T * removed).sum(dim=0))
            layer.mask.copy_(new_mask)


def prune_edge(net, data, threshold=3e-2, *, output_weighting="sum", fold_constants=True, keep_paths=True):
    """a network whose numeric edges with attribution scores at most threshold are masked (their mean outputs over
    data folded into the biases with fold_constants); with keep_paths every output that had a path from the inputs
    keeps its best one; output_weighting as in prune_node"""
    rec, _ = recorded(net, data)
    scores = network_attribution(net, rec, per_output=False, output_weighting=output_weighting).edge
    new = copy.deepcopy(net)
    _mask_edges(new, rec, scores, [(scores[l] > threshold).T for l in range(net.depth)], fold_constants=fold_constants,
                keep_paths=keep_paths)
    return new


def prune_to_budget(net, data, evaluate, *, budget=0.02, fine_tune=None, step=0.25, min_step=1 / 64, max_rounds=100,
                    output_weighting="sum", fold_constants=True, keep_paths=True, model=None):
    """
    Prunes net's numeric edges in place, as far as a validation budget allows. Each round masks the lowest-scoring
    share `step` of the remaining numeric edges (attribution scores on data, every output keeping a path with
    keep_paths; their mean outputs folded), calls fine_tune(net) (optional: a few epochs of the training loop, say)
    and evaluate(net), a validation loss. A round whose loss stays within (1 + budget) times the loss of net as
    given is kept; otherwise the state before it is restored and step halved, until step is below min_step,
    max_rounds rounds have run or no edge can go (each edge left carries an output's path). model: the module whose
    state a rejected round restores, when fine_tune trains more than net (the model that holds net's layers; default
    net).

    Returns a report: {"reference": loss as given, "limit", "loss": the kept state's, "edges": numeric edges left,
    "rounds": [{"edges", "loss", "accepted", "step"}, ...], "disconnected": disconnected_outputs(net)}.
    """
    model = net if model is None else model
    reference = float(evaluate(net))
    limit = reference * (1 + budget) if reference >= 0 else reference * (1 - budget)
    state, loss, rounds = copy.deepcopy(model.state_dict()), reference, []
    while step >= min_step and len(rounds) < max_rounds:
        live = sum(int((layer.mask != 0).sum()) for layer in net.layers)
        n = int(step * live)
        if n < 1:
            break
        rec, _ = recorded(net, data)
        scores = network_attribution(net, rec, per_output=False, output_weighting=output_weighting).edge
        flat = torch.cat([torch.where(layer.mask.T != 0, scores[l], torch.tensor(float("inf"), device=scores[l].device)
                                      ).flatten() for l, layer in enumerate(net.layers)])
        cut = torch.kthvalue(flat.float().cpu(), n).values
        keep = [(scores[l].float() > cut.to(scores[l].device)).T for l in range(net.depth)]
        _mask_edges(net, rec, scores, keep, fold_constants=fold_constants, keep_paths=keep_paths)
        if n_edges(net) == live:  # every edge left carries an output's path
            break
        if fine_tune is not None:
            fine_tune(net)
        value = float(evaluate(net))
        accepted = value == value and value <= limit
        rounds.append({"edges": n_edges(net), "loss": value, "accepted": accepted, "step": step})
        if accepted:
            state, loss = copy.deepcopy(model.state_dict()), value
        else:
            model.load_state_dict(state)
            step /= 2
    return {"reference": reference, "limit": limit, "loss": loss, "edges": n_edges(net), "rounds": rounds,
            "disconnected": disconnected_outputs(net)}


def prune(net, data, node_th=1e-2, edge_th=3e-2, *, output_weighting="sum", fold_constants=True, keep_paths=True,
          prune_dead_nodes=True):
    """prune_node, then prune_edge, then (prune_dead_nodes) prune_node again for the nodes left without edges"""
    kw = dict(output_weighting=output_weighting, fold_constants=fold_constants, keep_paths=keep_paths)
    new = prune_edge(prune_node(net, data, node_th, **kw), data, edge_th, **kw)
    return prune_node(new, data, node_th, **kw) if prune_dead_nodes else new


def prune_input(net, data, threshold=1e-2, *, keep=None, output_weighting="sum", fold_constants=True, keep_paths=True):
    """
    A network without the inputs whose attribution score is at most threshold (keep: the inputs to keep, positions
    in the network's input layer); the network still takes the same input columns (input_id selects the kept
    ones). fold_constants: the removed inputs' mean edge outputs move into the first layer's biases. keep_paths (and
    no keep): every output that had a path from the inputs keeps its best one. output_weighting as in prune_node.
    """
    rec, _ = recorded(net, data)
    if keep is None:
        attribution = network_attribution(net, rec, per_output=False, output_weighting=output_weighting)
        mask = (attribution.node[0] > threshold).cpu()
        if not mask.any():
            mask[int(torch.argmax(attribution.node[0]))] = True
            warnings.warn(f"prune_input: no input is above threshold {threshold}; the most important one is kept")
        if keep_paths:
            edges = _edges_present(net)
            _keep_paths(net.config, attribution.edge, edges, [e.clone() for e in edges],
                        [mask] + [torch.ones(w, dtype=torch.bool) for w in net.width_in[1:]])
        keep = torch.where(mask)[0].tolist()
    keep = [int(i) for i in keep]
    new = copy.deepcopy(net)
    if fold_constants:
        removed = torch.ones(net.width_in[0], dtype=torch.bool, device=net.device)
        removed[keep] = False
        if removed.any():
            with torch.no_grad():
                _fold(new, 0, rec.edge_outputs[0][:, :, removed].mean(dim=0).sum(dim=1))
    return reindex_nodes(new, 0, keep)


# ----------------------------------------------------------------------------------------------------- grids, depth, width

def refine(net, new_grid, data, refit_mode="sample"):
    """a network of net's configuration with new_grid grid intervals, every edge's curve refitted from net's (at the
    layers' inputs on data, or on the grid points with refit_mode='grid' / 'sample+uniform'); scales, masks,
    affine parameters and symbolic edges copied"""
    rec, _ = recorded(net, data)
    cfg = replace(net.config, grid=tuple(new_grid) if isinstance(new_grid, (list, tuple)) else new_grid)
    new = KANNetwork.from_config(cfg, device=net.device, dtype=net.dtype)
    with torch.no_grad():
        for l, (layer, parent) in enumerate(zip(new.layers, net.layers)):
            layer.out_of_grid = parent.out_of_grid  # a layer set apart from the network's options keeps its setting
            layer.initialize_grid_from_parent(parent, rec.inputs[l], mode=refit_mode)
            layer.scale_base.data = parent.scale_base.data.clone()
            layer.scale_sp.data = parent.scale_sp.data.clone()
            layer.mask.data = parent.mask.data.clone()
            if layer.aggregation == "mean" == parent.aggregation:
                layer.agg_scale = parent.agg_scale.clone()
        for name in ("subnode_scale", "subnode_bias", "node_scale", "node_bias"):
            for a, b in zip(getattr(new, name), getattr(net, name)):
                a.data = b.data.clone()
        new.input_id = net.input_id.clone()
    new.symbolic = copy.deepcopy(net.symbolic)
    new._setup()
    new.train(net.training)
    return new


def expand_depth(net):
    """a network with one more layer at the end that passes its input through: symbolic identity edges on the
    diagonal (numeric edges masked)"""
    from ..symbolic.library import default_library
    d = net.width_in[-1]
    with torch.random.fork_rng(devices=[]):
        layer = KANLayer.from_config(replace(net.layers[-1].config, in_dim=d, out_dim=d), device=net.device, dtype=net.dtype)
    with torch.no_grad():
        layer.mask.zero_()
    sym = SymbolicLayer(d, d, device=net.device, dtype=net.dtype)
    x = default_library()["x"]
    for i in range(d):
        sym.set_function(i, i, x)
    with torch.no_grad():
        sym.mask.copy_(torch.eye(d, device=net.device, dtype=net.dtype))
    sym.refresh()
    cfg = net.config
    arities = [list(cfg.arities(m)) for m in range(len(cfg.width))] + [[]]
    width = [list(w) for w in cfg.width] + [[d, 0]]
    layers, symbolic = list(net.layers) + [layer], list(net.symbolic) + [sym]
    one, zero = torch.ones(d, device=net.device, dtype=net.dtype), torch.zeros(d, device=net.device, dtype=net.dtype)
    sub_scale, sub_bias, node_scale, node_bias = (list(p) for p in _parts(net))
    affine = (sub_scale + [one], sub_bias + [zero], node_scale + [one], node_bias + [zero])
    return KANNetwork._assemble(_config_of(net, width, arities, layers), [copy.deepcopy(m) for m in layers],
                                [copy.deepcopy(m) for m in symbolic], affine, net.input_id, net.training)


def expand_width(net, l, n_added, *, sum_bool=True, mult_arity=None):
    """
    A network with n_added more nodes in hidden node layer l, disconnected (their edges masked; train or fix them
    afterwards): addition nodes before the existing ones, or (sum_bool=False) multiplication nodes of arity
    mult_arity (an int or one per node; default the network's) after them. Every existing curve is kept.
    """
    if not 0 < l < net.depth:
        raise ValueError(f"node layer {l} is not a hidden layer")
    if n_added < 1:
        raise ValueError("n_added must be at least 1")
    old = list(range(net.width_in[l]))
    if sum_bool:
        return reindex_nodes(copy.deepcopy(net), l, [None] * n_added + old)
    a = mult_arity if mult_arity is not None else (net.config.mult_arity if isinstance(net.config.mult_arity, int) else 2)
    arities = [a] * n_added if isinstance(a, int) else list(a)
    if len(arities) != n_added or min(arities) < 2:
        raise ValueError("one arity of at least 2 per added multiplication node")
    return reindex_nodes(copy.deepcopy(net), l, old + [None] * n_added, arities)


def swap(net, l, i1, i2):
    """a network with nodes i1 and i2 of hidden node layer l exchanged (both addition or both multiplication nodes;
    the function computed is the same)"""
    if not 0 < l < net.depth:
        raise ValueError(f"node layer {l} is not a hidden layer")
    n_sum = net.config.width[l][0]
    if (i1 < n_sum) != (i2 < n_sum):
        raise ValueError("swap exchanges two addition nodes or two multiplication nodes")
    order = list(range(net.width_in[l]))
    order[i1], order[i2] = order[i2], order[i1]
    return reindex_nodes(copy.deepcopy(net), l, order)


def connection_cost(net, data=None, scores=None):
    """the sum over layers of the edge scores times the distances of the nodes they join (positions in [0, 1])"""
    if scores is None:
        scores = attribute(net, data, per_output=False).edge
    cost = 0.
    for t in scores:
        n_out, n_in = t.shape
        pos = lambda n: torch.linspace(0, 1, steps=n + 1, device=t.device)[:n] + 1 / (2 * n)  # noqa: E731
        cost = cost + torch.sum(torch.abs(pos(n_out)[:, None] - pos(n_in)[None, :]) * t)
    return cost


def auto_swap(net, data):
    """a network whose hidden nodes are permuted (greedily, node by node) to lower the connection cost"""
    for l in range(1, net.depth):
        n_sum = net.config.width[l][0]
        for group in (range(n_sum), range(n_sum, net.width_in[l])):
            for i in group:
                costs = [float(connection_cost(swap(net, l, i, j), data)) for j in group]
                best = list(group)[min(range(len(costs)), key=costs.__getitem__)]
                if best != i:
                    net = swap(net, l, i, best)
    return net


# ----------------------------------------------------------------------------------------------------- edits in place

def remove_edge(net, l, i, j):
    """mask edge (input i, output j) of layer l, numeric and symbolic"""
    with torch.no_grad():
        net.layers[l].mask[i, j] = 0.
    net.symbolic[l].set_mask(j, i, 0.)


def remove_node(net, l, i, mode="all"):
    """mask the edges into ('down'), out of ('up') or both ('all') node i of node layer l"""
    if mode in ("down", "all") and l > 0:
        with torch.no_grad():
            for sub in _subnodes(net.config, l, i):
                net.layers[l - 1].mask[:, sub] = 0.
                net.symbolic[l - 1].mask[sub, :] = 0.
        net.symbolic[l - 1].refresh()
    if mode in ("up", "all") and l < net.depth:
        with torch.no_grad():
            net.layers[l].mask[i, :] = 0.
            net.symbolic[l].mask[:, i] = 0.
        net.symbolic[l].refresh()
    if mode not in ("down", "up", "all"):
        raise ValueError("mode is 'down', 'up' or 'all'")


def module(net, start_layer, chain):
    """keep only the edges of a module given as '[0,1]->[2]->[0]' (nodes of consecutive node layers from
    start_layer): edges between the module's nodes and the rest are masked"""
    groups = chain.split("->")
    for k in range(len(groups) - 1):
        l = start_layer + k
        ids_in = [int(i) for i in groups[k].strip()[1:-1].split(",")]
        ids_out = [int(i) for i in groups[k + 1].strip()[1:-1].split(",")]
        other_in = sorted(set(range(net.width_in[l])) - set(ids_in))
        other_out = sorted(set(range(net.width_out[l + 1])) - set(ids_out))
        with torch.no_grad():
            for a, b in ((other_in, ids_out), (ids_in, other_out)):
                if a and b:
                    net.layers[l].mask[torch.tensor(a)[:, None], torch.tensor(b)[None, :]] = 0.
                    net.symbolic[l].mask[torch.tensor(b)[:, None], torch.tensor(a)[None, :]] = 0.
        net.symbolic[l].refresh()


def n_edges(net):
    """the number of numeric edges with a non-zero mask"""
    return int(sum(int((layer.mask > 0).sum()) for layer in net.layers))


def edge_range(net, data, l, i, j):
    """(x min, x max, y min, y max) of edge (input i, output j) of layer l over data"""
    rec, _ = recorded(net, data)
    x, y = rec.inputs[l][:, i], rec.edge_outputs[l][:, j, i]
    return float(x.min()), float(x.max()), float(y.min()), float(y.max())


def perturb(net, mag=1.0, mode="non-intrusive"):
    """
    Switch numeric edges back on next to the symbolic ones, in place, to let training explore (pykan's perturb): an
    edge is active ('a') when its input or output node has a non-zero symbolic edge; mode 'all' sets the numeric mask
    of every edge to mag, 'minimal' of the edges between active nodes that are already symbolic, 'non-intrusive' of
    the edges into or out of inactive nodes, the last layer's numeric edges switched on with zero scales.
    """
    perturbed = {"all": dict(aa_a=True, aa_i=True, ai=True, ia=True, ii=True),
                 "non-intrusive": dict(aa_a=False, aa_i=False, ai=True, ia=False, ii=True),
                 "minimal": dict(aa_a=True, aa_i=False, ai=False, ia=False, ii=False)}
    if mode not in perturbed:
        raise ValueError(f"unknown mode {mode!r}; one of {sorted(perturbed)}")
    with torch.no_grad():
        for l in range(net.depth):
            names = net.symbolic[l].names
            layer = net.layers[l]
            for j in range(net.width_out[l + 1]):
                for i in range(net.width_in[l]):
                    in_active = any(names[jj][i] != "0" for jj in range(net.width_out[l + 1]))
                    out_active = any(n != "0" for n in names[j])
                    edge = ("a" if in_active else "i") + ("a" if out_active else "i")
                    if l < net.depth - 1 or mode != "non-intrusive":
                        if edge == "aa":
                            edge += "_i" if names[j][i] == "0" else "_a"
                        if perturbed[mode][edge]:
                            layer.mask[i, j] = mag
                    if l == net.depth - 1 and mode == "non-intrusive":
                        layer.mask[i, j] = 1.
                        layer.scale_base[i, j] = 0.
                        layer.scale_sp[i, j] = 0.
