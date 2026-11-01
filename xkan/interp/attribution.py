"""Attribution scores of a KAN network's nodes, subnodes and edges, propagated back from its outputs through the
recorded edge scales (Liu et al., KAN 2.0)."""
from dataclasses import dataclass

import torch


@dataclass
class Attribution:
    """
    The scores of one attribution: per layer l (node layers 0 .. L for nodes; layers 0 .. L - 1 for edges and their
    subnodes) the averaged scores (node[l] (width_in[l],), edge[l] (out, in), subnode[l] (width_out[l + 1],)) and the
    per-track ones (node_all[l] (tracks, width_in[l]), ...): one track per queried output node, or a single averaged
    track (per_output=False).
    """
    node_all: list
    edge_all: list
    subnode_all: list
    node: list
    edge: list
    subnode: list


def _node_to_subnode(score, width, arities):
    # the score of a multiplication node goes to each of its subnodes
    n_sum = width[0]
    parts = [score[:, :n_sum]]
    for m, a in enumerate(arities):
        parts.append(score[:, n_sum + m][:, None].expand(score.shape[0], a))
    return torch.cat(parts, dim=1)


def network_attribution(network, record, *, layer=None, out_score=None, per_output=True, output_weighting="sum"):
    """
    The attribution of network from its record (a NetworkRecord with at least 2 samples; autograd graphs kept, so
    that regularisers differentiate through the scores).

    layer : None (default), from the outputs; or a node layer l, from its nodes, weighted by their scores of the
        attribution from the outputs (out_score)
    out_score : the weights of the nodes it starts from (default: ones)
    per_output : from the outputs, one track per output node (True, default) or one averaged track (the same
        averages, memory independent of the number of outputs)
    output_weighting : 'sum' (default; every output weighs 1, so the scores of a network's last layer are each edge's
        share of its output whatever the number of outputs) or 'mean' (pykan's: every output weighs 1 / outputs, which
        with many outputs puts every score below pykan's pruning thresholds)
    """
    if output_weighting not in ("mean", "sum"):
        raise ValueError(f"unknown output_weighting {output_weighting!r}; 'mean' or 'sum'")
    if record.batch < 2:
        raise ValueError("attribution needs the activations of at least 2 samples (standard deviations over the batch)")
    cfg = network.config
    if layer is not None:
        out_score = network_attribution(network, record, per_output=False, output_weighting=output_weighting).node[layer]
    l_end = cfg.depth if layer is None else layer
    out_dim = cfg.width_in[l_end]
    device = record.inputs[0].device
    if not per_output and layer is None:
        score = ((torch.ones(out_dim, device=device) if out_score is None else out_score) / out_dim)[None, :]
    elif out_score is None:
        score = torch.eye(out_dim, device=device)
    else:
        score = torch.diag(out_score)
    node_all, edge_all, subnode_all = [score], [], []
    for l in range(l_end, 0, -1):
        sub = _node_to_subnode(score, cfg.width[l], cfg.arities(l))
        edge = torch.einsum("ij,ki,i->kij", record.edge_std(l - 1), sub, 1 / (record.subnode_std(l - 1).detach() + 1e-4))
        score = torch.sum(edge, dim=1)
        subnode_all.append(sub)
        edge_all.append(edge)
        node_all.append(score)
    node_all, edge_all, subnode_all = node_all[::-1], edge_all[::-1], subnode_all[::-1]
    factor = out_dim if output_weighting == "sum" else 1
    mean = lambda scores: [torch.mean(s, dim=0) * factor if factor != 1 else torch.mean(s, dim=0) for s in scores]  # noqa: E731
    return Attribution(node_all, edge_all, subnode_all, mean(node_all), mean(edge_all), mean(subnode_all))


def node_attribution(network, record, output_weighting="sum"):
    """per node layer l = 1 .. L: (width_in[l], width_in[0]), the attribution of each node of layer l to the inputs"""
    return [network_attribution(network, record, layer=l, output_weighting=output_weighting).node_all[0]
            for l in range(1, network.depth + 1)]
