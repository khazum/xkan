"""Diagnostics of the KAN layers of any model: how much of each layer's edge output its splines carry, and how many of
its inputs fall outside its grid. Both find a layer's inputs with forward hooks, so they work for a KANNetwork and for
any module built from KANLayers."""
import torch

from ..layers import KANLayer


@torch.no_grad()
def layer_inputs(model, x, rows=None, **forward_kwargs):
    """{qualified name: (layer, its input)} for every KANLayer of model on x (eval mode, the first rows rows of x)"""
    layers = {name: m for name, m in model.named_modules() if isinstance(m, KANLayer)}
    seen, hooks = {}, []
    for name, m in layers.items():
        hooks.append(m.register_forward_pre_hook(lambda mod, args, name=name: seen.setdefault(name, args[0])))
    was = model.training
    model.eval()
    try:
        model(x if rows is None else x[:rows], **forward_kwargs)
    finally:
        model.train(was)
        for h in hooks:
            h.remove()
    return {name: (layers[name], seen[name]) for name in seen}


@torch.no_grad()
def spline_share(model, x, rows=256, **forward_kwargs):
    """
    Per KANLayer (in forward order): the share of its edge output that the splines carry, sum over edges of the
    spline part's standard deviation over the rows of x, over the same sum for the spline and base parts (1: the
    splines carry everything, 0: the layer is the base path alone, as after a spline collapse). Masked edges count 0.
    Returns {qualified name: share}.
    """
    out = {}
    for name, (layer, inp) in layer_inputs(model, x, rows, **forward_kwargs).items():
        edges = layer(inp, return_edges=True)
        live = (layer.mask.T * layer.scale_sp.T)[None]
        spline = (live * edges.postspline).std(0).sum()
        base = (layer.mask.T * layer.scale_base.T)[None] * layer._base_values(inp)[:, None, :]  # under out_of_grid
        total = spline + base.std(0).sum()
        out[name] = float(spline / total) if total > 0 else float("nan")
    return out


@torch.no_grad()
def outside_grid(model, x, **forward_kwargs):
    """Per KANLayer: the share of its input values (rows of x times inputs) outside the inner range of its grid, where
    a B-spline basis is incomplete. Returns {qualified name: share}."""
    out = {}
    for name, (layer, inp) in layer_inputs(model, x, **forward_kwargs).items():
        lo, hi = layer.grid[:, layer.k], layer.grid[:, -layer.k - 1]
        out[name] = float(((inp < lo) | (inp > hi)).float().mean())
    return out
