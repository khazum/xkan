"""Symbolic regression on KAN networks: fixing edges to library functions, suggestions, the automatic pass over a
network, and the formulas of networks and multi-head models.

The tools take the data explicitly: the edges are fitted to their activations in one forward of the data (all edges
of a pass from the same forward, as pykan). They change the network in place (its symbolic branch and masks)."""
import contextlib

import numpy as np
import sympy
import torch

from ..interp.attribution import network_attribution
from ..interp.recording import paused_recording, recorded
from ..layers.symbolic import SymbolicFunction
from .fitting import REAL_SYMPY, _symbolic_fit_inputs, _symbolic_unstandardize, fit_params, fit_params_batched
from .library import PlainFunction, SymbolicLibrary, default_library


def _r2_loss(r2):
    return np.log2(1 + 1e-5 - r2)


def _function(fun, library):
    if isinstance(fun, SymbolicFunction):
        return fun
    if isinstance(fun, str):
        return library[fun]
    return SymbolicFunction("anonymous", fun, fun, 1, PlainFunction(fun))


def set_mode(net, l, i, j, mode, mask_n=None):
    """edge (input i, output j) of layer l numeric ('n'), symbolic ('s'), both ('sn', the numeric part times mask_n)
    or neither (any other mode)"""
    mask_n, mask_s = {"s": (0., 1.), "n": (1., 0.), "sn": (1. if mask_n is None else mask_n, 1.),
                      "ns": (1. if mask_n is None else mask_n, 1.)}.get(mode, (0., 0.))
    with torch.no_grad():
        net.layers[l].mask[i, j] = mask_n
    net.symbolic[l].set_mask(j, i, mask_s)


def _edge_samples(rec, l, i, j):
    x, y = rec.inputs[l][:, i], rec.edge_outputs[l][:, j, i]
    if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
        raise ValueError(f"edge ({l},{i},{j}) has non-finite activations; fit the network (or check its input) first")
    return x, y


def fix_symbolic(net, l, i, j, fun, data=None, *, record=None, fit=True, library=None, a_range=(-10, 10),
                 b_range=(-10, 10), verbose=False, random=False, r2_eps=1e-4, a_range_relative="widen", r2_tau=None,
                 widen_ranges=False, input_standardize=False):
    """
    Fix edge (input i, output j) of layer l to fun (a library name, a SymbolicFunction or a callable): with fit, its
    affine parameters fitted to the edge's activations on data (or record, a NetworkRecord), else (1, 0, 1, 0)
    (random: uniform in [-1, 1]). The edge becomes symbolic (numeric part masked). Returns the fit's r2 (-1e8 for an
    edge whose numeric part was removed; None without fit).

    a_range, b_range : the grid-search ranges of a and b; a_range_relative: 'widen' (default) divides a_range by the
        largest |x| of the edge's input when that is below 1 (widened for small inputs, never narrowed), True divides it
        always, False keeps it as given (pykan's ranges)
    input_standardize : fit on inputs standardised by their 1% / 99% quantiles (the parameters mapped back)
    r2_eps, r2_tau, widen_ranges : see xkan.utils.fit_params
    """
    library = default_library() if library is None else library
    f = _function(fun, library)
    sym = net.symbolic[l]
    r2 = None
    if not fit:
        sym.set_function(j, i, f, torch.rand(4) * 2 - 1 if random else None)
    else:
        rec = record if record is not None else recorded(net, data)[0]
        x, y = _edge_samples(rec, l, i, j)
        x_fit, a_rng, center, spread = _symbolic_fit_inputs(x, a_range, input_standardize, a_range_relative)
        params, r2 = fit_params(x_fit, y, f.fn, a_range=a_rng, b_range=b_range, verbose=verbose, device=sym.mask.device,
                                r2_eps=r2_eps, r2_tau=r2_tau, widen_ranges=widen_ranges)
        sym.set_function(j, i, f, params)
        if input_standardize:
            with torch.no_grad():
                aff = sym.affine[j, i]
                aff[0], aff[1] = _symbolic_unstandardize(aff[0].clone(), aff[1].clone(), center, spread)
        if net.layers[l].mask[i, j] == 0:
            r2 = -1e8
    set_mode(net, l, i, j, "s")
    return r2


def unfix_symbolic(net, l, i, j):
    """edge (input i, output j) of layer l back to numeric"""
    set_mode(net, l, i, j, "n")
    net.symbolic[l].functions[j][i] = default_library()["0"]


def suggest_symbolic(net, l, i, j, data=None, *, record=None, lib=None, library=None, topk=5, verbose=True,
                     weight_simple=0.8, r2_loss_fun=_r2_loss, c_loss_fun=lambda c: c, **fit_options):
    """
    The library functions ranked for edge (input i, output j) of layer l by weight_simple * complexity loss +
    (1 - weight_simple) * r2 loss (each function fitted to the edge's activations); the network is unchanged.
    lib: the names to try (default all of library). Returns (name, SymbolicFunction, r2, complexity) of the best;
    verbose prints the topk.
    """
    library = (default_library() if library is None else library).resolve(lib)
    rec = record if record is not None else recorded(net, data)[0]
    names = list(library)
    if net.layers[l].mask[i, j] == 0:  # a removed edge: every candidate gives the zero function
        r2s = np.full(len(names), -1e8)
    else:
        x, y = _edge_samples(rec, l, i, j)
        x_fit, a_rng, _, _ = _symbolic_fit_inputs(x, fit_options.get("a_range", (-10, 10)),
                                                  fit_options.get("input_standardize", False),
                                                  fit_options.get("a_range_relative", "widen"))
        r2s = np.array([float(fit_params(x_fit, y, library[n].fn, a_range=a_rng, b_range=fit_options.get("b_range", (-10, 10)),
                                         verbose=False, device=net.symbolic[l].mask.device,
                                         r2_eps=fit_options.get("r2_eps", 1e-4), r2_tau=fit_options.get("r2_tau"),
                                         widen_ranges=fit_options.get("widen_ranges", False))[1]) for n in names])
    cs = np.array([library[n].complexity for n in names])
    r2_loss = r2_loss_fun(r2s).astype("float")
    loss = weight_simple * c_loss_fun(cs) + (1 - weight_simple) * r2_loss
    order = np.argsort(loss)[:topk]
    if verbose:
        print(f"{'function':>16s}  {'fitting r2':>16s}  {'r2 loss':>16s}  {'complexity':>16s}  {'total loss':>16s}")
        for k in order:
            print(f"{names[k]:>16s}  {r2s[k]:>16.6g}  {r2_loss[k]:>16.6g}  {cs[k]:>16.6g}  {loss[k]:>16.6g}")
    best = order[0]
    return names[best], library[names[best]], r2s[best], cs[best]


@contextlib.contextmanager
def _kept_edge(net, l, i, j):
    sym, layer = net.symbolic[l], net.layers[l]
    saved = (layer.mask[i, j].clone(), sym.mask[j, i].clone(), sym.affine[j, i].clone(), sym.functions[j][i])
    try:
        yield
    finally:
        with torch.no_grad():
            layer.mask[i, j], sym.mask[j, i], sym.affine[j, i] = saved[0], saved[1], saved[2]
        sym.functions[j][i] = saved[3]
        sym.refresh()


def _network_deviations(net, data, l, i, j, functions, params, ref_var, forward_kwargs):
    # the mean squared change of the network's output, relative to its variance, with edge (l, i, j) fixed to each
    # candidate
    devs = []
    with _kept_edge(net, l, i, j), torch.no_grad(), paused_recording(net):
        y_cur = net(data, **forward_kwargs)
        for f, p in zip(functions, params):
            net.symbolic[l].set_function(j, i, f, p.to(net.symbolic[l].affine.device))
            set_mode(net, l, i, j, "s")
            d = torch.mean((net(data, **forward_kwargs) - y_cur) ** 2) / ref_var
            devs.append(float(d) if torch.isfinite(d) else np.inf)
    return np.array(devs)


def auto_symbolic(net, data, *, lib=None, library=None, verbose=1, weight_simple=0.8, r2_threshold=0., batched=False,
                  selection="edge", zero_gate=True, zero_gate_threshold=3e-2, a_range=(-10, 10), b_range=(-10, 10),
                  r2_eps=1e-4, a_range_relative="widen", r2_tau=None, widen_ranges=False, input_standardize=False,
                  forward_kwargs=None):
    """
    Fix every numeric edge of net to its best library function (suggest_symbolic's ranking), in place; edges already
    symbolic are kept and removed ones fixed to '0'. All edges are fitted to their activations in one forward of
    data. Returns {(l, i, j): (name, r2, complexity)} of the edges fixed.

    batched : fit each layer's edges at once (fit_params_batched) instead of one by one
    selection : 'edge' (by each fit's r2) or 'network' (by the change of the network's output on data when the edge
        alone is replaced by the candidate; batched fits)
    zero_gate : edges with attribution scores of at least zero_gate_threshold are not fixed to '0'
    r2_threshold : edges whose best r2 is lower stay numeric
    a_range, b_range, a_range_relative, r2_eps, r2_tau, widen_ranges, input_standardize : see fix_symbolic
    """
    if selection not in ("edge", "network"):
        raise ValueError(f"unknown selection {selection!r}; 'edge' or 'network'")
    forward_kwargs = dict(forward_kwargs or {})
    library = (default_library() if library is None else library).resolve(lib)
    rec, _ = recorded(net, data, **forward_kwargs)
    scores = [s.detach() for s in network_attribution(net, rec, per_output=False).edge] if zero_gate else None
    names = list(library)
    cs = np.array([library[n].complexity for n in names])
    fit_kw = dict(a_range=a_range, b_range=b_range, r2_eps=r2_eps, a_range_relative=a_range_relative, r2_tau=r2_tau,
                  widen_ranges=widen_ranges, input_standardize=input_standardize)
    fixed = {}
    if selection == "network":
        with torch.no_grad(), paused_recording(net):
            ref_var = net(data, **forward_kwargs).var(dim=0).mean().clamp(min=1e-12)
    for l in range(net.depth):
        sym, layer = net.symbolic[l], net.layers[l]
        edges = []
        for i in range(net.width_in[l]):
            for j in range(net.width_out[l + 1]):
                if sym.mask[j, i] > 0. and layer.mask[i, j] == 0.:
                    if verbose >= 1:
                        print(f"skipping ({l},{i},{j}) since already symbolic")
                elif sym.mask[j, i] == 0. and layer.mask[i, j] == 0.:
                    fix_symbolic(net, l, i, j, "0", record=rec, library=default_library(), **fit_kw)
                    if verbose >= 1:
                        print(f"fixing ({l},{i},{j}) with 0")
                else:
                    edges.append((i, j))
        if not edges:
            continue
        gated = lambda i, j: zero_gate and float(scores[l][j, i]) >= zero_gate_threshold  # noqa: E731
        if not batched and selection == "edge":
            for i, j in edges:
                lib_e = [n for n in names if n != "0"] if gated(i, j) else names
                name, f, r2, c = suggest_symbolic(net, l, i, j, record=rec, library=library, lib=lib_e, verbose=False,
                                                  weight_simple=weight_simple, **fit_kw)
                if r2 >= r2_threshold:
                    fix_symbolic(net, l, i, j, f, record=rec, library=library, **fit_kw)
                    fixed[(l, i, j)] = (name, r2, c)
                    if verbose >= 1:
                        print(f"fixing ({l},{i},{j}) with {name}, r2={r2}, c={c}")
                elif verbose >= 1:
                    print(f"({l},{i},{j}): the best fit {name} has r2 = {r2} < {r2_threshold}; the edge stays numeric")
            continue
        ii = torch.tensor([e[0] for e in edges], device=rec.inputs[l].device)
        jj = torch.tensor([e[1] for e in edges], device=rec.inputs[l].device)
        x = rec.inputs[l][:, ii].permute(1, 0).detach()  # (edges, samples)
        y = rec.edge_outputs[l][:, jj, ii].permute(1, 0).detach()
        bad = ~(torch.isfinite(x).all(dim=1) & torch.isfinite(y).all(dim=1))
        if bad.any():
            which = ", ".join(f"({l},{edges[e][0]},{edges[e][1]})" for e in torch.where(bad)[0].tolist())
            raise ValueError(f"edge(s) {which} have non-finite activations; fit the network (or check its input) first")
        x_fit, a_rng, center, spread = _symbolic_fit_inputs(x, a_range, input_standardize, a_range_relative)
        params, r2s = [], []
        for n in names:
            p, r2 = fit_params_batched(x_fit, y, library[n].fn, a_range=a_rng, b_range=b_range, r2_eps=r2_eps,
                                       r2_tau=r2_tau, widen_ranges=widen_ranges)
            if input_standardize:
                p = p.clone()
                p[:, 0], p[:, 1] = _symbolic_unstandardize(p[:, 0], p[:, 1], center, spread)
            params.append(p)
            r2s.append(r2)
        r2s = torch.stack(r2s, dim=1).cpu().double().numpy()  # (edges, functions)
        for e, (i, j) in enumerate(edges):
            if selection == "network":
                devs = _network_deviations(net, data, l, i, j, [library[n] for n in names], [p[e] for p in params],
                                           ref_var, forward_kwargs)
                loss = weight_simple * cs + (1 - weight_simple) * np.log2(1e-5 + devs)
            else:
                loss = weight_simple * cs + (1 - weight_simple) * _r2_loss(r2s[e]).astype("float")
            if "0" in names and gated(i, j):
                loss[names.index("0")] = np.inf
            best = int(np.argsort(loss)[0])
            name, r2, c = names[best], r2s[e][best], cs[best]
            if r2 >= r2_threshold:
                sym.set_function(j, i, library[name], params[best][e].to(sym.affine.device))
                set_mode(net, l, i, j, "s")
                fixed[(l, i, j)] = (name, r2, c)
                if verbose >= 1:
                    print(f"fixing ({l},{i},{j}) with {name}, r2={r2}, c={c}")
            elif verbose >= 1:
                print(f"({l},{i},{j}): the best fit {name} has r2 = {r2} < {r2_threshold}; the edge stays numeric")
    return fixed


# ----------------------------------------------------------------------------------------------------- formulas

def network_formula(net, inputs=None, *, real=False, simplify=False, warn=True):
    """
    The outputs of net as sympy expressions of inputs (default x_1, x_2, ...): the symbolic branch of every edge
    (numeric edges are left out; a warning lists how many have an active spline).
    """
    import warnings
    x = list(inputs) if inputs is not None else [sympy.Symbol(f"x_{k}") for k in range(1, net.width_in[0] + 1)]
    if len(x) != net.width_in[0]:
        raise ValueError(f"the network has {net.width_in[0]} inputs, got {len(x)} variables")
    n_spline = sum(int((layer.mask != 0).sum()) for layer in net.layers)
    if n_spline and warn:
        warnings.warn(f"symbolic_formula: {n_spline} edge(s) have an active spline, which the formula leaves out")
    cfg = net.config
    for l in range(net.depth):
        sym = net.symbolic[l]
        mask, affine = sym.mask.detach().cpu(), sym.affine.detach().cpu()
        sub_scale, sub_bias = net.subnode_scale[l].detach().cpu(), net.subnode_bias[l].detach().cpu()
        y = []
        for j in range(net.width_out[l + 1]):
            yj = 0.
            for i in range(net.width_in[l]):
                m = float(mask[j, i])
                if m == 0.:
                    continue
                a, b, c, d = (float(v) for v in affine[j, i])
                f = sym.functions[j][i]
                fn = REAL_SYMPY[f.name] if real and f.name in REAL_SYMPY else f.sympy
                try:
                    term = c * fn(a * x[i] + b) + d
                except Exception as e:
                    raise ValueError(f"edge ({l},{i},{j}) with function {f.name!r} has no sympy form: {e}") from e
                yj += term if m == 1. else m * term
            if isinstance(yj, float):
                yj = sympy.Float(yj)
            yj = float(sub_scale[j]) * yj + float(sub_bias[j])
            y.append(sympy.simplify(yj) if simplify else yj)
        n_sum, n_mult = cfg.width[l + 1]
        arities = cfg.arities(l + 1)
        mult, off = [], n_sum
        for a in arities:
            p = y[off]
            for t in range(1, a):
                p = p * y[off + t]
            mult.append(p)
            off += a
        y = y[:n_sum] + mult
        node_scale, node_bias = net.node_scale[l].detach().cpu(), net.node_bias[l].detach().cpu()
        x = [float(node_scale[k]) * y[k] + float(node_bias[k]) for k in range(len(y))]
    return x


def symbolic_formula(model, var=None, normalizer=None, output_normalizer=None, *, real=False, simplify=False):
    """
    The formulas of a KANNetwork (a list of sympy expressions, one per output) or of a MultiHead of KANNetworks
    ({head: list}, the trunk's formulas substituted for the heads' inputs), and the input variables.

    var : names or sympy symbols of the input columns (one per column of the input, default x_1, x_2, ..., numbered by
        column: a network that kept columns 0 and 2 (prune_input) is written in x_1 and x_3), or one per network
        input; normalizer (means, stds), per column or per network input likewise: the formulas in the unnormalised
        inputs; output_normalizer (means, stds): in the unnormalised outputs (one network only); real: the
        real-valued forms of functions with singularity avoidance (sqrt, log, ... of |x|). The variables returned
        are the network inputs'.
    """
    from ..nn import KANNetwork, MultiHead
    first = model.trunk if isinstance(model, MultiHead) else model
    if not isinstance(first, KANNetwork):
        raise TypeError("symbolic_formula takes a KANNetwork or a MultiHead of KANNetworks")
    x0 = [v if isinstance(v, sympy.Expr) else sympy.Symbol(v) for v in first.input_names(var)]
    if normalizer is None:
        x = x0
    else:
        means, stds = first.input_names(normalizer[0]), first.input_names(normalizer[1])
        x = [(x0[k] - means[k]) / stds[k] for k in range(len(x0))]
    out = network_formula(first, x, real=real, simplify=simplify)
    if isinstance(model, MultiHead):
        heads = {}
        for name, head in model.heads.items():
            if not isinstance(head, KANNetwork):
                raise TypeError(f"head {name!r} is not a KANNetwork")
            heads[name] = network_formula(head, out, real=real, simplify=simplify)
        return heads, x0
    if output_normalizer is not None:
        means, stds = output_normalizer
        if len(means) != len(out) or len(stds) != len(out):
            raise ValueError("output_normalizer does not match the outputs")
        out = [out[k] * stds[k] + means[k] for k in range(len(out))]
    return out, x0


__all__ = ["fix_symbolic", "unfix_symbolic", "suggest_symbolic", "auto_symbolic", "set_mode", "symbolic_formula",
           "network_formula", "SymbolicLibrary"]
