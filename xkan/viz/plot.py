"""The plot of a KAN network: every edge's curve on the network's graph, edges faded by their scores (pykan's figure).
Needs matplotlib (the [plot] extra)."""
import io
import os

import numpy as np
import sympy
import torch

from ..interp.attribution import network_attribution
from ..interp.recording import recorded

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "img")


def _edge_color(symbolic_mask, numeric_mask):
    if symbolic_mask > 0. and numeric_mask > 0.:
        return "purple", 1.
    if symbolic_mask > 0.:
        return "red", 1.
    if numeric_mask > 0.:
        return "black", 1.
    return "white", 0.


def plot(net, data, *, folder=None, beta=3, metric="backward", scale=0.5, tick=False, sample=False,
         in_vars=None, out_vars=None, title=None, varscale=1.0):
    """
    Plot net (a KANNetwork) from a forward of data (at least 2 rows): each edge's curve over the data, coloured black
    (numeric), red (symbolic) or purple (both), faded by its score: metric 'backward' (attribution, default),
    'forward_n' (output over input standard deviation) or 'forward_u' (output standard deviation), through
    tanh(beta * score). The edge images are drawn in memory (with folder, also saved there as sp_<l>_<i>_<j>.png);
    in_vars / out_vars label the inputs / outputs (strings or sympy expressions; in_vars one per input column, or one
    per network input, see KANNetwork.input_names); returns the figure.
    """
    import matplotlib.pyplot as plt
    rec, _ = recorded(net, data)
    if rec.batch < 2:
        raise ValueError("plot needs the activations of at least 2 samples")
    acts = rec.acts
    post = [p.detach() for p in rec.edge_outputs]
    if metric == "backward":
        scores = network_attribution(net, rec, per_output=False).edge
    elif metric == "forward_n":
        scores = [rec.edge_scale(l).detach() for l in range(net.depth)]
    elif metric == "forward_u":
        scores = [rec.edge_std(l).detach() for l in range(net.depth)]
    else:
        raise ValueError(f"unknown metric {metric!r}; 'backward', 'forward_n' or 'forward_u'")
    if folder is not None:
        os.makedirs(folder, exist_ok=True)
    cfg = net.config
    width, width_in, width_out = np.array(net.width), np.array(net.width_in), np.array(net.width_out)
    images = {}  # the PNG bytes of each edge's curve
    for l in range(net.depth):
        sym_mask, num_mask = net.symbolic[l].mask.detach().cpu(), net.layers[l].mask.detach().cpu()
        for i in range(width_in[l]):
            xs = acts[l][:, i]
            rank = torch.argsort(xs)
            for j in range(width_out[l + 1]):
                fig, ax = plt.subplots(figsize=(2., 2.))
                color, alpha_mask = _edge_color(float(sym_mask[j, i]), float(num_mask[i, j]))
                x, y = xs[rank].cpu().numpy(), post[l][:, j, i][rank].cpu().numpy()
                if tick:
                    ax.tick_params(axis="y", direction="in", pad=-22, labelsize=50)
                    ax.tick_params(axis="x", direction="in", pad=-15, labelsize=50)
                    plt.xticks([x.min(), x.max()], ["%2.f" % x.min(), "%2.f" % x.max()])
                    plt.yticks([y.min(), y.max()], ["%2.f" % y.min(), "%2.f" % y.max()])
                else:
                    plt.xticks([])
                    plt.yticks([])
                plt.gca().patch.set_edgecolor("black" if alpha_mask == 1 else "white")
                plt.gca().patch.set_linewidth(1.5)
                plt.plot(x, y, color=color, lw=5)
                if sample:
                    plt.scatter(x, y, color=color, s=400 * scale ** 2)
                plt.gca().spines[:].set_color(color)
                buf = io.BytesIO()
                plt.savefig(buf, format="png", bbox_inches="tight", dpi=400)
                plt.close()
                images[l, i, j] = buf.getvalue()
                if folder is not None:
                    with open(f"{folder}/sp_{l}_{i}_{j}.png", "wb") as f:
                        f.write(images[l, i, j])
    alpha = [np.nan_to_num(np.tanh(beta * s.cpu().detach().numpy())) for s in scores]
    A, y0, z0 = 1, 0.3, 0.1  # y0: input to pre-mult height, z0: pre-mult to post-mult
    depth_nodes = len(width)
    min_spacing = A / np.maximum(np.max(width_out), 5)
    max_neuron = np.max(width_out)
    max_num_weights = np.max(width_in[:-1] * width_out[1:])
    y1 = 0.4 / np.maximum(max_num_weights, 5)  # size of the function diagrams
    y2 = 0.15 / np.maximum(max_neuron, 5)  # size of the sum / mult symbols
    fig, ax = plt.subplots(figsize=(10 * scale, 10 * scale * (depth_nodes - 1) * (y0 + z0)))
    to_nfc = lambda p: fig.transFigure.inverted().transform(ax.transData.transform(p))  # noqa: E731
    for l in range(depth_nodes):
        n = width_in[l]
        for i in range(n):
            plt.scatter(1 / (2 * n) + i / n, l * (y0 + z0), s=min_spacing ** 2 * 10000 * scale ** 2, color="black")
        if l < depth_nodes - 1:
            sym_mask, num_mask = net.symbolic[l].mask.detach().cpu(), net.layers[l].mask.detach().cpu()
            n_next = width_out[l + 1]
            N = n * n_next
            for i in range(n):
                for j in range(n_next):
                    id_ = i * n_next + j
                    color, alpha_mask = _edge_color(float(sym_mask[j, i]), float(num_mask[i, j]))
                    a = alpha[l][j][i] * alpha_mask
                    plt.plot([1 / (2 * n) + i / n, 1 / (2 * N) + id_ / N], [l * (y0 + z0), l * (y0 + z0) + y0 / 2 - y1],
                             color=color, lw=2 * scale, alpha=a)
                    plt.plot([1 / (2 * N) + id_ / N, 1 / (2 * n_next) + j / n_next],
                             [l * (y0 + z0) + y0 / 2 + y1, l * (y0 + z0) + y0], color=color, lw=2 * scale, alpha=a)
            # subnodes to nodes (the multiplication nodes take consecutive subnodes)
            n_in, n_out = width_out[l + 1], width_in[l + 1]
            targets = list(range(width[l + 1][0]))
            for m, a in enumerate(cfg.arities(l + 1)):
                targets += [width[l + 1][0] + m] * a
            for i, j in enumerate(targets):
                plt.plot([1 / (2 * n_in) + i / n_in, 1 / (2 * n_out) + j / n_out], [l * (y0 + z0) + y0, (l + 1) * (y0 + z0)],
                         color="black", lw=2 * scale)
        plt.xlim(0, 1)
        plt.ylim(-0.1 * (y0 + z0), (depth_nodes - 1 + 0.1) * (y0 + z0))
    plt.axis("off")
    for l in range(depth_nodes - 1):
        n, n_next = width_in[l], width_out[l + 1]
        N = n * n_next
        for i in range(n):
            for j in range(n_next):
                id_ = i * n_next + j
                im = plt.imread(io.BytesIO(images[l, i, j]))
                left, right = to_nfc([1 / (2 * N) + id_ / N - y1, 0])[0], to_nfc([1 / (2 * N) + id_ / N + y1, 0])[0]
                bottom, up = to_nfc([0, l * (y0 + z0) + y0 / 2 - y1])[1], to_nfc([0, l * (y0 + z0) + y0 / 2 + y1])[1]
                newax = fig.add_axes([left, bottom, right - left, up - bottom])
                newax.imshow(im, alpha=alpha[l][j][i])
                newax.axis("off")
        for symbol, count, offset, height in (("sum", width_out[l + 1], 0, l * (y0 + z0) + y0),
                                              ("mult", width[l + 1][1], width[l + 1][0], (l + 1) * (y0 + z0))):
            N = width_out[l + 1] if symbol == "sum" else width_in[l + 1]
            im = plt.imread(os.path.join(ASSETS, f"{symbol}_symbol.png"))
            for j in range(count):
                id_ = j + offset
                left, right = to_nfc([1 / (2 * N) + id_ / N - y2, 0])[0], to_nfc([1 / (2 * N) + id_ / N + y2, 0])[0]
                bottom, up = to_nfc([0, height - y2])[1], to_nfc([0, height + y2])[1]
                newax = fig.add_axes([left, bottom, right - left, up - bottom])
                newax.imshow(im)
                newax.axis("off")
    first = fig.get_axes()[0]
    if in_vars is not None:
        in_vars = net.input_names(in_vars)  # names per input column: those of the columns the network uses
    for labels, n, height in ((in_vars, width_in[0], -0.1), (out_vars, width_in[-1], (y0 + z0) * (depth_nodes - 1) + 0.15)):
        if labels is not None:
            for i in range(n):
                text = f"${sympy.latex(labels[i])}$" if isinstance(labels[i], sympy.Expr) else labels[i]
                first.text(1 / (2 * n) + i / n, height, text, fontsize=40 * scale * varscale,
                           horizontalalignment="center", verticalalignment="center")
    if title is not None:
        first.text(0.5, (y0 + z0) * (depth_nodes - 1) + 0.3, title, fontsize=40 * scale, horizontalalignment="center",
                   verticalalignment="center")
    return fig


def plot_edge(net, data, l, i, j, ax=None):
    """plot edge (input i, output j) of layer l over data (its input against its output, numeric plus symbolic);
    returns the sorted inputs and outputs (numpy)"""
    import matplotlib.pyplot as plt
    rec, _ = recorded(net, data)
    x = rec.inputs[l][:, i].cpu().numpy()
    y = rec.edge_outputs[l][:, j, i].detach().cpu().numpy()
    rank = np.argsort(x)
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 3))
    ax.plot(x[rank], y[rank], marker="o")
    return x[rank], y[rank]
