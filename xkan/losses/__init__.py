"""Losses of one or several heads and the regularisers of KAN networks.

CompositeLoss(terms, regularizers=()) is the objective a Trainer minimises: loss(outputs, batch, model, rec) returns
the total and keeps its parts (weighted, detached) in loss.components. Batches are dicts of tensors: the input "x"
and the targets, by name. A term's name is the head it scores (outputs[name], for models with dict outputs) and the
batch key of its target; for models with one output tensor the term scores that tensor (CompositeLoss({"y": MSE()})).
Regularisers that need activations (KANRegularizer with l1 or entropy) declare requires_recording: the trainer then
records the forward (xkan.interp.record_activations) and passes the recorder.
"""
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

from ..interp.attribution import network_attribution, node_attribution
from ..nn.traversal import kan_networks


def mse(pred, target):
    """mean squared error; the shapes must match (no broadcasting of a (n,) target against (n, 1) outputs)"""
    if pred.shape != target.shape:
        raise ValueError(f"the target's shape {tuple(target.shape)} differs from the output's {tuple(pred.shape)}")
    return torch.mean((pred - target) ** 2)


class MSE(torch.nn.Module):
    def forward(self, pred, target):
        return mse(pred, target)


CrossEntropy = torch.nn.CrossEntropyLoss
PoissonNLL = torch.nn.PoissonNLLLoss


@dataclass(frozen=True)
class Term:
    """weight * fn(prediction, target): prediction outputs[output] (dict outputs; default the term's name) or the
    output tensor, target batch[target] (default the term's name)"""
    fn: Callable
    weight: float = 1.
    output: str | None = None
    target: str | None = None


@dataclass(frozen=True)
class FunctionalTerm:
    """weight * fn(outputs, batch, model): any objective of the whole output, batch and model (count likelihoods
    with size factors, say)"""
    fn: Callable
    weight: float = 1.


class CompositeLoss:
    """
    CompositeLoss({"cls": (CrossEntropy(), 0.9), "recon": (MSE(), 0.1)}, regularizers=[KANRegularizer(1e-3)]).

    terms : name -> Term, FunctionalTerm, (fn, weight) or fn (weight 1)
    regularizers : callables reg(model, rec) -> scalar (weighted), with requires_recording when they read activations
    """

    def __init__(self, terms, regularizers=()):
        if isinstance(terms, Callable) and not isinstance(terms, Mapping):
            terms = {"y": terms}
        self.terms = {name: self._term(t) for name, t in dict(terms).items()}
        self.regularizers = tuple(regularizers)
        if not self.terms and not self.regularizers:
            raise ValueError("a loss needs a term or a regulariser")
        self.components = {}

    @staticmethod
    def _term(t):
        if isinstance(t, (Term, FunctionalTerm)):
            return t
        if isinstance(t, tuple):
            return Term(*t)
        return Term(t)

    @property
    def requires_recording(self):
        return any(getattr(r, "requires_recording", False) for r in self.regularizers)

    def _term_parts(self, outputs, batch, model):
        parts = {}
        for name, t in self.terms.items():
            if isinstance(t, FunctionalTerm):
                parts[name] = t.weight * t.fn(outputs, batch, model)
                continue
            pred = outputs[t.output or name] if isinstance(outputs, Mapping) else outputs
            parts[name] = t.weight * t.fn(pred, batch[t.target or name])
        return parts

    @staticmethod
    def _total(parts):
        total = None
        for v in parts.values():
            total = v if total is None else total + v
        if not isinstance(total, torch.Tensor) or total.ndim != 0:
            raise ValueError("loss terms must give scalar tensors")
        return total

    def terms_value(self, outputs, batch, model=None):
        """the loss without the regularisers (validation); components set likewise"""
        parts = self._term_parts(outputs, batch, model)
        total = self._total(parts)
        self.components = {k: v.detach() for k, v in parts.items()} | {"total": total.detach()}
        return total

    def __call__(self, outputs, batch, model=None, rec=None):
        parts = self._term_parts(outputs, batch, model)
        for i, reg in enumerate(self.regularizers):
            if model is None:
                raise ValueError("regularizers need the model")
            parts[f"reg/{getattr(reg, 'name', type(reg).__name__)}" + (f"/{i}" if len(self.regularizers) > 1 else "")] = reg(model, rec)
        total = self._total(parts)
        self.components = {k: v.detach() for k, v in parts.items()} | {"total": total.detach()}
        return total


REG_METRICS = ("edge_forward_spline_n", "edge_forward_spline_u", "edge_forward_sum", "edge_backward", "node_backward",
               "edge_weights")


class KANRegularizer:
    """
    pykan's regulariser of every KANNetwork in a model, times weight: per layer l1 * sum(s) + entropy * (row +
    column entropies of s) for the edge scales s of the metric, plus coef * mean |coef| and coefdiff * mean |diff
    coef| (both over the outputs) of every layer.

    metric : 'edge_forward_spline_n' (default; numeric edges' output std over input std), 'edge_forward_spline_u'
        (numeric edges' output std), 'edge_forward_sum' (numeric plus symbolic, over input std), 'edge_backward'
        (attribution edge scores), 'node_backward' (attribution node scores), 'edge_weights' (efficient-KAN's: each
        numeric edge's |scale_sp| times its mean |coef|, from the weights alone: no activations recorded, so the memory
        does not grow with the batch, for networks too large for the others)
    """

    name = "kan"

    def __init__(self, weight=1., *, metric="edge_forward_spline_n", l1=1., entropy=2., coef=0., coefdiff=0.):
        if metric not in REG_METRICS:
            raise ValueError(f"unknown metric {metric!r}; one of {REG_METRICS}")
        self.weight, self.metric = weight, metric
        self.l1, self.entropy, self.coef, self.coefdiff = l1, entropy, coef, coefdiff

    @property
    def requires_recording(self):
        return bool(self.l1 or self.entropy) and self.metric != "edge_weights"

    def _scales(self, network, record):
        if self.metric == "edge_weights":
            return [(layer.scale_sp.abs() * layer.coef.abs().mean(dim=-1) * layer.mask).t() for layer in network.layers]
        if self.metric == "edge_forward_spline_n":
            return [record.edge_scale(l, numeric=True) for l in range(network.depth)]
        if self.metric == "edge_forward_spline_u":
            return [record.edge_std(l, numeric=True) for l in range(network.depth)]
        if self.metric == "edge_forward_sum":
            return [record.edge_scale(l) for l in range(network.depth)]
        if self.metric == "edge_backward":  # pykan's weighting of the outputs
            return network_attribution(network, record, per_output=False, output_weighting="mean").edge
        return node_attribution(network, record, output_weighting="mean")

    def value(self, model, rec=None):
        """the unweighted value"""
        total = 0.
        for network in kan_networks(model):
            if self.l1 or self.entropy:
                if self.requires_recording and rec is None:
                    raise ValueError("this regulariser reads activations: compute it inside record_activations(model)")
                for vec in self._scales(network, rec[network] if self.requires_recording else None):
                    l1 = torch.sum(vec)
                    p_row = vec / (torch.sum(vec, dim=1, keepdim=True) + 1)
                    p_col = vec / (torch.sum(vec, dim=0, keepdim=True) + 1)
                    entropy_row = -torch.mean(torch.sum(p_row * torch.log2(p_row + 1e-4), dim=1))
                    entropy_col = -torch.mean(torch.sum(p_col * torch.log2(p_col + 1e-4), dim=0))
                    total += self.l1 * l1 + self.entropy * (entropy_row + entropy_col)
            for layer in network.layers:
                coef_l1 = torch.sum(torch.mean(torch.abs(layer.coef), dim=1))
                coef_diff = torch.sum(torch.mean(torch.abs(torch.diff(layer.coef)), dim=1))
                total += self.coef * coef_l1 + self.coefdiff * coef_diff
        if not isinstance(total, torch.Tensor):
            p = next(model.parameters())
            total = torch.zeros((), device=p.device, dtype=p.dtype)
        return total

    def __call__(self, model, rec=None):
        return self.weight * self.value(model, rec)


class EdgeL1(KANRegularizer):
    """the l1 part of KANRegularizer"""
    name = "edge_l1"

    def __init__(self, weight=1., *, metric="edge_forward_spline_n"):
        super().__init__(weight, metric=metric, l1=1., entropy=0.)


class EdgeEntropy(KANRegularizer):
    """the entropy part of KANRegularizer"""
    name = "edge_entropy"

    def __init__(self, weight=1., *, metric="edge_forward_spline_n"):
        super().__init__(weight, metric=metric, l1=0., entropy=1.)


class CoefL1(KANRegularizer):
    """mean |coef| over the outputs, summed over the edges' inputs and coefficients and the layers"""
    name = "coef_l1"

    def __init__(self, weight=1.):
        super().__init__(weight, l1=0., entropy=0., coef=1.)


class CoefSmoothness(KANRegularizer):
    """mean |coef[k + 1] - coef[k]| over the outputs, summed likewise"""
    name = "coef_smoothness"

    def __init__(self, weight=1.):
        super().__init__(weight, l1=0., entropy=0., coefdiff=1.)


class OutputSparsity:
    """
    A group lasso over each output's incoming edges, one group per edge, times weight: per KAN layer the mean over
    its outputs j of sum_i g_ij / sqrt(sum_i g_ij^2), g_ij = sqrt(scale_base_ij^2 + scale_sp_ij^2 * mean coef_ij^2)
    the size of edge (i, j)'s weights (masked edges 0). Divided by the output's own size (normalize=True, default),
    the term lies between 1 (one edge) and sqrt(in_dim) (equal edges) and does not change when an output's edges
    scale together: it concentrates every output on few inputs without shrinking large outputs more than small ones
    (normalize=False: sum_ij g_ij, the plain group lasso). From the weights alone, no activations recorded.

    layers : the KAN layers to penalise (default every KANLayer of the model)
    """

    name = "output_sparsity"
    requires_recording = False

    def __init__(self, weight=1., *, layers=None, normalize=True, eps=1e-8):
        self.weight, self.layers, self.normalize, self.eps = weight, layers, normalize, eps

    def value(self, model, rec=None):
        from ..layers import KANLayer
        layers = self.layers if self.layers is not None else [m for m in model.modules() if isinstance(m, KANLayer)]
        total = None
        for layer in layers:
            g = torch.sqrt(layer.scale_base ** 2 + layer.scale_sp ** 2 * layer.coef.pow(2).mean(dim=-1) + self.eps)
            g = g * (layer.mask != 0)  # (in, out)
            if self.normalize:
                v = torch.mean(g.sum(dim=0) / torch.sqrt((g ** 2).sum(dim=0) + self.eps))
            else:
                v = g.sum()
            total = v if total is None else total + v
        if total is None:
            p = next(model.parameters())
            total = torch.zeros((), device=p.device, dtype=p.dtype)
        return total

    def __call__(self, model, rec=None):
        return self.weight * self.value(model, rec)


__all__ = ["CompositeLoss", "Term", "FunctionalTerm", "MSE", "mse", "CrossEntropy", "PoissonNLL", "KANRegularizer",
           "REG_METRICS", "EdgeL1", "EdgeEntropy", "CoefL1", "CoefSmoothness", "OutputSparsity"]
