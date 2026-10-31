"""Activations on request: record_activations(model) records the forwards of every KANNetwork in model."""
from contextlib import contextmanager

import torch


class NetworkRecord:
    """
    The values of one forward of a KANNetwork, per layer l (autograd graphs kept; detach() for analysis):

    inputs[l] (batch, width_in[l]): the input of layer l (the network's selected input for l = 0)
    edge_outputs[l] (batch, out, in): every edge's output, numeric plus symbolic
    edge_numeric[l] (batch, out, in): the numeric edges' outputs
    edge_spline[l] (batch, out, in): the numeric edges' spline part (before scale_sp)
    subnode[l] (batch, width_out[l + 1]): the sums of the edges, before the subnode affine transform
    premult[l] (batch, width_out[l + 1]): after the subnode affine transform, before the multiplication nodes
    outputs[l] (batch, width_in[l + 1]): node layer l + 1
    """

    FIELDS = ("inputs", "edge_outputs", "edge_numeric", "edge_spline", "subnode", "premult", "outputs")

    def __init__(self, values):
        for name in self.FIELDS:
            setattr(self, name, list(getattr(values, name)))

    @property
    def depth(self):
        return len(self.inputs)

    @property
    def batch(self):
        return self.inputs[0].shape[0]

    @property
    def acts(self):
        """the node layers 0 .. depth (detached)"""
        return [self.inputs[0].detach()] + [y.detach() for y in self.outputs]

    def edge_inputs(self, l):
        """(batch, out, in): every edge's input"""
        x = self.inputs[l]
        return x[:, None, :].expand(x.shape[0], self.edge_outputs[l].shape[1], x.shape[1])

    def input_range(self, l):
        """(1, in): the standard deviations of layer l's inputs over the batch, plus 0.1"""
        return torch.std(self.inputs[l], dim=0)[None, :] + 0.1

    def edge_std(self, l, numeric=False):
        """(out, in): the standard deviations of the edges' outputs over the batch (numeric edges only, or numeric
        plus symbolic)"""
        return torch.std((self.edge_numeric if numeric else self.edge_outputs)[l], dim=0)

    def edge_scale(self, l, numeric=False):
        """(out, in): the edges' output standard deviations relative to their inputs' (input_range)"""
        return self.edge_std(l, numeric) / self.input_range(l)

    def subnode_std(self, l):
        """(width_out[l + 1],): the standard deviations of the subnodes over the batch"""
        return torch.std(self.subnode[l], dim=0)

    def detach(self):
        new = NetworkRecord.__new__(NetworkRecord)
        for name in self.FIELDS:
            setattr(new, name, [v.detach() for v in getattr(self, name)])
        return new


class record_activations:
    """
    with record_activations(model) as rec: ...  records every forward of every KANNetwork in model (the networks of a
    MultiHead, say) while open: rec[network] (or rec.record for the only one) is the NetworkRecord of its latest
    forward, with autograd graphs (regularisers differentiate through it). A forward is recorded by every open
    scope (nested ones included); on exit the networks keep no references to the values.
    """

    def __init__(self, model):
        self.model = model
        self.records = {}
        self._networks = []

    def __enter__(self):
        from ..nn.traversal import kan_networks
        self._networks = kan_networks(self.model)
        if not self._networks:
            raise ValueError("the model has no KANNetwork to record")
        for net in self._networks:
            net._recorders = (*net._recorders, self)
        return self

    def __exit__(self, *exc):
        for net in self._networks:
            rest = tuple(r for r in net._recorders if r is not self)
            if rest:
                net._recorders = rest
            else:
                net.__dict__.pop("_recorders", None)
        return False

    def _store(self, network, values):
        self.records[network] = NetworkRecord(values)

    def __getitem__(self, network):
        if network not in self.records:
            raise KeyError("no forward of this network was recorded")
        return self.records[network]

    @property
    def record(self):
        """the record of the model's only KANNetwork"""
        if len(self._networks) != 1:
            raise ValueError(f"the model has {len(self._networks)} KANNetworks: index by network")
        return self[self._networks[0]]


@contextmanager
def paused_recording(model):
    """forwards inside record nothing (open recorders keep their records)"""
    from ..nn.traversal import kan_networks
    saved = [(net, net.__dict__.get("_recorders")) for net in kan_networks(model)]
    for net, _ in saved:
        net._recorders = ()
    try:
        yield
    finally:
        for net, rec in saved:
            if rec is None:
                net.__dict__.pop("_recorders", None)
            else:
                net._recorders = rec


def recorded(model, x, **forward_kwargs):
    """the NetworkRecord of model(x) for a model with one KANNetwork (no autograd), and the output"""
    with torch.no_grad(), record_activations(model) as rec:
        y = model(x, **forward_kwargs)
    return rec.record.detach(), y
