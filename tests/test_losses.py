"""Losses and regularisers: KANRegularizer against pykan's formula written out here (every metric, values and
gradients); composite losses of several heads, functional terms, shape checks."""
import pytest
import torch

from xkan import KANNetwork, MultiHead, record_activations
from xkan.interp.attribution import network_attribution, node_attribution
from xkan.losses import (REG_METRICS, CoefL1, CoefSmoothness, CompositeLoss, CrossEntropy, EdgeEntropy, EdgeL1,
                        FunctionalTerm, KANRegularizer, MSE, Term, mse)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def network(width=(3, 5, 2), seed=2, device="cpu"):
    return KANNetwork(list(width), grid=5, k=3, seed=seed, device=device)


def reference_reg(net, rec, metric, l1, entropy, coef, coefdiff):
    """pykan's regularisation: the edge scales of the metric (input ranges from the expanded per-edge inputs), l1 and
    row / column entropies, coefficient size and smoothness"""
    r = rec[net]
    scales = []
    for l in range(net.depth):
        rng = torch.std(r.edge_inputs(l), dim=0) + 0.1
        scales.append({"edge_forward_spline_n": torch.std(r.edge_numeric[l], dim=0) / rng,
                       "edge_forward_sum": torch.std(r.edge_outputs[l], dim=0) / rng,
                       "edge_forward_spline_u": torch.std(r.edge_numeric[l], dim=0)}.get(metric))
    if metric == "edge_backward":
        scales = network_attribution(net, r, per_output=False, output_weighting="mean").edge
    if metric == "node_backward":
        scales = node_attribution(net, r, output_weighting="mean")
    if metric == "edge_weights":  # efficient-KAN's: from the weights alone, per edge |scale_sp| mean_g |coef|
        scales = [torch.stack([torch.stack([layer.scale_sp[i, j].abs() * layer.coef[i, j].abs().mean() * layer.mask[i, j]
                                            for i in range(layer.in_dim)]) for j in range(layer.out_dim)])
                  for layer in net.layers]
    total = 0.
    for vec in scales:
        p_row = vec / (torch.sum(vec, dim=1, keepdim=True) + 1)
        p_col = vec / (torch.sum(vec, dim=0, keepdim=True) + 1)
        ent = -torch.mean(torch.sum(p_row * torch.log2(p_row + 1e-4), dim=1)) - torch.mean(torch.sum(p_col * torch.log2(p_col + 1e-4), dim=0))
        total = total + l1 * torch.sum(vec) + entropy * ent
    for layer in net.layers:
        total = total + coef * torch.sum(torch.mean(torch.abs(layer.coef), dim=1))
        total = total + coefdiff * torch.sum(torch.mean(torch.abs(torch.diff(layer.coef)), dim=1))
    return total


def data(n=60, d=3, device="cpu"):
    g = torch.Generator().manual_seed(0)
    return (torch.rand(n, d, generator=g) * 2 - 1).to(device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("metric", REG_METRICS)
def test_kan_regularizer_is_pykans_formula(metric, device):
    lambdas = dict(l1=0.7, entropy=1.9, coef=0.3, coefdiff=0.2)
    a, b = network(device=device), network(device=device)
    x = data(device=device)
    with record_activations(a) as rec:
        a(x)
        value = KANRegularizer(1., metric=metric, **lambdas)(a, rec)
    with record_activations(b) as rec_b:
        b(x)
        ref = reference_reg(b, rec_b, metric, **lambdas)
    assert torch.allclose(value, ref, rtol=1e-6, atol=0)
    value.backward()
    ref.backward()
    # the gradients within rounding (the input ranges' standard deviations taken before broadcasting)
    for p, q in zip(a.layers, b.layers):
        assert float((p.coef.grad - q.coef.grad).abs().max()) <= 1e-6 * float(q.coef.grad.abs().max())


def test_regularizer_weight_and_parts():
    new = network()
    x = data()
    with record_activations(new) as rec:
        new(x)
        full = KANRegularizer(1., l1=1., entropy=2., coef=0.5, coefdiff=0.25).value(new, rec)
        parts = (EdgeL1()(new, rec) + 2 * EdgeEntropy()(new, rec) + 0.5 * CoefL1()(new) + 0.25 * CoefSmoothness()(new))
        assert torch.allclose(full, parts, rtol=1e-6)
        assert torch.allclose(KANRegularizer(3.)(new, rec), 3 * KANRegularizer(1.)(new, rec))
    assert not CoefL1().requires_recording and EdgeL1().requires_recording
    with pytest.raises(ValueError, match="record_activations"):
        EdgeL1()(new)
    with pytest.raises(ValueError):
        KANRegularizer(metric="nonsense")


def test_regularizer_over_every_network_of_a_multihead():
    trunk, head = KANNetwork([3, 4], seed=1), KANNetwork([4, 2], seed=2)
    model = MultiHead(trunk, {"a": head})
    x = data()
    with record_activations(model) as rec:
        model(x)
        both = KANRegularizer()(model, rec)
        assert torch.allclose(both, KANRegularizer()(trunk, rec) + KANRegularizer()(head, rec))


def test_composite_loss_heads_and_components():
    torch.manual_seed(0)
    outputs = {"cls": torch.randn(8, 3, requires_grad=True), "recon": torch.randn(8, 5, requires_grad=True)}
    batch = {"x": torch.rand(8, 5), "cls": torch.randint(0, 3, (8,)), "recon": torch.rand(8, 5)}
    loss = CompositeLoss({"cls": (CrossEntropy(), 0.9), "recon": Term(MSE(), 0.1)})
    total = loss(outputs, batch)
    ce = torch.nn.functional.cross_entropy(outputs["cls"], batch["cls"])
    assert torch.allclose(total, 0.9 * ce + 0.1 * mse(outputs["recon"], batch["recon"]))
    assert set(loss.components) == {"cls", "recon", "total"} and not loss.components["total"].requires_grad
    total.backward()
    # one output tensor: the term named after the batch key; a bare callable; a functional term
    y = torch.randn(8, 2)
    pred = torch.randn(8, 2)
    assert torch.equal(CompositeLoss({"y": MSE()})(pred, {"y": y}), mse(pred, y))
    assert torch.equal(CompositeLoss(mse)(pred, {"y": y}), mse(pred, y))
    poisson = FunctionalTerm(lambda out, b, m: torch.nn.functional.poisson_nll_loss(out, b["counts"]), 2.)
    counts = torch.rand(8, 2)
    assert torch.allclose(CompositeLoss({"nll": poisson})(pred, {"counts": counts}),
                          2 * torch.nn.functional.poisson_nll_loss(pred, counts))
    # a renamed output / target
    assert torch.equal(CompositeLoss({"r": Term(mse, output="recon", target="recon")})(outputs, batch),
                       mse(outputs["recon"], batch["recon"]))


def test_mse_rejects_broadcasting():
    with pytest.raises(ValueError, match="shape"):
        mse(torch.zeros(4, 1), torch.zeros(4))


def test_loss_with_regularizer():
    new = network()
    x, y = data(), data(d=2) * 0.3
    loss = CompositeLoss({"y": MSE()}, regularizers=[KANRegularizer(0.01)])
    assert loss.requires_recording
    with record_activations(new) as rec:
        pred = new(x)
        value = loss(pred, {"x": x, "y": y}, new, rec)
        ref = torch.mean((pred - y) ** 2) + 0.01 * KANRegularizer(1.).value(new, rec)
    assert torch.equal(value, ref)
    assert set(loss.components) == {"y", "reg/kan", "total"}


def test_attribution_needs_two_samples():
    new = network()
    with record_activations(new) as rec:
        new(data(n=1))
    with pytest.raises(ValueError, match="2 samples"):
        network_attribution(new, rec.record)


def test_edge_weights_regularizer_needs_no_activations():
    # efficient-KAN's regulariser: from the weights alone, so no record (and constant memory in the batch size)
    net = network()
    reg = KANRegularizer(1., metric="edge_weights", l1=1., entropy=2.)
    assert not reg.requires_recording
    value = reg(net)  # no record
    with record_activations(net) as rec:
        net(data())
        assert torch.equal(value, reg(net, rec))
    value.backward()
    assert all(layer.coef.grad is not None and layer.scale_sp.grad is not None for layer in net.layers)
    # a removed edge does not count
    with torch.no_grad():
        net.layers[0].mask[0, 0] = 0.
    assert float(reg(net)) < float(value)
