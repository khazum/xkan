"""The Trainer: a plain training loop written out here reproduced bit for bit (Adam full batch and minibatch, LBFGS,
grid updates, regularisation, DropKAN), multi-head training, loaders, callbacks, schedulers, resuming."""
import copy

import numpy as np
import pytest
import torch

from xkan import KANNetwork, MultiHead, record_activations
from xkan.nn import update_grids
from xkan.losses import MSE, CompositeLoss, CrossEntropy, KANRegularizer, mse
from xkan.train import LBFGS, BestState, EarlyStopping, GridUpdate, Logger, Trainer
from xkan.train.trainer import LBFGS_OPTIONS

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def dataset(n=200, device="cpu", seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 2, generator=g) * 2 - 1
    y = torch.sin(torch.pi * x[:, :1]) + x[:, 1:] ** 2
    x, y = x.to(device), y.to(device)
    return {"x": x[:150], "y": y[:150]}, {"x": x[150:], "y": y[150:]}


def pair(seed=3, device="cpu", **kw):
    return (KANNetwork([2, 5, 1], grid=5, k=3, seed=seed, device=device, **kw),
            KANNetwork([2, 5, 1], grid=5, k=3, seed=seed, device=device, **kw))


def plain_loop(net, train, steps, *, opt="adam", lr=1e-2, seed=3, batch=None, grid=None, lamb=0.):
    """the training loop written out: per step the batch rows, the DropKAN seed and the grid rows from
    np.random.default_rng(seed) in that order, a grid update on all rows (grid = (every, stop)), the step in training
    mode on the mean squared error (+ lamb * KANRegularizer on a recorded forward)"""
    rng = np.random.default_rng(seed)
    params = [p for p in net.parameters() if p.requires_grad]
    o = (torch.optim.Adam(params, lr=lr) if opt == "adam"
         else LBFGS(params, lr=lr, **LBFGS_OPTIONS, tolerance_ys=1e-32))
    drop = [layer for layer in net.layers if layer.dropkan > 0.]
    n = len(train["x"])
    for step in range(steps):
        b = train if batch is None else {k: v[idx] for idx in [rng.choice(n, batch, replace=False)] for k, v in train.items()}
        step_seed = int(rng.integers(2**62)) if drop else None
        if grid and step % grid[0] == 0 and step < grid[1]:
            net.eval()
            update_grids(net, train["x"])
        gen = torch.Generator(device=net.device) if drop else None
        for layer in drop:
            layer._dropkan_generator = gen
        net.train()

        def closure():
            if drop:
                gen.manual_seed(step_seed)
            o.zero_grad()
            if lamb:
                with record_activations(net) as rec:
                    pred = net(b["x"])
                    loss = 1. * mse(pred, b["y"]) + lamb * KANRegularizer(1.).value(net, rec)
            else:
                loss = 1. * mse(net(b["x"]), b["y"])
            loss.backward()
            return loss
        if opt == "adam":
            closure()
            o.step()
        else:
            o.step(closure)
        net.eval()
        for layer in drop:
            layer._dropkan_generator = None











def same_params(a, b):
    return all(torch.equal(p, q) for p, q in zip(a.state_dict().values(), b.state_dict().values()))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", [None, 32])
@pytest.mark.parametrize("lamb", [0., 1e-3])
def test_adam_is_the_plain_loop(batch, lamb, device):
    a, b = pair(device=device)
    train, val = dataset(device=device)
    loss = CompositeLoss({"y": MSE()}, regularizers=[KANRegularizer(lamb)] if lamb else ())
    h = Trainer(a, loss, "adam", seed=3, batch_size=batch, callbacks=[GridUpdate(5, 20)]).fit(train, val, steps=25)
    plain_loop(b, train, 25, seed=3, batch=batch, grid=(4, 20), lamb=lamb)
    assert same_params(a, b)
    assert len(h["train/y"]) == len(h["val/y"]) == 25 and ("train/reg/kan" in h) == bool(lamb)








@pytest.mark.parametrize("lamb", [0., 1e-3])
def test_lbfgs_is_the_plain_loop(lamb):
    a, b = pair()
    train, val = dataset()
    loss = CompositeLoss({"y": MSE()}, regularizers=[KANRegularizer(lamb)] if lamb else ())
    h = Trainer(a, loss, "lbfgs", seed=3, callbacks=[GridUpdate(2, 4)]).fit(train, val, steps=5)
    plain_loop(b, train, 5, opt="lbfgs", lr=1., seed=3, grid=(2, 4), lamb=lamb)
    assert same_params(a, b) and h["train/y"][-1] < h["train/y"][0]


def test_dropkan_masks_are_the_plain_loops():
    a, b = pair(dropkan=0.2)
    train, val = dataset()
    Trainer(a, CompositeLoss({"y": MSE()}), "adam", seed=3).fit(train, val, steps=8)
    plain_loop(b, train, 8, seed=3)
    assert same_params(a, b)
    assert not a.training  # DropKAN models stay in eval mode outside training steps


def test_multihead_training_and_components():
    torch.manual_seed(0)
    x = torch.rand(64, 6)
    labels = (x[:, 0] > 0.5).long()
    model = MultiHead(KANNetwork([6, 4], seed=1), {"cls": KANNetwork([4, 2], seed=2), "recon": KANNetwork([4, 6], seed=3)})
    loss = CompositeLoss({"cls": (CrossEntropy(), 0.7), "recon": (MSE(), 0.3)}, regularizers=[KANRegularizer(1e-4)])
    trainer = Trainer(model, loss, "adam", lr=1e-2, callbacks=[GridUpdate(2, 4)])
    h = trainer.fit({"x": x, "cls": labels, "recon": x}, {"x": x, "cls": labels, "recon": x}, steps=15)
    assert {"train/cls", "train/recon", "train/reg/kan", "train/total", "val/cls", "val/recon", "val/total", "lr"} <= set(h)
    assert "val/reg/kan" not in h  # validation: the loss terms only
    assert h["train/total"][-1] < h["train/total"][0]
    assert np.allclose(np.array(h["train/total"]), np.array(h["train/cls"]) + np.array(h["train/recon"]) + np.array(h["train/reg/kan"]), rtol=1e-5)


def test_loader_epochs_and_steps():
    torch.manual_seed(0)
    x = torch.rand(100, 3)
    y = x.sum(1, keepdim=True)
    ds = torch.utils.data.StackDataset(x=x, y=y)
    loader = torch.utils.data.DataLoader(ds, batch_size=25, shuffle=True, generator=torch.Generator().manual_seed(0))
    net = KANNetwork([3, 3, 1], seed=1)
    trainer = Trainer(net, CompositeLoss({"y": MSE()}), "adam", callbacks=[GridUpdate(every=4, stop=8, rows="all")])
    h = trainer.fit(loader, loader, epochs=3)
    assert len(h["train/y"]) == 12 and trainer.epoch == 3 and trainer.step == 12
    h = trainer.fit(loader, steps=5)
    assert len(h["train/y"]) == 5 and trainer.step == 17
    with pytest.raises(ValueError):
        trainer.fit(loader)


def test_evaluation_in_chunks_is_one_loss_of_all_rows():
    new, _ = pair()
    train, val = dataset()
    a = Trainer(new, CompositeLoss({"y": MSE()})).evaluate(val)
    b = Trainer(new, CompositeLoss({"y": MSE()}), eval_rows=7).evaluate(val)
    assert torch.equal(a["y"], b["y"])


def test_early_stopping_best_state_and_logger(capsys):
    new, _ = pair()
    train, val = dataset()
    best = BestState(monitor="val/total")
    trainer = Trainer(new, CompositeLoss({"y": MSE()}), "adam", lr=0.5,
                      callbacks=[EarlyStopping(monitor="val/total", patience=2), best, Logger(every=1, keys=["val/total"])])
    h = trainer.fit(train, val, steps=200)
    assert len(h["val/total"]) < 200
    assert best.best == min(h["val/total"])
    after = Trainer(new, CompositeLoss({"y": MSE()})).evaluate(val)["total"]
    assert float(after) == pytest.approx(best.best)
    assert "val/total" in capsys.readouterr().out


def test_scheduler_steps():
    new, _ = pair()
    train, val = dataset()
    trainer = Trainer(new, CompositeLoss({"y": MSE()}), "adam", lr=1e-2,
                      scheduler=lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5))
    h = trainer.fit(train, steps=5)
    assert h["lr"] == [1e-2, 1e-2, 5e-3, 5e-3, 2.5e-3]


def test_resume_reproduces_one_fit():
    train, val = dataset()
    ref, a = KANNetwork([2, 5, 1], seed=1), KANNetwork([2, 5, 1], seed=1)
    loss = lambda: CompositeLoss({"y": MSE()})  # noqa: E731
    Trainer(ref, loss(), "adam", batch_size=16, seed=4).fit(train, steps=10)
    first = Trainer(a, loss(), "adam", batch_size=16, seed=4)
    first.fit(train, steps=6)
    state = copy.deepcopy(first.state_dict())
    b = KANNetwork([2, 5, 1], seed=9)
    second = Trainer(b, loss(), "adam", batch_size=16, seed=0)
    second.load_state_dict(state)
    second.fit(train, steps=4)
    assert all(torch.equal(p, q) for p, q in zip(ref.parameters(), b.parameters()))


def test_options_checked():
    new, _ = pair()
    with pytest.raises(ValueError, match="unknown optimizer"):
        Trainer(new, CompositeLoss({"y": MSE()}), "nonsense")
    with pytest.raises(ValueError, match="LBFGS"):
        Trainer(new, CompositeLoss({"y": MSE()}), "lbfgs", amp=True)
    with pytest.raises(ValueError, match="rows like"):
        Trainer(new, CompositeLoss({"y": MSE()})).fit({"x": torch.rand(5, 2), "y": torch.rand(4, 1)}, steps=1)
    with pytest.raises(ValueError, match="steps or epochs"):
        Trainer(new, CompositeLoss({"y": MSE()})).fit({"x": torch.rand(5, 2), "y": torch.rand(5, 1)})


def test_fit_is_the_trainer_composition():
    from xkan import fit
    train, val = dataset()
    a, b = KANNetwork([2, 5, 1], seed=4), KANNetwork([2, 5, 1], seed=4)
    h = fit(a, train, val, steps=12, opt="adam", lamb=1e-3, lamb_coef=0.1, grid_update_num=3, stop_grid_update_step=9,
            batch=40)
    loss = CompositeLoss({"y": MSE()}, [KANRegularizer(1e-3, coef=0.1)])
    ref = Trainer(b, loss, "adam", seed=4, batch_size=40, callbacks=[GridUpdate(3, 9)]).fit(train, val, steps=12)
    assert h == ref and all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    h = fit(KANNetwork([2, 5, 1], seed=4), train, val, steps=5)  # LBFGS, full batch, grid updates
    assert h["train/y"][-1] < h["train/y"][0]


def test_fastkan_style_composition_trains():
    # FastKAN / FasterKAN normalise every layer's input (LayerNorm) onto a fixed grid; here by composition: the layer
    # norms are ordinary modules around KANNetworks, which fit, the grid updates (on the normalised inputs) and the
    # regularisers handle like any model
    from torch import nn
    from xkan import KANNetwork, fit
    from xkan.nn import kan_layers
    torch.manual_seed(0)
    x = torch.randn(300, 6) * 3 + 1
    y = torch.sin(x[:, :1]) + 0.1 * x[:, 1:2] ** 2
    for basis, update_grid in (("rbf", False), ("rswaf", True)):
        model = nn.Sequential(nn.LayerNorm(6), KANNetwork([6, 8], basis=basis, grid_range=(-2, 2), seed=1),
                              nn.LayerNorm(8), KANNetwork([8, 1], basis=basis, grid_range=(-2, 2), seed=2))
        with torch.no_grad():
            before = float(torch.mean((model(x) - y) ** 2))
        history = fit(model, {"x": x, "y": y}, steps=30, opt="adam", lr=1e-2, lamb=1e-4, reg_metric="edge_weights",
                      update_grid=update_grid)
        assert history["train/y"][-1] < 0.5 * before and len(list(kan_layers(model))) == 2
