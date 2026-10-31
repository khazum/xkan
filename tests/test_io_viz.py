"""Checkpoints (networks, pruned and symbolic ones, multi-head models: the same outputs after a round trip) and the
network plot (a smoke test)."""
import json

import pytest
import torch

from xkan import KANNetwork, MultiHead
from xkan.interp import prune_input, refine
from xkan.io import load_checkpoint, save_checkpoint
from xkan.symbolic import default_library, fix_symbolic


def test_round_trip_of_networks(tmp_path):
    torch.manual_seed(0)
    x = torch.rand(100, 3) * 2 - 1
    net = KANNetwork([3, [3, 1], 2], grid=4, basis="rbf", seed=5)
    fix_symbolic(net, 0, 1, 2, "sin", x)
    variants = [net, prune_input(net, x, keep=[0, 2]), refine(net, 7, x)]
    for k, model in enumerate(variants):
        save_checkpoint(model, tmp_path / str(k), note=k)
        back, meta = load_checkpoint(tmp_path / str(k), return_metadata=True)
        assert meta == {"note": k} and back.config == model.config
        assert torch.equal(back(x), model(x)) and back.symbolic[0].names == model.symbolic[0].names
    cfg = json.loads((tmp_path / "0" / "config.json").read_text())
    assert cfg["format"] == "xkan-checkpoint" and cfg["version"] == 2


def test_round_trip_of_a_multihead_and_custom_functions(tmp_path):
    model = MultiHead(KANNetwork([4, 3], seed=1), {"a": KANNetwork([3, 2], seed=2), "b": KANNetwork([3, 1], seed=3)})
    lib = default_library().add("cube", lambda t: t ** 3, None, 3)
    fix_symbolic(model.heads["b"], 0, 0, 0, "cube", fit=False, library=lib)
    save_checkpoint(model, tmp_path)
    back = load_checkpoint(tmp_path, library=lib)
    x = torch.rand(20, 4)
    out, ref = back(x), model(x)
    assert set(out) == {"a", "b"} and all(torch.equal(out[k], ref[k]) for k in out)
    with pytest.raises(KeyError):
        load_checkpoint(tmp_path)  # 'cube' is not in the default library


def test_unsavable_models(tmp_path):
    net = KANNetwork([2, 1])
    fix_symbolic(net, 0, 0, 0, torch.cos, fit=False)
    with pytest.raises(ValueError, match="unnamed"):
        save_checkpoint(net, tmp_path / "a")
    with pytest.raises(TypeError):
        save_checkpoint(torch.nn.Linear(2, 1), tmp_path / "b")


def test_plot_smoke(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from xkan.viz import plot
    torch.manual_seed(0)
    x = torch.rand(50, 2) * 2 - 1
    net = KANNetwork([2, [2, 1], 1], seed=1)
    fix_symbolic(net, 0, 0, 1, "x^2", x)
    for metric in ("backward", "forward_n", "forward_u"):
        fig = plot(net, x, folder=str(tmp_path), metric=metric, in_vars=["a", "b"], out_vars=["y"], title="t")
        assert fig is not None
    assert (tmp_path / "sp_0_0_1.png").exists()
    # without a folder the edge images stay in memory: nothing is written (the working directory stays empty)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    plot(net, x)
    assert list(work.iterdir()) == []
    with pytest.raises(ValueError):
        plot(net, x, folder=str(tmp_path), metric="nonsense")


def test_plot_edge_smoke():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from xkan.viz import plot_edge
    net = KANNetwork([2, 3, 1], seed=1)
    x = torch.rand(30, 2)
    xs, ys = plot_edge(net, x, 0, 1, 2)
    assert xs.shape == ys.shape == (30,) and (xs[1:] >= xs[:-1]).all()
