"""Checkpoints of KAN models: a directory with config.json (format version, the construction options, the symbolic
functions by name, metadata) and state.pt (the state dict, read with torch.load(weights_only=True))."""
import json
from pathlib import Path

import torch

from ..nn import KANNetwork, KANNetworkConfig, MultiHead

FORMAT, VERSION = "xkan-checkpoint", 2


def _describe(module):
    if isinstance(module, KANNetwork):
        cfg = module.config.to_dict()
        if cfg["layer"].get("base_fun", "silu") is None:
            raise ValueError("a network with a module as base function cannot be saved; use a base function name")
        names = [layer.names for layer in module.symbolic]
        if any("anonymous" in row for layer in names for row in layer):
            raise ValueError("a symbolic edge holds an unnamed function: add it to a SymbolicLibrary and fix the edge "
                             "with its name to save the network")
        return {"kind": "KANNetwork", "config": cfg, "symbolic": names}
    if isinstance(module, MultiHead):
        return {"kind": "MultiHead", "trunk": _describe(module.trunk),
                "heads": {name: _describe(head) for name, head in module.heads.items()}}
    raise TypeError(f"checkpoints hold KANNetworks and MultiHeads of them, not {type(module).__name__}")


def _build(desc, library):
    if desc["kind"] == "KANNetwork":
        net = KANNetwork.from_config(KANNetworkConfig.from_dict(desc["config"]))
        for layer, names in zip(net.symbolic, desc["symbolic"]):
            for j, row in enumerate(names):
                for i, name in enumerate(row):
                    if name != "0":
                        layer.functions[j][i] = library[name]
        return net
    if desc["kind"] == "MultiHead":
        return MultiHead(_build(desc["trunk"], library), {n: _build(h, library) for n, h in desc["heads"].items()})
    raise ValueError(f"unknown module kind {desc['kind']!r}")


def save_checkpoint(model, path, **metadata):
    """save model (a KANNetwork or a MultiHead of them) to the directory path; metadata: JSON-serialisable values"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    config = {"format": FORMAT, "version": VERSION, "model": _describe(model), "training": model.training,
              "metadata": metadata}
    (path / "config.json").write_text(json.dumps(config, indent=1))
    torch.save(model.state_dict(), path / "state.pt")


def load_checkpoint(path, map_location="cpu", library=None, return_metadata=False):
    """the model saved at path (its symbolic functions resolved by name in library, default the default library)"""
    from ..symbolic.library import default_library
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    if config.get("format") != FORMAT or config.get("version") != VERSION:
        raise ValueError(f"{path} is not a version {VERSION} KAN checkpoint")
    model = _build(config["model"], library or default_library())
    model.load_state_dict(torch.load(path / "state.pt", map_location=map_location, weights_only=True))
    model.to(map_location)
    model.train(config.get("training", True))
    return (model, config["metadata"]) if return_metadata else model


__all__ = ["save_checkpoint", "load_checkpoint", "FORMAT", "VERSION"]
