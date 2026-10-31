"""Finding the KAN parts of a module tree, and the grid update of all its layers."""
import torch

from ..layers import KANLayer


def kan_networks(model):
    """the KANNetworks in model (model itself included), in module order, each once"""
    from .network import KANNetwork
    return [m for m in model.modules() if isinstance(m, KANNetwork)]


def kan_layers(model):
    """the KANLayers in model, in module order, each once"""
    return [m for m in model.modules() if isinstance(m, KANLayer)]


@torch.no_grad()
def update_grids(model, x, mode="sample", **forward_kwargs):
    """
    Update the grid of every KAN layer of model from its inputs in model(x, **forward_kwargs): one forward without
    autograd, each layer updated by a forward pre-hook from its input, so that the next layer sees the updated one.
    mode: the refit mode of KANLayer.update_grid_from_samples ('sample', 'grid' or 'sample+uniform').
    """
    from ..interp.recording import paused_recording
    handles = [layer.register_forward_pre_hook(lambda layer, inputs: layer.update_grid_from_samples(inputs[0], mode=mode))
               for layer in kan_layers(model)]
    try:
        with paused_recording(model):
            model(x, **forward_kwargs)
    finally:
        for handle in handles:
            handle.remove()
