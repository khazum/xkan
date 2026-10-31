"""Kolmogorov-Arnold Networks.

KANNetwork (xkan.nn) computes; Trainer / fit (xkan.train) and the losses (xkan.losses) train; xkan.interp interprets and
prunes; xkan.symbolic finds formulas; xkan.io saves and loads; xkan.viz plots. KANLayer (xkan.layers) is the layer.
"""
__version__ = "1.0.0.dev0"

from . import interp, io, layers, losses, nn, spline, symbolic, train, utils, viz
from .interp import record_activations
from .layers import KANLayer, KANLayerConfig
from .losses import CompositeLoss
from .nn import KANNetwork, KANNetworkConfig, MultiHead
from .train import GridUpdate, Trainer, fit
from .utils import create_dataset, create_dataset_from_data

__all__ = ["KANNetwork", "KANNetworkConfig", "MultiHead", "KANLayer", "KANLayerConfig", "record_activations",
           "CompositeLoss", "Trainer", "GridUpdate", "fit", "create_dataset", "create_dataset_from_data", "layers", "nn",
           "interp", "losses", "train", "symbolic", "io", "viz", "spline", "utils"]
