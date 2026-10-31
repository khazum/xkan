"""Networks of KAN layers and their composition."""
from .heads import MultiHead
from .network import NETWORK_LAYER_DEFAULTS, KANNetwork, KANNetworkConfig
from .traversal import kan_layers, kan_networks, update_grids

__all__ = ["KANNetwork", "KANNetworkConfig", "NETWORK_LAYER_DEFAULTS", "MultiHead", "kan_layers", "kan_networks",
           "update_grids"]
