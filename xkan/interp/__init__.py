"""Interpretation tools: activations, attribution, pruning and structural changes of KAN networks."""
from .attribution import Attribution, network_attribution, node_attribution
from .recording import NetworkRecord, paused_recording, record_activations, recorded
from .structure import (attribute, auto_swap, connection_cost, edge_range, expand_depth, expand_width, feature_interaction,
                        feature_score, module, n_edges, perturb, prune, prune_edge, prune_input, prune_node, refine,
                        reindex_nodes, remove_edge, remove_node, swap)

__all__ = ["NetworkRecord", "record_activations", "paused_recording", "recorded", "Attribution", "network_attribution",
           "node_attribution", "attribute", "feature_score", "feature_interaction", "prune", "prune_node", "prune_edge",
           "prune_input", "refine", "expand_depth", "expand_width", "swap", "auto_swap", "connection_cost",
           "remove_edge", "remove_node", "module", "perturb", "n_edges", "edge_range", "reindex_nodes"]
