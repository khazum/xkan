"""Interpretation tools: activations, attribution, pruning and structural changes of KAN networks."""
from .attribution import Attribution, network_attribution, node_attribution
from .diagnostics import layer_inputs, outside_grid, spline_share
from .recording import NetworkRecord, paused_recording, record_activations, recorded
from .structure import (attribute, auto_swap, connection_cost, disconnected_outputs, edge_range, expand_depth,
                        expand_width, feature_interaction, feature_score, module, n_edges, perturb, prune, prune_edge,
                        prune_input, prune_node, prune_to_budget, refine, reindex_nodes, remove_edge, remove_node, swap)

__all__ = ["NetworkRecord", "record_activations", "paused_recording", "recorded", "layer_inputs", "outside_grid",
           "spline_share", "Attribution", "network_attribution",
           "node_attribution", "attribute", "feature_score", "feature_interaction", "prune", "prune_node", "prune_edge",
           "prune_input", "prune_to_budget", "disconnected_outputs", "refine", "expand_depth", "expand_width", "swap",
           "auto_swap", "connection_cost", "remove_edge", "remove_node", "module", "perturb", "n_edges", "edge_range",
           "reindex_nodes"]
