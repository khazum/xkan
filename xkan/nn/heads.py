"""Several named heads on a shared trunk."""
from torch import nn

from .network import _NoCompiledCopy


class MultiHead(_NoCompiledCopy, nn.Module):
    """
    MultiHead(trunk, heads): the trunk (any module) evaluated once, then every head (a name -> module mapping) on
    its output; model(x) returns {name: head output}. Trunk and heads may be KANNetworks or any other modules (a
    head of an encoder's latent, say, a classifier and a decoder).
    """

    def __init__(self, trunk, heads):
        super().__init__()
        if not heads:
            raise ValueError("a MultiHead needs at least one head")
        self.trunk = trunk
        self.heads = nn.ModuleDict(heads)

    def forward(self, x):
        shared = self.trunk(x)
        return {name: head(shared) for name, head in self.heads.items()}
