"""The symbolic branch of a network layer: edge (input i, output j), when its mask is set, computes
mask_ji * (c * f(a * x_i + b) + d) with affine_ji = (a, b, c, d); each output the sum of its edges."""
from typing import Callable, NamedTuple

import torch


class SymbolicFunction(NamedTuple):
    """a function of the symbolic library: its name, torch and sympy forms, complexity and the singularity-avoiding
    form (x, y_th) -> (penalty, value)"""
    name: str
    fn: Callable
    sympy: Callable
    complexity: float
    singular: Callable


def _zero(x):
    return x * 0.


def _zero_singular(x, y_th):
    return (), x * 0.


ZERO = SymbolicFunction("0", _zero, _zero, 0, _zero_singular)


class SymbolicLayer(torch.nn.Module):
    """
    The symbolic branch of a layer of in_dim inputs and out_dim outputs.

    mask (out_dim, in_dim; a buffer) switches edges on, affine (out_dim, in_dim, 4) holds their (a, b, c, d). The
    edges with a non-zero mask are kept as Python state, so that a forward needs no device synchronisation: change
    the mask through set_mask (or call refresh() after writing it directly); load_state_dict refreshes it.
    layer(x) returns the output (batch, out_dim) and the per-edge values (batch, out_dim, in_dim).
    """

    def __init__(self, in_dim, out_dim, *, device=None, dtype=None):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.register_buffer("mask", torch.zeros(out_dim, in_dim, device=device, dtype=dtype))
        self.affine = torch.nn.Parameter(torch.zeros(out_dim, in_dim, 4, device=device, dtype=dtype))
        self.functions = [[ZERO] * in_dim for _ in range(out_dim)]
        self._active = frozenset()

    def __getstate__(self):
        # functions of the default library by name (their lambdas do not pickle); others as they are
        from ..symbolic.library import SYMBOLIC_LIB
        state = self.__dict__.copy()
        state["functions"] = [[f.name if f.name in SYMBOLIC_LIB and f.fn is SYMBOLIC_LIB[f.name][0] else f for f in row]
                              for row in self.functions]
        return state

    def __setstate__(self, state):
        from ..symbolic.library import default_library
        lib = None
        rows = []
        for row in state["functions"]:
            if any(isinstance(f, str) for f in row) and lib is None:
                lib = default_library()
            rows.append([lib[f] if isinstance(f, str) else f for f in row])
        state["functions"] = rows
        super().__setstate__(state)

    @property
    def active(self):
        """whether any edge is on"""
        return bool(self._active)

    @property
    def names(self):
        return [[f.name for f in row] for row in self.functions]

    def refresh(self):
        """re-read the edges that are on from the mask"""
        self._active = frozenset(map(tuple, torch.nonzero(self.mask.detach()).tolist()))

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.refresh()

    def set_function(self, j, i, function, affine=None):
        """edge (j, i) computes function (a SymbolicFunction) with affine (a, b, c, d) (default (1, 0, 1, 0))"""
        self.functions[j][i] = function
        with torch.no_grad():
            self.affine[j, i] = torch.as_tensor((1., 0., 1., 0.) if affine is None else affine, dtype=self.affine.dtype)

    def set_mask(self, j, i, value):
        with torch.no_grad():
            self.mask[j, i] = value
        self.refresh()

    def forward(self, x, singularity_avoiding=False, y_th=1000.):
        # pykan's computation (a stack of every edge, inactive ones zero, summed over the inputs), bit for bit
        zero = None
        y_th = torch.tensor(y_th)
        postacts = []
        for i in range(self.in_dim):
            column = []
            for j in range(self.out_dim):
                if (j, i) not in self._active:
                    if zero is None:
                        zero = torch.zeros_like(x[:, [i]] * self.affine[0, 0, 0])
                    column.append(zero)
                    continue
                a, b, c, d = self.affine[j, i]
                f = self.functions[j][i]
                v = f.singular(a * x[:, [i]] + b, y_th)[1] if singularity_avoiding else f.fn(a * x[:, [i]] + b)
                column.append(self.mask[j][i] * (c * v + d))
            postacts.append(torch.stack(column))
        postacts = torch.stack(postacts).permute(2, 1, 0, 3)[:, :, :, 0]
        return torch.sum(postacts, dim=2), postacts

    def subset(self, in_id, out_id):
        """the layer of the inputs in_id and the outputs out_id"""
        return self.reindexed(list(in_id), list(out_id))

    def reindexed(self, in_map, out_map):
        """a layer whose input i is this layer's input in_map[i] and output j its output out_map[j]; None in a map:
        a new input / output without symbolic edges"""
        in_map = [None if i is None else int(i) for i in in_map]
        out_map = [None if j is None else int(j) for j in out_map]
        new = SymbolicLayer(len(in_map), len(out_map), device=self.mask.device, dtype=self.mask.dtype)
        ii = [k for k, i in enumerate(in_map) if i is not None]
        jj = [k for k, j in enumerate(out_map) if j is not None]
        dev = self.mask.device
        src_i = torch.tensor([in_map[k] for k in ii], dtype=torch.long, device=dev)
        src_j = torch.tensor([out_map[k] for k in jj], dtype=torch.long, device=dev)
        rows = torch.tensor(jj, dtype=torch.long, device=dev)[:, None]
        cols = torch.tensor(ii, dtype=torch.long, device=dev)[None, :]
        with torch.no_grad():
            new.mask[rows, cols] = self.mask[src_j][:, src_i]
            new.affine[rows, cols] = self.affine[src_j][:, src_i]
        for nj in jj:
            row, old = new.functions[nj], self.functions[out_map[nj]]
            for ni in ii:
                row[ni] = old[in_map[ni]]
        new.affine.requires_grad_(self.affine.requires_grad)
        new.refresh()
        return new

    def swap(self, i1, i2, mode="in"):
        """swap inputs (mode 'in') or outputs ('out') i1 and i2"""
        with torch.no_grad():
            if mode == "in":
                for row in self.functions:
                    row[i1], row[i2] = row[i2], row[i1]
                self.mask[:, [i1, i2]] = self.mask[:, [i2, i1]].clone()
                self.affine[:, [i1, i2]] = self.affine[:, [i2, i1]].clone()
            else:
                self.functions[i1], self.functions[i2] = self.functions[i2], self.functions[i1]
                self.mask[[i1, i2]] = self.mask[[i2, i1]].clone()
                self.affine[[i1, i2]] = self.affine[[i2, i1]].clone()
        self.refresh()

    def extra_repr(self):
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}, active={len(self._active)}"
