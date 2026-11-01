"""The configuration of a KAN layer: one place for its options, their defaults and their validation."""
from dataclasses import asdict, dataclass

import torch


class Zero(torch.nn.Module):
    """The zero base function: 0 * x (keeps NaN and inf as pykan 0.2.8's lambda x: x * 0. did)."""

    def forward(self, x):
        return x * 0.


# what an edge does with an input outside its grid's inner range (KANLayer)
OUT_OF_GRID = ("zero", "clamp", "soft_clamp")

# base functions b(x) of the residual path, by name
BASE_FUNCTIONS = {"silu": torch.nn.SiLU, "identity": torch.nn.Identity, "zero": Zero, "gelu": torch.nn.GELU,
                  "relu": torch.nn.ReLU, "tanh": torch.nn.Tanh}


def base_function(spec):
    """The base function module for a name in BASE_FUNCTIONS, or spec itself (a module or a callable)."""
    if isinstance(spec, str):
        if spec not in BASE_FUNCTIONS:
            raise ValueError(f"unknown base function {spec!r}; one of {sorted(BASE_FUNCTIONS)}, or a module")
        return BASE_FUNCTIONS[spec]()
    if not callable(spec):
        raise ValueError(f"base_fun must be a name, a module or a callable, got {spec!r}")
    return spec


@dataclass(frozen=True)
class KANLayerConfig:
    """
    The options of a KANLayer (see KANLayer for their meaning); validated on construction.

    base_fun is a name of BASE_FUNCTIONS (serialisable) or a module / callable (to_dict gives None for it).
    """
    in_dim: int
    out_dim: int
    num: int = 5
    k: int = 3
    basis: str = "bspline"
    base_fun: object = "silu"
    noise_scale: float = 0.5
    scale_base_mu: float = 0.0
    scale_base_sigma: float = 1.0
    scale_sp: float = 1.0
    scale_sp_fan_in: bool = True
    grid_eps: float = 0.02
    grid_range: tuple = (-1, 1)
    sp_trainable: bool = True
    sb_trainable: bool = True
    sparse_init: bool = False
    knot_placement: str = "quantile"
    min_knot_gap: float | None = None
    refit_guard: float | None = None
    dropkan: float = 0.0
    aggregation: str = "sum"
    bias: bool = False
    out_of_grid: str = "zero"

    def __post_init__(self):
        from .bases import BASES
        from .grids import KNOT_PLACEMENTS
        if self.in_dim < 1 or self.out_dim < 1:
            raise ValueError(f"in_dim and out_dim must be at least 1, got {self.in_dim}, {self.out_dim}")
        if self.num < 1 or self.k < 0:
            raise ValueError(f"num must be at least 1 and k at least 0, got {self.num}, {self.k}")
        if self.basis not in BASES:
            raise ValueError(f"unknown basis {self.basis!r}; one of {sorted(BASES)}")
        if self.knot_placement not in KNOT_PLACEMENTS:
            raise ValueError(f"unknown knot_placement {self.knot_placement!r}; one of {sorted(KNOT_PLACEMENTS)}")
        if self.out_of_grid not in OUT_OF_GRID:
            raise ValueError(f"unknown out_of_grid {self.out_of_grid!r}; one of {OUT_OF_GRID}")
        if self.aggregation not in ("sum", "mean"):
            raise ValueError(f"unknown aggregation {self.aggregation!r}; 'sum' or 'mean'")
        if not 0. <= float(self.dropkan) < 1.:
            raise ValueError(f"dropkan must be in [0, 1), got {self.dropkan}")
        if self.min_knot_gap is not None and not float(self.min_knot_gap) > 0.:
            raise ValueError(f"min_knot_gap must be None or positive, got {self.min_knot_gap}")
        if self.refit_guard is not None and not float(self.refit_guard) > 0.:
            raise ValueError(f"refit_guard must be None or positive, got {self.refit_guard}")
        if isinstance(self.base_fun, str):
            base_function(self.base_fun)

    def to_dict(self):
        """the options as plain values (a base function given as a module: None)"""
        d = asdict(self)
        d["base_fun"] = self.base_fun if isinstance(self.base_fun, str) else None
        d["grid_range"] = list(self.grid_range)
        return d
