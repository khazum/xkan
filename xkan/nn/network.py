"""KAN networks: a stack of KAN layers with their symbolic branches, node / subnode affine transforms and
multiplication nodes. A network only computes; training, interpretation and symbolic tools act on it from outside."""
import copy
from dataclasses import dataclass, field
from numbers import Integral

import torch
from torch import nn

from ..layers import KANLayer, KANLayerConfig
from ..layers.symbolic import SymbolicLayer

# defaults of the layers of a network that differ from a single layer's (pykan's network defaults)
NETWORK_LAYER_DEFAULTS = {"noise_scale": 0.3}


def _width(width):
    out = []
    for w in width:
        n = (int(w), 0) if isinstance(w, Integral) else tuple(int(v) for v in w)
        if len(n) != 2 or min(n) < 0 or sum(n) < 1:
            raise ValueError(f"a width is a positive int or [n_sum, n_mult], got {w!r}")
        out.append(n)
    if len(out) < 2:
        raise ValueError("width needs an input and an output layer")
    if out[0][1]:
        raise ValueError("the input layer has no multiplication nodes")
    return tuple(out)


@dataclass(frozen=True)
class KANNetworkConfig:
    """
    The construction options of a KANNetwork (validated; to_dict / from_dict for checkpoints).

    width : per node layer (input first) an int or (n_sum, n_mult): addition and multiplication nodes
    grid, k : grid intervals and spline order, one value or one per layer
    mult_arity : the arity of every multiplication node (int), or per node layer a list of arities
    seed : the seed of the initial parameters
    affine_trainable : whether the node / subnode scales and biases train
    in_features : the columns of the network's input (default width[0]; more after prune_input, which keeps some)
    layer : KANLayerConfig options of every layer (basis, base_fun, noise_scale, grid_eps, knot_placement, ...)
    """
    width: tuple
    grid: object = 3
    k: object = 3
    mult_arity: object = 2
    seed: int = 1
    affine_trainable: bool = False
    in_features: int | None = None
    layer: dict = field(default_factory=dict)

    def __post_init__(self):
        width = _width(self.width)
        object.__setattr__(self, "width", width)
        depth = len(width) - 1
        for name in ("grid", "k"):
            v = getattr(self, name)
            if isinstance(v, (list, tuple)):
                if len(v) != depth:
                    raise ValueError(f"{name} needs one value per layer ({depth}), got {v!r}")
                object.__setattr__(self, name, tuple(int(a) for a in v))
        if isinstance(self.mult_arity, Integral):
            if self.mult_arity < 2:
                raise ValueError("a multiplication node multiplies at least 2 subnodes")
        else:
            arity = tuple(tuple(int(a) for a in row) for row in self.mult_arity)
            if len(arity) != len(width) or any(len(row) != w[1] or min(row, default=2) < 2 for row, w in zip(arity, width)):
                raise ValueError("mult_arity: per node layer one arity (>= 2) per multiplication node")
            object.__setattr__(self, "mult_arity", arity)
        if self.in_features is not None and self.in_features < width[0][0]:
            raise ValueError(f"in_features ({self.in_features}) is less than the input width ({width[0][0]})")
        unknown = set(self.layer) - (set(KANLayerConfig.__dataclass_fields__) - {"in_dim", "out_dim", "num", "k"})
        if unknown:
            raise TypeError(f"unknown layer options {sorted(unknown)}")
        object.__setattr__(self, "layer", {**NETWORK_LAYER_DEFAULTS, **self.layer})
        for l in range(depth):
            self.layer_config(l)

    @property
    def depth(self):
        return len(self.width) - 1

    def arities(self, l):
        """the arities of the multiplication nodes of node layer l"""
        if isinstance(self.mult_arity, Integral):
            return (self.mult_arity,) * self.width[l][1]
        return self.mult_arity[l]

    @property
    def width_in(self):
        """the nodes of every node layer"""
        return [n_sum + n_mult for n_sum, n_mult in self.width]

    @property
    def width_out(self):
        """the subnodes of every node layer (the outputs of the layer before it)"""
        return [w[0] + sum(self.arities(l)) for l, w in enumerate(self.width)]

    def layer_config(self, l):
        grid = self.grid[l] if isinstance(self.grid, tuple) else self.grid
        k = self.k[l] if isinstance(self.k, tuple) else self.k
        return KANLayerConfig(self.width_in[l], self.width_out[l + 1], num=grid, k=k, **self.layer)

    def to_dict(self):
        """plain values (a base function given as a module: None)"""
        d = {"width": [list(w) for w in self.width], "grid": self.grid, "k": self.k,
             "mult_arity": self.mult_arity if isinstance(self.mult_arity, Integral) else [list(r) for r in self.mult_arity],
             "seed": self.seed, "affine_trainable": self.affine_trainable, "in_features": self.in_features}
        layer = dict(self.layer)
        if not isinstance(layer.get("base_fun", "silu"), str):
            layer["base_fun"] = None
        if "grid_range" in layer:
            layer["grid_range"] = list(layer["grid_range"])
        d["layer"] = layer
        return {key: list(v) if isinstance(v, tuple) else v for key, v in d.items()}

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        layer = dict(d.pop("layer", {}))
        if "grid_range" in layer:
            layer["grid_range"] = tuple(layer["grid_range"])
        if layer.get("base_fun", "") is None:
            raise ValueError("the configuration's base function is not serialisable (a module): pass it as base_fun")
        return cls(**d, layer=layer)


class _NoCompiledCopy:
    """copies (deepcopy, as the structural tools make) drop the compiled forward of module.compile(), which is bound
    to the original module"""

    def __deepcopy__(self, memo):
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        state = {key: v for key, v in self.__dict__.items() if key not in ("_compiled_call_impl", "_recorders")}
        new.__setstate__(copy.deepcopy(state, memo))
        return new


class KANNetwork(_NoCompiledCopy, nn.Module):
    """
    A KAN network.

    KANNetwork(width, grid=3, k=3, *, mult_arity=2, seed=1, affine_trainable=False, device=None, dtype=None,
    **layer_options): see KANNetworkConfig; layer_options are KANLayerConfig fields of every layer (for networks,
    noise_scale defaults to 0.3).

    Node layer l + 1 is computed from node layer l by layers[l] (numeric edges) plus symbolic[l] (symbolic edges),
    the subnode affine transform, the multiplication nodes (each the product of its arity of consecutive subnodes
    after the addition subnodes) and the node affine transform. The input columns input_id of x (all, unless
    prune_input kept some) enter node layer 0.

    network(x) returns the output (batch, width[-1]); inside xkan.interp.record_activations(network) every forward is
    also recorded, with the per-edge values. Without symbolic edges and without recording the forward is the layers'
    fused one, with no synchronisation of the device.
    """

    _recorders = ()

    def __init__(self, width, grid=3, k=3, *, mult_arity=2, seed=1, affine_trainable=False, in_features=None,
                 device=None, dtype=None, **layer_options):
        super().__init__()
        self._build(KANNetworkConfig(width, grid, k, mult_arity, seed, affine_trainable, in_features, layer_options))
        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    @classmethod
    def from_config(cls, config, device=None, dtype=None):
        new = cls.__new__(cls)
        nn.Module.__init__(new)
        new._build(config)
        if device is not None or dtype is not None:
            new.to(device=device, dtype=dtype)
        return new

    def _build(self, cfg):
        self._config = cfg
        width_in, width_out = cfg.width_in, cfg.width_out
        # the initial parameters from a generator seeded with cfg.seed (the caller's global stream unchanged); the
        # draws in pykan's order
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.seed)
            self.layers = nn.ModuleList(KANLayer.from_config(cfg.layer_config(l)) for l in range(cfg.depth))
        self.symbolic = nn.ModuleList(SymbolicLayer(width_in[l], width_out[l + 1]) for l in range(cfg.depth))
        ones = lambda n: nn.Parameter(torch.ones(n), requires_grad=cfg.affine_trainable)  # noqa: E731
        zeros = lambda n: nn.Parameter(torch.zeros(n), requires_grad=cfg.affine_trainable)  # noqa: E731
        self.subnode_scale = nn.ParameterList(ones(width_out[l + 1]) for l in range(cfg.depth))
        self.subnode_bias = nn.ParameterList(zeros(width_out[l + 1]) for l in range(cfg.depth))
        self.node_scale = nn.ParameterList(ones(width_in[l + 1]) for l in range(cfg.depth))
        self.node_bias = nn.ParameterList(zeros(width_in[l + 1]) for l in range(cfg.depth))
        self.register_buffer("input_id", torch.arange(width_in[0]))
        self._setup()
        if cfg.layer.get("dropkan", 0.) > 0.:
            self.eval()  # DropKAN applies in training forwards only; a trainer switches to training mode

    @classmethod
    def from_layers(cls, layers, *, affine_trainable=False):
        """
        A network of existing KANLayer objects, used as they are (their parameters are the network's, not copies):
        layers[l] maps node layer l to node layer l + 1 (addition nodes only), with identity affine transforms and no
        symbolic edges. It lets the network tools (record_activations, attribution, formulas, plots) work on layers
        built elsewhere, e.g. by a subclass of KANLayer; the configuration records the first layer's options, so the
        tools that rebuild a network (refine, expand, swap, pruning) return one of plain KANLayers.
        """
        layers = list(layers)
        if not layers:
            raise ValueError("from_layers needs at least one layer")
        for l, (a, b) in enumerate(zip(layers, layers[1:])):
            if a.out_dim != b.in_dim:
                raise ValueError(f"layer {l} has {a.out_dim} outputs, layer {l + 1} {b.in_dim} inputs")
        options = {k: v for k, v in vars(layers[0].config).items() if k not in ("in_dim", "out_dim", "num", "k")}
        config = KANNetworkConfig([layers[0].in_dim] + [layer.out_dim for layer in layers],
                                  tuple(layer.num for layer in layers), tuple(layer.k for layer in layers),
                                  affine_trainable=affine_trainable, layer=options)
        widths = config.width_out[1:]
        p = layers[0].coef
        affine = tuple([fill(n, dtype=p.dtype, device=p.device) for n in widths]
                       for fill in (torch.ones, torch.zeros, torch.ones, torch.zeros))
        symbolic = [SymbolicLayer(layer.in_dim, layer.out_dim, device=p.device, dtype=p.dtype) for layer in layers]
        return cls._assemble(config, layers, symbolic, affine, torch.arange(layers[0].in_dim, device=p.device),
                             training=layers[0].training)

    def _setup(self):
        # derived structure: whether the input columns are all used as they are, and the multiplication nodes per
        # layer (homogeneous arities as one strided product)
        cfg = self._config
        self._all_inputs = self.input_id.numel() == self.in_features and bool(
            torch.equal(self.input_id.cpu(), torch.arange(self.in_features)))
        self._mult = []
        for l in range(1, cfg.depth + 1):
            n_sum, n_mult = cfg.width[l]
            arities = cfg.arities(l)
            self._mult.append((n_sum, n_mult, arities[0] if len(set(arities)) == 1 and n_mult else None, arities))

    @classmethod
    def _assemble(cls, config, layers, symbolic, affine, input_id, training=True):
        """a network of existing parts (no initialisation): affine = (subnode_scale, subnode_bias, node_scale,
        node_bias), each a list of tensors per layer (copied)"""
        new = cls.__new__(cls)
        nn.Module.__init__(new)
        new._config = config
        new.layers = nn.ModuleList(layers)
        new.symbolic = nn.ModuleList(symbolic)
        for name, values in zip(("subnode_scale", "subnode_bias", "node_scale", "node_bias"), affine):
            setattr(new, name, nn.ParameterList(nn.Parameter(v.detach().clone(), requires_grad=config.affine_trainable)
                                                for v in values))
        new.register_buffer("input_id", input_id.detach().clone())
        new._setup()
        new.train(training)
        return new

    # ------------------------------------------------------------------------------------------------- structure

    @property
    def config(self):
        return self._config

    @property
    def width(self):
        return [list(w) for w in self._config.width]

    @property
    def width_in(self):
        return self._config.width_in

    @property
    def width_out(self):
        return self._config.width_out

    @property
    def depth(self):
        return self._config.depth

    @property
    def in_features(self):
        """the columns of the input"""
        return self._config.in_features or self.width_in[0]

    @property
    def device(self):
        return self.layers[0].grid.device

    @property
    def dtype(self):
        return self.layers[0].coef.dtype

    def input_names(self, names=None, prefix="x_"):
        """the names of the network's inputs: for the input columns they use (input_id) out of names given per column
        of the input (in_features of them; default prefix + column number from 1), or names given per network input
        (width_in[0] of them)"""
        cols = self.input_id.tolist()
        if names is None:
            return [f"{prefix}{c + 1}" for c in cols]
        names = list(names)
        if len(names) == self.in_features:
            return [names[c] for c in cols]
        if len(names) == self.width_in[0]:
            return names
        raise ValueError(f"give {self.in_features} names (one per input column) or {self.width_in[0]} (one per network "
                         f"input), got {len(names)}")

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._setup()

    # ------------------------------------------------------------------------------------------------- forward

    def _multiply(self, l, x):
        n_sum, n_mult, arity, arities = self._mult[l]
        if not n_mult:
            return x
        if arity is not None:
            prod = x[:, n_sum::arity] * x[:, n_sum + 1::arity]
            for i in range(2, arity):
                prod = prod * x[:, n_sum + i::arity]
        else:
            products, start = [], n_sum
            for a in arities:
                p = x[:, [start]] * x[:, [start + 1]]
                for i in range(2, a):
                    p = p * x[:, [start + i]]
                products.append(p)
                start += a
            prod = torch.cat(products, dim=1)
        return torch.cat([x[:, :n_sum], prod], dim=1)

    def forward(self, x, singularity_avoiding=False, y_th=1000.):
        if x.shape[1] != self.in_features:
            raise ValueError(f"the network takes inputs of {self.in_features} columns, got {x.shape[1]}")
        x = (x if self._all_inputs else x[:, self.input_id]).to(self.dtype)
        record = _Record() if self._recorders else None
        for l in range(self.depth):
            layer, sym = self.layers[l], self.symbolic[l]
            if record is None:
                y = layer(x)
            else:
                edges = layer(x, return_edges=True)
                y, edge_out = edges.y, edges.postacts
            if sym.active:
                y_sym, edges_sym = sym(x, singularity_avoiding=singularity_avoiding, y_th=y_th)
                y = y + y_sym
                if record is not None:
                    edge_out = edge_out + edges_sym
            subnode = y
            y = self.subnode_scale[l][None, :] * y + self.subnode_bias[l][None, :]
            premult = y
            y = self._multiply(l, y)
            y = self.node_scale[l][None, :] * y + self.node_bias[l][None, :]
            if record is not None:
                record.add(x, edges, edge_out, subnode, premult, y)
            x = y
        if record is not None:
            for recorder in self._recorders:
                recorder._store(self, record)
        return x


class _Record:
    """the per-layer values of one forward, collected for xkan.interp.NetworkRecord"""

    def __init__(self):
        self.inputs, self.edge_outputs, self.edge_numeric, self.edge_spline = [], [], [], []
        self.subnode, self.premult, self.outputs = [], [], []

    def add(self, x, edges, edge_out, subnode, premult, y):
        self.inputs.append(x)
        self.edge_numeric.append(edges.postacts)
        self.edge_spline.append(edges.postspline)
        self.edge_outputs.append(edge_out)
        self.subnode.append(subnode)
        self.premult.append(premult)
        self.outputs.append(y)
