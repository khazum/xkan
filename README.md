# xKAN: Kolmogorov-Arnold Networks, up to date

xKAN continues [pykan](https://github.com/KindXiaoming/pykan), the code Ziming Liu and colleagues wrote for
["KAN: Kolmogorov-Arnold Networks"](https://arxiv.org/abs/2404.19756) and
["KAN 2.0: Kolmogorov-Arnold Networks Meet Science"](https://arxiv.org/abs/2408.10205). pykan's last release, 0.2.8,
dates from November 2024 and its last commit from January 2025. Other projects improved single parts of it:
[efficient-KAN](https://github.com/Blealtan/efficient-kan), [FastKAN](https://github.com/ZiyaoLi/fast-kan),
[FasterKAN](https://github.com/AthanasiosDelis/faster-kan), [Deep-KAN](https://github.com/Sid2690/Deep-KAN), ChebyKAN
and [KANLib](https://github.com/julianhoever/KANLib). xKAN fixes pykan, moves it to current Python and PyTorch, and
combines those improvements with pykan's interpretability tools: pruning, attribution, symbolic regression, formulas
and plots.

- **Fixes.** xKAN fixes 66 problems reported in pykan's issues, plus numerical ones found in testing: rank-deficient
  spline fits, NaN gradients at coincident knots, grid updates on minibatches, devices and dtypes, checkpoints, pruning
  that dropped constants, symbolic fits that missed the right function, regularisers and attribution.
  [CHANGELOG.md](CHANGELOG.md) lists them.
- **A current stack.** Python 3.12+, PyTorch 2.14+, NumPy 2.5+ and SymPy 1.14+. `KANNetwork` computes; separate tools
  train, prune, interpret and plot it. `MultiHead` puts several heads on a shared trunk, `CompositeLoss` combines the
  losses of several outputs, and the trainer takes callbacks. Checkpoints load with `weights_only`. Over 430 tests and
  48 executed notebooks come with the package.
- **Improvements from other KAN libraries.**

  | from | in xKAN |
  |---|---|
  | efficient-KAN, Deep-KAN, KANLib | the forward as one product of the basis values and the weighted coefficients (no per-edge tensor) |
  | efficient-KAN | a regulariser computed from the weights (`reg_metric="edge_weights"`): no recorded activations, 15x faster on the CPU |
  | FastKAN | Gaussian RBF bases (`basis="rbf"`) |
  | FasterKAN | reflectional switch (sech^2) bases (`basis="rswaf"`) |
  | ChebyKAN, Deep-KAN | Chebyshev bases (`basis="chebyshev"`) |
  | FastKAN, FasterKAN | inputs normalised onto a fixed grid: `nn.LayerNorm` composed with `KANNetwork`s (below) |
  | KANLib | grid updates of every layer in one forward |
  | DropKAN, mean aggregation (papers) | `dropkan=`, `aggregation="mean"` |

On the CPU, the RBF, sech^2 and Chebyshev bases train about 3x faster than B-splines; on a GPU the four bases stay
within 30% of each other. A grid update of a 2000 x 128 x 2000 network takes 2.5 GB and 0.47 s on an H100, where
pykan needs 70.6 GB and 2.8 s. The changelog has the other measurements.

Code written for pykan needs the changes in the [migration guide](docs/migration.md); xKAN has no compatibility layer.

## Installation

xKAN needs Python 3.12 or later and PyTorch 2.14 or later. It is not on PyPI yet; install it from GitHub:

```bash
pip install "xkan @ git+https://github.com/khazum/xkan"           # the core: torch, numpy, sympy
pip install "xkan[plot] @ git+https://github.com/khazum/xkan"     # adds matplotlib, for xkan.viz
pip install "xkan[contrib] @ git+https://github.com/khazum/xkan"  # adds the MLP baseline, Feynman datasets and hypothesis tests
```

From a clone: `pip install -e ".[dev]"`, then `pytest tests`.

## Quick start

```python
import torch
from xkan import KANNetwork, create_dataset, fit
from xkan.interp import prune, refine
from xkan.symbolic import auto_symbolic, symbolic_formula

f = lambda x: torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)
train, test = create_dataset(f, n_var=2)              # dicts with "x" and "y"

model = KANNetwork([2, 5, 1], grid=3, k=3, seed=1)
fit(model, train, test, steps=50, lamb=1e-3)          # LBFGS with a sparsity penalty
model = prune(model, train["x"])                      # a smaller network (shape-changing tools return a new one)
fit(model, train, test, steps=50)
model = refine(model, 10, train["x"])                 # a finer grid
fit(model, train, test, steps=50)

auto_symbolic(model, train["x"], lib=["x", "x^2", "exp", "sin"])
fit(model, train, test, steps=50)
print(symbolic_formula(model)[0][0])                  # exp(x_2**2 + sin(pi*x_1)), up to rounding
```

`hellokan.ipynb` walks through these steps with plots. `tutorials/` holds 47 more notebooks on the API,
interpretability, examples and physics, all executed with xKAN.

## The package

| module | contents |
|---|---|
| `xkan.layers` | `KANLayer` (B-spline, RBF, sech^2 or Chebyshev bases; knot placements; DropKAN; mean aggregation; bias), `AdaptiveGrid`, `SymbolicLayer` |
| `xkan.nn` | `KANNetwork` (addition and multiplication nodes), `MultiHead`, `update_grids` |
| `xkan.losses` | `CompositeLoss`, `MSE`, `CrossEntropy`, `PoissonNLL`, `FunctionalTerm`, `KANRegularizer` (pykan's metrics and `edge_weights`) |
| `xkan.train` | `Trainer` (Adam, LBFGS, minibatches, gradient clipping, mixed precision), callbacks (grid updates, early stopping, best state, logging, figures), `fit` |
| `xkan.interp` | `record_activations`, `attribute`, `prune` / `prune_node` / `prune_edge` / `prune_input`, `refine`, `expand_width` / `expand_depth`, `swap` / `auto_swap`, `perturb` |
| `xkan.symbolic` | the function library, `fix_symbolic`, `suggest_symbolic`, `auto_symbolic`, `symbolic_formula`, `compile_formula` |
| `xkan.io` | `save_checkpoint`, `load_checkpoint` |
| `xkan.viz` | `plot`, `plot_edge` |
| `xkan.contrib` | an MLP with the same tools, Feynman datasets, hypothesis tests (separability, symmetry, trees) |

Several outputs and losses on a shared trunk:

```python
from xkan import KANNetwork, MultiHead, CompositeLoss, Trainer
from xkan.losses import CrossEntropy, MSE, KANRegularizer

model = MultiHead(KANNetwork([20, 8]), {"cls": KANNetwork([8, 3]), "recon": KANNetwork([8, 20])})
loss = CompositeLoss({"cls": (CrossEntropy(), 0.9), "recon": (MSE(), 0.1)}, regularizers=[KANRegularizer(1e-3)])
history = Trainer(model, loss, "adam", lr=1e-2).fit({"x": x, "cls": labels, "recon": x}, steps=200)
```

`KANNetwork.from_layers([layer_1, layer_2])` turns layers you built elsewhere (a `KANLayer` subclass, say) into a
network the tools accept, sharing their parameters.

FastKAN-style networks normalise the inputs onto a fixed grid and use a fast basis and the weight regulariser:

```python
from torch import nn
model = nn.Sequential(nn.LayerNorm(784), KANNetwork([784, 256], basis="rbf", grid_range=(-2, 2)),
                      nn.LayerNorm(256), KANNetwork([256, 10], basis="rbf", grid_range=(-2, 2)))
fit(model, train, test, opt="adam", lr=1e-3, lamb=1e-4, reg_metric="edge_weights", update_grid=False)
```

A layer norm mixes the inputs of a layer, so each edge depends on all of them. Use it for accuracy and speed, and leave
it out when you want to prune the network or read formulas from it.

## Advice on hyperparameters

Ziming Liu's advice from the pykan README, with xKAN's names:

- Start small: a small network, a small grid, a small dataset and no regularisation (`lamb=0`). MLP practice starts
  at widths around 100; for a task with 5 inputs and 1 output, try `KANNetwork([5, 1, 1], grid=3, k=3)` first. If it
  fails, widen the network before you deepen it.
- Once the accuracy is acceptable, refine the network for accuracy or for interpretability, one goal at a time.
- For accuracy, extend the grid with `refine` (as in `tutorials/Example/Example_1_function_fitting.ipynb`) and watch
  for overfitting.
- For interpretability, train with a sparsity penalty (`fit(..., lamb=0.01)`, raising `lamb` in steps), plot the
  network, prune it with `prune`, then train further or fit formulas.
- Compare the train and test losses. A large gap means overfitting: add data or shrink the model, where the grid size
  matters more than the width.

## Citation

If you use xKAN, cite the KAN papers:

```bibtex
@article{liu2024kan,
  title={KAN: Kolmogorov-Arnold Networks},
  author={Liu, Ziming and Wang, Yixuan and Vaidya, Sachin and Ruehle, Fabian and Halverson, James and Solja{\v{c}}i{\'c}, Marin and Hou, Thomas Y and Tegmark, Max},
  journal={arXiv preprint arXiv:2404.19756},
  year={2024}
}

@article{liu2024kan2,
  title={KAN 2.0: Kolmogorov-Arnold Networks Meet Science},
  author={Liu, Ziming and Ma, Pingchuan and Wang, Yixuan and Matusik, Wojciech and Tegmark, Max},
  journal={arXiv preprint arXiv:2408.10205},
  year={2024}
}
```

## License

MIT ([LICENSE](LICENSE)). Ziming Liu wrote pykan; Marcin Malec maintains xKAN.
