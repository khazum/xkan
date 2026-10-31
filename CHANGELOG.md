# Changelog

## 1.0.0 (unreleased): xKAN, from pykan 0.2.8

xKAN 1.0 starts from pykan 0.2.8 (pykan commit ecde4ec, January 2025). Code written for pykan needs the changes in
[docs/migration.md](docs/migration.md).

### Breaking changes

- `KANNetwork` replaces `KAN` and `MultKAN`. It computes; tools in separate modules act on it: `xkan.train` (`fit`,
  `Trainer`), `xkan.losses`, `xkan.interp` (attribution, pruning, refine, expand, swap), `xkan.symbolic`, `xkan.io`
  and `xkan.viz`. `MLP`, the Feynman datasets and the hypothesis tests moved to `xkan.contrib`.
- Data are dicts of tensors (`{"x": ..., "y": ...}`); `create_dataset` and `create_dataset_from_data` return
  `(train, test)`.
- Tools that change a network's shape (pruning, refine, expand, swap) return a new network. Tools that change values
  (`fix_symbolic`, `auto_symbolic`, `perturb`, `update_grids`, `remove_edge`) work in place. `record_activations`
  records activations on request, and the network keeps none.
- `auto_save`, the history and `rewind` are gone. Checkpoints use format 2 (`config.json` and `state.pt`, loaded with
  `weights_only`); pykan's checkpoints and pickles do not load.
- `KANNetwork(seed=s)` draws from its own generator and leaves the global ones alone. The trainer draws its batches
  from its own generator too.
- `fit` returns the history of the losses themselves (MSE, where pykan logged RMSE). With LBFGS, a step's training
  loss is the loss at the parameters the step starts from; pykan logged the last line-search trial.
- `plot` returns the figure and draws the edge images in memory, and `folder=` also saves them. pykan wrote them to
  `./figures` and read them back, which broke when two notebooks plotted into one folder.
- Requirements: Python 3.12+, PyTorch 2.14+, NumPy 2.5+, SymPy 1.14+. matplotlib (`[plot]`), scikit-learn and tqdm
  (`[contrib]`) are optional; xKAN does not need pandas or pyyaml.

### Changed defaults (each measured before the change)

- Grid updates during training use every training row (pykan: the minibatch).
- Adam's default learning rate is 1e-2 (pykan: 1, LBFGS's convention). It gave the lower test RMSE in 19 of 25 runs.
- The forward multiplies the basis values by the weighted coefficients in one product, with no per-edge tensor.
- `y_th` defaults to 1000 in every forward, as in `fit` (pykan evaluated a model trained at 1000 at 10).
- `auto_symbolic(zero_gate=True)` keeps important edges from becoming `'0'`. pykan's selection let `'0'` beat any fit
  with r2 below 0.996 and left 19 of 40 benchmark runs with a near-constant model.
- Symbolic fits search `a` of `c f(a x + b) + d` in a range widened for edges whose inputs stay below 1
  (`a_range_relative='widen'`). The widened range beat the fixed one in 13 of 120 benchmark runs and lost in none. The
  fixed range missed `a` for small inputs (issue #529), and a range relative to every input also narrowed it for large
  inputs.
- Pruning folds the constant contributions of removed nodes and edges into biases and removes nodes left without
  edges. pykan dropped those contributions, which shifted the output.

### Added

- Bases: B-splines, Gaussian RBFs (FastKAN), reflectional switch functions (FasterKAN, `'rswaf'`) and Chebyshev
  polynomials. Knot placements (quantile, nonzero-quantile, zero-aware) and a minimum knot gap. DropKAN, mean
  aggregation and a per-output bias. `scale_sp_fan_in=False` gives pykan 0.2.7's initial spline scales.
- `MultiHead` puts several heads on a shared trunk. `CompositeLoss` takes terms per output (`MSE`, `CrossEntropy`,
  `PoissonNLL`, `FunctionalTerm`). `KANRegularizer` offers pykan's metrics and `'edge_weights'`, efficient-KAN's
  regulariser computed from the weights without recorded activations.
- `Trainer`: Adam, LBFGS or any optimiser, minibatches, gradient clipping, mixed precision, and callbacks for grid
  updates, early stopping, the best state, logging and figures.
- `KANNetwork.from_layers` makes a network of existing layers for the tools. `AdaptiveGrid` and adaptive layers update
  their grid while another model trains. `update_grids` updates every layer in one forward (KANLib).
- Grid updates and refine: refit modes (`'sample'`, `'sample+uniform'`, `'grid'`), `refit_guard`, and refits for any
  basis through the bases' transfer matrix. `xkan.spline.refit_counts()` reports how many least-squares systems the
  refits solved and how many they re-solved as rank-deficient.
- Symbolic regression: batched fits (`batched=True`), `selection='network'`, `input_standardize`, `r2_tau` and
  `widen_ranges`; `symbolic_formula(real=True)`; `compile_formula(..., **network_options)`.
- `get_derivative` goes through `torch.func.functional_call`; pykan's version returned zero derivatives for the affine
  parameters.

### Fixed

66 problems reported in pykan's issues, plus numerical ones found in testing:

- Spline fits: a pseudo-inverse fallback when least squares fails. Rank-deficient systems (tied or zero-inflated
  samples) get the minimum-norm SVD solution on every device, where CUDA's solver returned values of 1e20. Bases and
  gradients stay free of NaN at coincident knots, and refits leave contiguous coefficients.
- Grid updates and rebuilt models (#89, #235, #269, #346, #425, #482): refine and pruning keep the options; grid
  updates run without autograd; non-finite samples keep their grid; pruning keeps at least one node; `prune_input`
  composes.
- Devices, dtypes and copies (#12, #36, #75, #84, #98, #115, #186, #337, #354, #426, #443, #483).
- Checkpoints and copies (#343, #362, #395, #452, #475, #492).
- Attribution, regularisers and plots (#247, #383, #414, #501): arities of multiplication nodes, a gradient for
  `edge_forward_sum`, NaN scores.
- Training and data checks (#307, #406, #439, #504, #534, #551), the training data left in the cache (#391, #508),
  private draws and the whole test set (#308, #351, #381, #424, #486), custom losses reported as computed (#32, #445).
- The symbolic branch (#92, #130, #137, #174, #185, #287, #404, #432, #433, #451, #532, #533, #550, #553, #564):
  singularity-avoiding forms with finite gradients, masked NaNs, multiplication nodes of any arity in formulas.
- Pruning (#149, #416), Adam's learning rate (#342), real formulas (#289), the a-range (#529).
- `plot_tree` ignored its variable names. `augment_input` works on data dicts. Formulas and plots name the input
  columns a network keeps after `prune_input`. `ex_round` printed values of one unit in the last digit as `0.e-2`,
  because SymPy's rounding kept 3 bits of them.

### Performance

- A grid update of a 2000 x 128 x 2000 network takes 2.5 GB and 0.47 s on an H100 (pykan: 70.6 GB and 2.8 s).
  Minibatch training uses about 80% less memory. On each GPU workload we measured, xKAN ran as fast as pykan or faster.
- Symbolic fits run about 6x faster on the CPU. The grid search works in cache-sized blocks and picks the same
  functions on 600 benchmark edges.
- On the CPU, the RBF, RSWAF and Chebyshev bases train about 3x faster than B-splines, and the `'edge_weights'`
  regulariser runs about 15x faster than the recorded ones.

### Removed

`MultKAN` / `KAN`, `Symbolic_KANLayer`, the old compiler names (`kanpiler`, `expr2kan`: use `compile_formula`),
`experiment.py`, `auto_save` / `rewind` / history, `speed()`, `save_act`, `symbolic_enabled`, `cache_data`, the
checkpoint format 1, `sampling='legacy'`; the unused `extend` and `device` arguments of `xkan.spline.B_batch` and
`device` of `coef2curve`.

### Tutorials

The 47 tutorials and `hellokan` run on xKAN, with their outputs saved. Where pykan's saved outputs beat a fresh run,
we traced the pykan version that produced them and used its settings (Examples 3, 9, 10 and 11). Example 6 now shows
the two lines its text describes. Physics 1, 4B and 4C carry notes where pykan's outputs came from sessions run out of
order or from unreleased code.
