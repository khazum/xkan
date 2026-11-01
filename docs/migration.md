# Moving from pykan 0.2.x to xkan 1.0

xkan 1.0 (`import xkan`) replaces pykan's `kan` package. It splits pykan's `KAN` / `MultKAN` class into a network
that computes (`KANNetwork`) and tools that act on it: training (`xkan.train`), losses (`xkan.losses`),
interpretation and pruning (`xkan.interp`), symbolic regression (`xkan.symbolic`), checkpoints (`xkan.io`) and plots
(`xkan.viz`). xkan has no compatibility layer; the old class, its methods and its checkpoint format are gone.

Conventions:

- Data are dicts of tensors: the input under `"x"`, targets by name (`"y"` for a single output).
  `create_dataset(...)` returns `(train, test)` such dicts.
- Tools that need activations take the data as an argument (`prune(model, train["x"])`). The model keeps no
  activations and has no `cache_data`; `with record_activations(model) as rec:` records them on request.
- Tools that change a network's shape (pruning, refine, expand, swap) **return a new network** and leave the given one
  unchanged: `model = prune(model, x)`. Tools that change values work in place (`fix_symbolic`, `auto_symbolic`,
  `remove_edge`, `perturb`, `update_grids`).
- `KANNetwork` has no `auto_save`, history or `rewind`; keep copies (`copy.deepcopy`) or checkpoints instead.
- The forward saves no activations and skips the symbolic branch until a symbolic edge is active, so `save_act`,
  `symbolic_enabled` and `speed()` are gone.
- `KANNetwork(..., seed=s)` draws its initial parameters from a generator seeded with `s` and leaves torch's, numpy's
  and Python's global generators as they were (`KAN(seed=s)` reseeded them). Seed your own data draws
  (`torch.manual_seed`) where a notebook relied on that. The trainer draws its batches from its own generator.
- Grid updates during training use every training row by default (`fit(..., grid_update_rows="all")`; the old
  minibatch fit updated on the batch: `grid_update_rows="batch"`).
- `fix_symbolic` returns the fit's r2 (a 0-d tensor) and prints nothing (`verbose=True` prints the grid search).
- Symbolic fits search `a` of `c * f(a * x + b) + d` in `a_range` widened for edges whose inputs are smaller than 1
  (divided by their largest |x|; `a_range_relative='widen'`, default); `a_range_relative=False` is pykan's fixed range,
  `True` divides for every edge (also narrowing it for inputs larger than 1).
- `KANNetwork` divides the initial spline scales by the square root of each layer's number of inputs, as pykan 0.2.8
  does. `scale_sp_fan_in=False` starts them at full strength, as pykan 0.2.7 and earlier did; several tutorials
  assume that.
- Attribution and pruning weigh every output 1 (`output_weighting='sum'`). pykan weighs each by 1 / outputs, which
  with many outputs puts every edge under the pruning thresholds; `output_weighting='mean'` restores it. Networks
  with one output score the same either way.
- Pruning leaves every output a path from the inputs (`keep_paths=True`), so no output turns constant;
  `xkan.interp.disconnected_outputs(model)` lists outputs that have none.
- The trainer's weight decay leaves the spline coefficients and scales alone (`Trainer(spline_weight_decay=0.)`);
  decay on them shrinks the splines to zero. `spline_weight_decay=None` decays every parameter, as torch does.

| 0.2.x | 1.0 |
|---|---|
| `from kan import *` | `from xkan import KANNetwork, fit, create_dataset` (+ the tools you use, below) |
| `KAN(width=[2,5,1], grid=5, k=3, seed=0)` | `KANNetwork([2, 5, 1], grid=5, k=3, seed=0)` (widths with multiplication nodes as before: `[2, [3, 2], 1]`; `mult_arity`, `base_fun`, `noise_scale`, `sparse_init`, `sp_trainable`, `grid_range`, ... as keyword arguments; `device=`) |
| `model = KAN(...); model.to(device)` | `KANNetwork(..., device=device)` or `.to(device)` |
| `dataset = create_dataset(f, n_var=2)` | `train, test = create_dataset(f, n_var=2)` |
| `dataset['train_input']`, `dataset['train_label']` | `train['x']`, `train['y']` (`test['x']`, `test['y']`) |
| `model.fit(dataset, opt="LBFGS", steps=50, lamb=0.01)` | `history = fit(model, train, test, steps=50, lamb=0.01)` (`opt="lbfgs"` default; `opt="adam"`, `lr=`, `batch=`, `lamb_l1`, `lamb_entropy`, `lamb_coef`, `lamb_coefdiff`, `reg_metric`, `update_grid`, `grid_update_num`, `stop_grid_update_step`, `loss=` a callable `loss(pred, target)`) |
| `results['train_loss']`, `results['test_loss']` (RMSE) | `history['train/y']`, `history['val/y']` (the loss itself: the mean squared error; `np.sqrt` for RMSE), `history['train/reg/kan']` (lamb times the regularisation) |
| `fit(..., loss_fn=f)` | `fit(..., loss=f)` |
| `fit(..., update_grid=False)` | the same |
| `fit(..., metrics=(...), display_metrics=...)` | compute them afterwards, or a `xkan.train` callback |
| `fit(..., save_fig=True, img_folder=...)` | `fit(..., callbacks=[xkan.train.Figures(train['x'], folder=...)])` (the plot after each step as `<step>.jpg`) |
| `fit(..., singularity_avoiding=True, y_th=10)` | `fit(..., forward_kwargs=dict(singularity_avoiding=True, y_th=10))` |
| own training loop over `model.parameters()` | `Trainer(model, CompositeLoss({...}), "adam").fit(train, test, steps=...)`, or a plain PyTorch loop (`model(x)` is a tensor) |
| `model(x)`; `model.get_act(x)` | `model(x)`; `with record_activations(model) as rec: model(x)` then `rec.record` (`.acts`, `.inputs[l]`, `.edge_outputs[l]`, `.edge_inputs(l)`, ...) |
| `model.acts`, `model.spline_preacts`, `model.spline_postacts` | `rec.record.acts`, `rec.record.edge_inputs(l)`, `rec.record.edge_outputs[l]` |
| `model.plot()` | `xkan.viz.plot(model, x)` (`in_vars`, `out_vars`, `beta`, `metric`, `scale`, `title`, `tick`, `sample`; it returns the figure: end the line with `;` in a notebook; the edge images are drawn in memory, `folder=` also saves them, which `plot()` did to `./figures` by default) |
| `model.get_fun(l, i, j)` | `xkan.viz.plot_edge(model, x, l, i, j)` |
| `model.get_range(l, i, j)` | `xkan.interp.edge_range(model, x, l, i, j)` |
| `model.update_grid_from_samples(x)`, `model.update_grid(x)` | `xkan.nn.update_grids(model, x)` (in place) |
| `model.prune()` | `model = xkan.interp.prune(model, x)` (`node_th`, `edge_th`, `fold_constants`) |
| `model.prune_node(th)`, `model.prune_edge(th)`, `model.prune_input(th)` | `prune_node(model, x, th)`, `prune_edge(model, x, th)`, `prune_input(model, x, th)` (each returns the new network; `prune_input(..., keep=[...])` for `active_inputs`) |
| `model.prune_node(mode="manual", active_neurons_id=[...])` | `prune_node(model, x, keep=[...])` |
| `model.remove_edge(l, i, j)`, `model.remove_node(l, i)` | `remove_edge(model, l, i, j)`, `remove_node(model, l, i)` (in place) |
| `model = model.refine(10)` | `model = xkan.interp.refine(model, 10, x)` |
| `model.expand_width(l, n)`, `model.expand_depth()` | `model = expand_width(model, l, n)`, `model = expand_depth(model)` (both keep every existing curve) |
| `model.swap(l, i, j)`, `model.auto_swap()` | `model = swap(model, l, i, j)`, `model = auto_swap(model, x)` |
| `model.module(0, '[0,1]->[1]')`, `model.perturb(...)` | `module(model, 0, '[0,1]->[1]')`, `perturb(model, ...)` (in place) |
| `model.attribute()`, `model.feature_score`, `model.node_scores`, `model.edge_scores` | `att = xkan.interp.attribute(model, x)`: `att.node[l]`, `att.edge[l]`, `att.subnode[l]`; `feature_score(model, x)` |
| `model.node_attribute()`, `model.feature_interaction(l)` | `xkan.interp.node_attribution(model, rec.record)`, `feature_interaction(model, x, l)` |
| `model.n_edge` | `xkan.interp.n_edges(model)` |
| `model.connection_cost` | `connection_cost(model, x)` |
| `model.fix_symbolic(l, i, j, 'sin')` | `xkan.symbolic.fix_symbolic(model, l, i, j, 'sin', x)` (in place; `fit=False` for `fit_params_bool=False`) |
| `model.unfix_symbolic(l, i, j)`, `model.set_mode(l, i, j, 's')` | `unfix_symbolic(model, l, i, j)`, `set_mode(model, l, i, j, 's')` |
| `model.suggest_symbolic(l, i, j)` | `suggest_symbolic(model, l, i, j, x)` |
| `model.auto_symbolic(lib=lib)` | `auto_symbolic(model, x, lib=lib)` (in place; returns the edges fixed) |
| `model.symbolic_formula()[0][0]` | `xkan.symbolic.symbolic_formula(model)[0][0]` (`var=`, `normalizer=`, `output_normalizer=`, `real=`) |
| `add_symbolic('J0', fun, c=1)` | `lib = xkan.symbolic.default_library().add('J0', fun, sympy_fun, 1)`, then `library=lib` in the symbolic tools |
| `kan.utils.SYMBOLIC_LIB` | `xkan.symbolic.default_library()` (or `xkan.symbolic.library.SYMBOLIC_LIB`, the table it is built from) |
| `from kan.compiler import kanpiler; kanpiler(vars, expr)` | `xkan.symbolic.compile_formula(vars, expr)` (network options such as `base_fun=`, `seed=` as keyword arguments) |
| `model.saveckpt('path')`, `KAN.loadckpt('path')` | `xkan.io.save_checkpoint(model, 'path')`, `xkan.io.load_checkpoint('path', map_location=device)` (the CPU by default) |
| `model.copy()` | `copy.deepcopy(model)` |
| `model.rewind('0.1')`, `model.history()` | keep copies or checkpoints of the states you want back |
| `model.tree(x)` | `xkan.contrib.hypothesis.plot_tree(model, x)` |
| `from kan.MLP import MLP`, `kan.feynman`, `kan.hypothesis` | `xkan.contrib.mlp`, `xkan.contrib.feynman`, `xkan.contrib.hypothesis` (`MLP.fit(train, test, ...)`) |
| `from kan.utils import batch_jacobian, batch_hessian, augment_input, ex_round, create_dataset_from_data` | the same (`create_dataset_from_data` returns `(train, test)`) |
| `get_derivative(model, x, y, derivative='hessian')` | the same (`loss_mode='reg'` / `'all'` with `regularizer=`) |
| `model.speed()`, `save_act`, `symbolic_enabled` | not needed |
| `model.width`, `model.width_in`, `model.width_out`, `model.depth` | the same |
| `model.act_fun[l]`, `model.symbolic_fun[l]` | `model.layers[l]` (a `KANLayer`), `model.symbolic[l]` (a `SymbolicLayer`; `.names`, `.affine`, `.mask`) |
| `model.node_bias[l]`, `model.subnode_scale[l]`, ... | the same (`nn.ParameterList`s) |
