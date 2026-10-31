"""fit: the usual training of a KAN regression model in one call (a Trainer with a mean squared error, pykan's
regulariser and grid updates)."""
from ..losses import CompositeLoss, KANRegularizer, MSE
from ..nn import KANNetwork
from .callbacks import GridUpdate
from .trainer import Trainer


def fit(model, train, val=None, *, steps=100, opt="lbfgs", lr=None, loss=None, lamb=0., lamb_l1=1., lamb_entropy=2.,
        lamb_coef=0., lamb_coefdiff=0., reg_metric="edge_forward_spline_n", update_grid=True, grid_update_num=10,
        start_grid_update_step=0, stop_grid_update_step=50, grid_update_rows="all", batch=None, seed=None,
        callbacks=(), **trainer_options):
    """
    Train model on train (a dict with "x" and "y") for steps steps and return the history (Trainer.fit's): loss
    (default the mean squared error of "y"), plus lamb * KANRegularizer(l1=lamb_l1, entropy=lamb_entropy,
    coef=lamb_coef, coefdiff=lamb_coefdiff, metric=reg_metric) when lamb > 0, the grids updated grid_update_num
    times until stop_grid_update_step (update_grid), batch rows per step (None: all), optimizer opt ('lbfgs' by
    default; lr default 1 for LBFGS, 1e-2 otherwise), the trainer's generator seeded with seed (default the network's
    seed). trainer_options go to Trainer (scheduler, grad_clip, forward_kwargs, ...).
    """
    regularizers = []
    if lamb > 0.:
        regularizers.append(KANRegularizer(lamb, metric=reg_metric, l1=lamb_l1, entropy=lamb_entropy, coef=lamb_coef,
                                           coefdiff=lamb_coefdiff))
    objective = loss if isinstance(loss, CompositeLoss) else CompositeLoss({"y": loss or MSE()}, regularizers)
    if isinstance(loss, CompositeLoss) and regularizers:
        raise ValueError("give the regulariser in the CompositeLoss, or lamb with a loss function")
    cbs = list(callbacks)
    if update_grid and grid_update_num > 0:
        cbs.insert(0, GridUpdate(grid_update_num, stop_grid_update_step, start=start_grid_update_step,
                                 rows=grid_update_rows))
    if seed is None:
        seed = model.config.seed if isinstance(model, KANNetwork) else 0
    trainer = Trainer(model, objective, opt, lr, callbacks=cbs, batch_size=batch, seed=seed, **trainer_options)
    return trainer.fit(train, val, steps=steps)
