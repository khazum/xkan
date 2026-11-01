"""Trainer callbacks: grid updates, early stopping, the best state, logging. A callback implements any of
on_fit_start(trainer), on_step_start(trainer, step), on_step_end(trainer, step, logs), on_fit_end(trainer), and
state_dict / load_state_dict when it has state to resume."""
import copy
import math

from ..nn.traversal import update_grids


class Callback:
    def on_fit_start(self, trainer):
        pass

    def on_step_start(self, trainer, step):
        pass

    def on_step_end(self, trainer, step, logs):
        pass

    def on_fit_end(self, trainer):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass


class GridUpdate(Callback):
    """
    Update the grids of every KAN layer (xkan.nn.update_grids) before steps start, start + every, ... while below
    stop: pykan's schedule (every = stop / num, at least 1; the steps of a fit counted from 0).

    rows : 'all' (default; every training row of an in-memory dataset or of a loader's map-style dataset), 'batch'
        (the step's batch) or an int (that many rows drawn without replacement from the trainer's generator)
    mode : the refit mode of KANLayer.update_grid_from_samples
    """

    def __init__(self, num=10, stop=50, *, every=None, start=0, rows="all", mode="sample"):
        if every is None:
            if num < 1:
                raise ValueError("num must be at least 1")
            every = max(1, int(stop / num))
        if every < 1:
            raise ValueError("every must be at least 1")
        if not (rows in ("all", "batch") or (isinstance(rows, int) and rows > 0)):
            raise ValueError(f"rows must be 'all', 'batch' or a positive int, got {rows!r}")
        self.every, self.stop, self.start, self.rows, self.mode = every, stop, start, rows, mode

    def on_step_start(self, trainer, step):
        if step % self.every or step >= self.stop or step < self.start:
            return
        if self.rows == "batch":
            x = trainer.batch["x"]
        else:
            x = trainer.all_inputs()
            if self.rows != "all" and self.rows < len(x):
                x = x[trainer.rng.choice(len(x), self.rows, replace=False)]
        with trainer.eval_mode():
            update_grids(trainer.model, x, mode=self.mode, **trainer.forward_kwargs)


class EarlyStopping(Callback):
    """stop when monitor (a logged value, default 'val/total') has not improved by min_delta for patience steps"""

    def __init__(self, monitor="val/total", patience=10, min_delta=0., mode="min"):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.monitor, self.patience, self.min_delta, self.mode = monitor, patience, min_delta, mode
        self.best, self.wait = None, 0

    def on_step_end(self, trainer, step, logs):
        value = float(logs[self.monitor])
        better = self.best is None or (value < self.best - self.min_delta if self.mode == "min"
                                       else value > self.best + self.min_delta)
        if better:
            self.best, self.wait = value, 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                trainer.stop = True

    def state_dict(self):
        return {"best": self.best, "wait": self.wait}

    def load_state_dict(self, state):
        self.best, self.wait = state["best"], state["wait"]


class BestState(Callback):
    """keep a copy of the model's state at the best monitor value; restore=True loads it at the end of the fit"""

    def __init__(self, monitor="val/total", mode="min", restore=True):
        self.monitor, self.mode, self.restore = monitor, mode, restore
        self.best, self.state, self.step = None, None, None

    def on_step_end(self, trainer, step, logs):
        value = float(logs[self.monitor])
        if math.isnan(value):
            return
        if self.best is None or (value < self.best if self.mode == "min" else value > self.best):
            self.best, self.step = value, step
            self.state = copy.deepcopy(trainer.model.state_dict())

    def on_fit_end(self, trainer):
        if self.restore and self.state is not None:
            trainer.model.load_state_dict(self.state)


class Logger(Callback):
    """print the logged values every `every` steps"""

    def __init__(self, every=1, keys=None):
        self.every, self.keys = every, keys

    def on_step_end(self, trainer, step, logs):
        if step % self.every == 0:
            keys = self.keys or sorted(logs)
            print(f"step {step}: " + " | ".join(f"{k} {float(logs[k]):.3e}" for k in keys if k in logs))


class SplineShare(Callback):
    """
    Record each KAN layer's spline share (xkan.interp.spline_share) on x every `every` steps (and after the fit) in
    self.records, a list of {"step": step, <layer name>: share, ...}: a share falling towards 0 is a spline collapse
    (a weight decay on the spline scales and coefficients does it).
    """

    def __init__(self, x, every=10, rows=256):
        if every < 1:
            raise ValueError("every must be at least 1")
        self.x, self.every, self.rows, self.records = x, every, rows, []

    def _record(self, trainer, step):
        from ..interp.diagnostics import spline_share
        self.records.append({"step": step, **spline_share(trainer.model, self.x, rows=self.rows, **trainer.forward_kwargs)})

    def on_step_end(self, trainer, step, logs):
        if step % self.every == 0:
            self._record(trainer, step)

    def on_fit_end(self, trainer):
        if not self.records or self.records[-1]["step"] != trainer.fit_step:
            self._record(trainer, trainer.fit_step)

    def state_dict(self):
        return {"records": list(self.records)}

    def load_state_dict(self, state):
        self.records = list(state.get("records", []))


class Figures(Callback):
    """save xkan.viz.plot(model, data, **plot_kwargs) of a KANNetwork as folder/<step>.jpg every `every` steps (the frames
    of a training video)"""

    def __init__(self, data, folder="./figures", every=1, dpi=200, **plot_kwargs):
        self.data, self.folder, self.every, self.dpi, self.plot_kwargs = data, folder, every, dpi, plot_kwargs

    def on_step_end(self, trainer, step, logs):
        if step % self.every:
            return
        import os

        import matplotlib.pyplot as plt

        from ..viz import plot
        os.makedirs(self.folder, exist_ok=True)
        with trainer.eval_mode():
            plot(trainer.model, self.data, title=f"Step {step}", **self.plot_kwargs)
        plt.savefig(os.path.join(self.folder, f"{step}.jpg"), bbox_inches="tight", dpi=self.dpi)
        plt.close("all")
