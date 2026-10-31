"""The training loop of any module (KAN networks, their compositions, other modules) on dict batches."""
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext

import numpy as np
import torch

from ..interp.recording import record_activations
from ..layers import KANLayer
from .lbfgs import LBFGS

DEFAULT_LR = {"lbfgs": 1., "torch_lbfgs": 1., "adam": 1e-2, "adamw": 1e-2, "sgd": 1e-2}
LBFGS_OPTIONS = dict(history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32)


class Trainer:
    """
    Trainer(model, loss, optimizer='adam', lr=None, *, callbacks=(), batch_size=None, seed=0, ...).fit(train, val,
    steps=...) trains model by loss (a xkan.losses.CompositeLoss or any callable loss(outputs, batch, model, rec)).

    Data are dicts of tensors with the input under "x" and the targets by name (all rows), or iterables of such
    batches (DataLoaders). For in-memory data a step is one batch: all rows (batch_size None) or batch_size rows
    drawn without replacement from the trainer's generator (np.random.default_rng(seed)); fit(steps=n). For loaders
    fit(epochs=n) or fit(steps=n). Validation data (dicts in chunks of eval_rows rows, or loaders) are evaluated
    after every step (eval_every=1) without autograd.

    optimizer : 'adam' (default), 'adamw', 'sgd', 'lbfgs' (pykan's LBFGS, strong Wolfe line search), 'torch_lbfgs'
        (torch.optim.LBFGS), an optimizer class or an instance; lr default 1 for LBFGS, 1e-2 otherwise; only the
        parameters that require gradients are optimised
    callbacks : xkan.train callbacks (GridUpdate, EarlyStopping, BestState, Logger, ...)
    scheduler : a learning-rate scheduler of the optimizer or a function optimizer -> scheduler, stepped per step
        (scheduler_interval='step') or per epoch (default; a step of in-memory data counts as an epoch)
    grad_clip : clip the gradient norm to this value
    amp : mixed precision (torch.autocast; float16 on CUDA with gradient scaling, else bfloat16); not with LBFGS
    forward_kwargs : keyword arguments of every forward (singularity_avoiding=True, say)

    The model trains in training mode and is evaluated in eval mode; DropKAN layers use the same edge masks for every
    evaluation of a step (LBFGS evaluates several times), seeded from the trainer's generator. When the loss reads
    activations (requires_recording) the training forward is recorded and the recorder passed to the loss.
    History: fit returns {name: list}, the logged values per step: 'train/<part>' at the parameters the step starts
    from (for LBFGS too, not a line-search trial), 'val/<part>' after the step, 'lr'.
    """

    def __init__(self, model, loss, optimizer="adam", lr=None, *, callbacks=(), batch_size=None, seed=0,
                 scheduler=None, scheduler_interval="epoch", grad_clip=None, amp=False, optimizer_options=None,
                 forward_kwargs=None, eval_every=1, eval_rows=None):
        self.model, self.loss = model, loss
        self.callbacks = list(callbacks)
        self.batch_size = batch_size
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be None (all rows) or positive")
        self.rng = np.random.default_rng(seed)
        self.forward_kwargs = dict(forward_kwargs or {})
        self.grad_clip, self.eval_every, self.eval_rows = grad_clip, eval_every, eval_rows
        self.optimizer = self._optimizer(optimizer, lr, dict(optimizer_options or {}))
        self.is_lbfgs = isinstance(self.optimizer, (LBFGS, torch.optim.LBFGS))
        if scheduler is not None and not hasattr(scheduler, "step"):
            scheduler = scheduler(self.optimizer)
        if scheduler_interval not in ("epoch", "step"):
            raise ValueError("scheduler_interval must be 'epoch' or 'step'")
        self.scheduler, self.scheduler_interval = scheduler, scheduler_interval
        device = self._device()
        if amp and self.is_lbfgs:
            raise ValueError("mixed precision with LBFGS is not supported")
        self.amp_dtype = (torch.float16 if device.type == "cuda" else torch.bfloat16) if amp else None
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.amp_dtype == torch.float16)
        self.dropkan = [m for m in model.modules() if isinstance(m, KANLayer) and m.dropkan > 0.]
        self.step, self.fit_step, self.epoch, self.stop = 0, 0, 0, False
        self.batch, self.train_data = None, None
        self.history = {}

    # ------------------------------------------------------------------------------------------------- setup

    def _device(self):
        p = next(self.model.parameters(), None)
        return p.device if p is not None else torch.device("cpu")

    def _optimizer(self, optimizer, lr, options):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if isinstance(optimizer, torch.optim.Optimizer):
            if lr is not None or options:
                raise ValueError("lr and optimizer_options go with an optimizer name or class, not an instance")
            return optimizer
        if isinstance(optimizer, str):
            name = optimizer.lower()
            if name not in DEFAULT_LR:
                raise ValueError(f"unknown optimizer {optimizer!r}; one of {sorted(DEFAULT_LR)}, a class or an instance")
            lr = DEFAULT_LR[name] if lr is None else lr
            if name in ("lbfgs", "torch_lbfgs"):
                options = {**LBFGS_OPTIONS, **({"tolerance_ys": 1e-32} if name == "lbfgs" else {}), **options}
                return (LBFGS if name == "lbfgs" else torch.optim.LBFGS)(params, lr=lr, **options)
            return {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}[name](params, lr=lr, **options)
        return optimizer(params, **({} if lr is None else {"lr": lr}), **options)

    # ------------------------------------------------------------------------------------------------- modes

    @contextmanager
    def eval_mode(self):
        """the model in eval mode inside (its modes restored after)"""
        modes = [(m, m.training) for m in self.model.modules()]
        self.model.eval()
        try:
            yield
        finally:
            for m, training in modes:
                m.training = training

    @contextmanager
    def _dropkan_masks(self, seed):
        if not self.dropkan:
            yield
            return
        generators = {}
        for layer in self.dropkan:
            dev = layer.coef.device
            if dev not in generators:
                generators[dev] = torch.Generator(device=dev)
            layer._dropkan_generator = generators[dev]
        self._reseed = lambda: [g.manual_seed(seed) for g in generators.values()]  # noqa: E731
        try:
            yield
        finally:
            for layer in self.dropkan:
                layer._dropkan_generator = None
            self._reseed = None

    # ------------------------------------------------------------------------------------------------- data

    @staticmethod
    def _check(data, name):
        if not isinstance(data, Mapping) or "x" not in data:
            raise ValueError(f"{name} data are dicts with the input under 'x' (and the targets by name)")
        n = len(data["x"])
        for key, v in data.items():
            if not isinstance(v, torch.Tensor) or v.ndim == 0 or len(v) != n:
                raise ValueError(f"{name} data: {key!r} needs {n} rows like 'x'")
        return n

    def all_inputs(self):
        """the inputs of every training row (in-memory data, or a loader's map-style dataset)"""
        if isinstance(self.train_data, Mapping):
            return self.train_data["x"]
        dataset = getattr(self.train_data, "dataset", None)
        if dataset is None or not hasattr(dataset, "__len__"):
            raise ValueError("all training rows need in-memory data or a loader over a map-style dataset; use rows='batch'")
        collate = getattr(self.train_data, "collate_fn", torch.utils.data.default_collate)
        return collate([dataset[i] for i in range(len(dataset))])["x"].to(self._device())

    def _batches(self, train, steps, epochs):
        if isinstance(train, Mapping):
            n = self._check(train, "training")
            full = self.batch_size is None or self.batch_size >= n
            for _ in range(steps if steps is not None else (epochs or 0)):
                yield train if full else {k: v[idx] for idx in [self.rng.choice(n, self.batch_size, replace=False)]
                                          for k, v in train.items()}, True
            return
        done = 0
        for _ in range(epochs if epochs is not None else 10**18):
            seen = False
            for batch in train:
                if steps is not None and done >= steps:
                    return
                seen = True
                done += 1
                yield batch, False
            if not seen:
                raise ValueError("the training loader is empty")
            yield None, True  # the end of an epoch
            if steps is not None and done >= steps:
                return

    # ------------------------------------------------------------------------------------------------- steps

    def _objective(self, batch):
        with (record_activations(self.model) if getattr(self.loss, "requires_recording", False) else nullcontext()) as rec:
            with torch.autocast(self._device().type, dtype=self.amp_dtype) if self.amp_dtype else nullcontext():
                outputs = self.model(batch["x"], **self.forward_kwargs)
                value = self.loss(outputs, batch, self.model, rec)
        if not isinstance(value, torch.Tensor) or value.ndim != 0:
            raise ValueError("the loss must give a scalar tensor")
        return value

    def train_step(self, batch):
        """one optimiser step on batch (callbacks' on_step_start first); returns the loss components at the parameters
        the step starts from (detached tensors): its first evaluation (LBFGS evaluates more points, line-search trials
        among them, whose losses are not those of the parameters it ends with)"""
        # the step's random draws in a fixed order: the batch rows (drawn by fit), the DropKAN seed, the grid rows
        seed = int(self.rng.integers(2**62)) if self.dropkan else None
        self._call("on_step_start", self.fit_step)
        first = {}
        params = [p for g in self.optimizer.param_groups for p in g["params"]]

        def closure():
            if self.dropkan:
                self._reseed()
            self.optimizer.zero_grad()
            value = self._objective(batch)
            self.scaler.scale(value).backward()
            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(params, self.grad_clip)
            if not first:
                first.update(getattr(self.loss, "components", None) or {"total": value.detach()})
            return value

        self.model.train()
        try:
            with self._dropkan_masks(seed):
                if self.is_lbfgs:
                    self.optimizer.step(closure)
                else:
                    closure()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
        finally:
            self.model.eval()
        return dict(first)

    def _eval_chunk(self, n):
        # rows per validation forward: all, unless one forward's values would exceed 2**26 elements per row budget (a
        # fused layer: its basis values and outputs; a layer with symbolic edges also their per-edge values)
        from ..nn.traversal import kan_networks
        sym = {id(net.layers[l]) for net in kan_networks(self.model) for l, s in enumerate(net.symbolic) if s.active}
        per_row = max([layer.in_dim * (layer.num + layer.k) + layer.out_dim + (layer.in_dim * layer.out_dim if id(layer) in sym else 0)
                       for layer in self.model.modules() if isinstance(layer, KANLayer)] or [1])
        return max(1, min(n, 2**26 // per_row))

    def _terms(self, outputs, batch):
        # the loss without regularisers
        if hasattr(self.loss, "terms_value"):
            self.loss.terms_value(outputs, batch, self.model)
            return dict(self.loss.components)
        return {"total": self.loss(outputs, batch, self.model, None).detach()}

    @torch.no_grad()
    def evaluate(self, data):
        """the loss terms (no regularisers) on data without autograd, in eval mode: in-memory data as one loss of all
        rows (the outputs computed in chunks of eval_rows rows), loaders as the mean over batches weighted by rows"""
        with self.eval_mode():
            if isinstance(data, Mapping):
                n = self._check(data, "validation")
                chunk = self.eval_rows or self._eval_chunk(n)
                outs = [self.model(data["x"][s:s + chunk], **self.forward_kwargs) for s in range(0, n, chunk)]
                if len(outs) == 1:
                    outputs = outs[0]
                elif isinstance(outs[0], Mapping):
                    outputs = {k: torch.cat([o[k] for o in outs]) for k in outs[0]}
                else:
                    outputs = torch.cat(outs)
                return self._terms(outputs, data)
            sums, rows = {}, 0
            for batch in data:
                parts = self._terms(self.model(batch["x"], **self.forward_kwargs), batch)
                m = len(batch["x"])
                for k, v in parts.items():
                    sums[k] = sums.get(k, 0.) + v * m
                rows += m
            if not rows:
                raise ValueError("the validation data are empty")
            return {k: v / rows for k, v in sums.items()}

    # ------------------------------------------------------------------------------------------------- fit

    def _call(self, event, *args):
        for cb in self.callbacks:
            getattr(cb, event)(self, *args)

    def _log(self, logs):
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def _scheduler_step(self, logs):
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(float(logs.get("val/total", logs["train/total"])))
        else:
            self.scheduler.step()

    def fit(self, train, val=None, *, steps=None, epochs=None):
        """train for steps (in-memory data: steps batches) or epochs (loaders); returns the history"""
        if steps is None and epochs is None:
            raise ValueError("give steps or epochs")
        if isinstance(val, Mapping):
            self._check(val, "validation")
        self.train_data, self.stop, self.history, self.fit_step = train, False, {}, 0
        logs = {}
        modes = [(m, m.training) for m in self.model.modules()]
        self._call("on_fit_start")
        try:
            for batch, epoch_end in self._batches(train, steps, epochs):
                if batch is not None:
                    self.batch = batch
                    parts = self.train_step(batch)
                    logs = {f"train/{k}": v for k, v in parts.items()}
                    if val is not None and self.fit_step % self.eval_every == 0:
                        logs.update({f"val/{k}": v for k, v in self.evaluate(val).items()})
                    logs["lr"] = self.optimizer.param_groups[0]["lr"]
                    self._log(logs)
                    if self.scheduler_interval == "step":
                        self._scheduler_step(logs)
                    self._call("on_step_end", self.fit_step, logs)
                    self.step += 1
                    self.fit_step += 1
                if epoch_end:
                    self.epoch += 1
                    if self.scheduler_interval == "epoch":
                        self._scheduler_step(logs)
                if self.stop:
                    break
            self._call("on_fit_end")
        finally:
            for m, training in modes:
                m.training = training
            self.batch = None
        self.history = {k: [float(v) for v in vs] for k, vs in self.history.items()}
        return self.history

    # ------------------------------------------------------------------------------------------------- state

    def state_dict(self):
        """model, optimiser (parameters by name), generator, scheduler, scaler, counters and callbacks"""
        names = {id(p): n for n, p in self.model.named_parameters()}
        return {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                "params": [[names.get(id(p)) for p in g["params"]] for g in self.optimizer.param_groups],
                "rng": self.rng.bit_generator.state, "step": self.step, "epoch": self.epoch,
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "scaler": self.scaler.state_dict(), "callbacks": [cb.state_dict() for cb in self.callbacks]}

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        params = dict(self.model.named_parameters())
        for group, names in zip(self.optimizer.param_groups, state["params"]):
            if None not in names:
                group["params"][:] = [params[n] for n in names]
        if self.is_lbfgs:
            self.optimizer._numel_cache = None
        self.optimizer.load_state_dict(state["optimizer"])
        self.rng.bit_generator.state = state["rng"]
        self.step, self.epoch = state["step"], state["epoch"]
        if self.scheduler is not None and state["scheduler"] is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        for cb, s in zip(self.callbacks, state["callbacks"]):
            cb.load_state_dict(s)
