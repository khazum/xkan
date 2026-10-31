import warnings

import torch

from .kan_layer import KANLayer


# Forward calls (in training mode) at which an AdaptiveKANLayer updates its grid: a geometric schedule over the early
# training steps
UPDATE_AT = (1, 20, 60, 150, 300, 600, 1200, 2400, 4800)


class AdaptiveGrid:
    '''
    Mixin of a KAN layer class that keeps the layer's grid on its own inputs while another model's training loop trains
    it (a xkan.train GridUpdate callback updates the grids of the model's layers; a layer trained by another loop keeps
    the grid it was created with).

    In training mode the layer collects its inputs and updates its grid (update_grid_from_samples, the curves
    preserved) at the forward calls in update_at, from the most recent min_rows collected rows. An update waits until
    min_rows rows have been collected, so the grid never follows one small batch. After the last scheduled update the
    grid stays fixed, so the late training and every evaluation see a stable layer.

    The class it is mixed into provides update_grid_from_samples(x, **grid_update_kwargs), grid and coef, and calls
    collect_and_update(x) at the start of its forward.
    '''

    def init_adaptive_grid(self, update_at=UPDATE_AT, min_rows=1024, grid_update_kwargs=None):
        '''
        Args:
        -----
            update_at : sequence of int
                forward calls (in training mode, counted from 1) that schedule a grid update
            min_rows : int
                rows an update uses: the most recent ones; an update waits until this many have been collected
            grid_update_kwargs : None or dict
                keyword arguments of update_grid_from_samples
        '''
        self.update_at = tuple(sorted(int(c) for c in update_at))
        if int(min_rows) < 1:
            raise ValueError(f"min_rows must be at least 1, got {min_rows}")
        self.min_rows = int(min_rows)
        self.grid_update_kwargs = dict(grid_update_kwargs or {})
        self.n_calls = 0
        self.n_updates = 0
        self.failed_updates = 0
        self._update_pending = False
        self._buffer = []
        self._buffered_rows = 0

    def collect_and_update(self, x):
        '''
        in training mode, collect the rows of x and update the grid when an update is due and min_rows rows have been
        collected

        Args:
        -----
            x : 2D torch.tensor
                the layer's input, shape (batch, in_dim)
        '''
        if not self.training or not self.update_at:
            return
        if self.n_calls >= self.update_at[-1] and not self._update_pending:
            self._buffer, self._buffered_rows = [], 0
            return
        self.n_calls += 1
        self._buffer.append(x.detach())
        self._buffered_rows += x.shape[0]
        while len(self._buffer) > 1 and self._buffered_rows - self._buffer[0].shape[0] >= self.min_rows:  # the most recent min_rows rows
            self._buffered_rows -= self._buffer.pop(0).shape[0]
        if self.n_calls in self.update_at:
            self._update_pending = True
        if self._update_pending and self._buffered_rows >= self.min_rows:
            self._update_pending = False
            grid, coef = self.grid.data.clone(), self.coef.data.clone()
            try:
                with torch.no_grad():
                    self.update_grid_from_samples(torch.cat(self._buffer), **self.grid_update_kwargs)
                self.n_updates += 1
            except RuntimeError as e:
                # keep the layer as it was (the grid may already have moved when the refit failed)
                self.grid.data, self.coef.data = grid, coef
                self.failed_updates += 1
                warnings.warn(f'grid update at forward call {self.n_calls} failed, grid kept: {e}')


class AdaptiveKANLayer(AdaptiveGrid, KANLayer):
    '''
    A KANLayer that updates its own grid during training (see AdaptiveGrid), for using a KAN layer inside another
    model's training loop. Same arguments and forward as KANLayer, plus:

    Args:
    -----
        update_at : sequence of int
            forward calls (in training mode) that schedule a grid update. Default: UPDATE_AT.
        min_rows : int
            rows an update uses; an update waits until this many have been collected. Default: 1024.
        grid_update_mode : str
            mode of the updates (see KANLayer.update_grid_from_samples). Default: 'sample'.

    Example
    -------
    >>> from xkan.layers import AdaptiveKANLayer
    >>> layer = AdaptiveKANLayer(in_dim=16, out_dim=100, num=3, k=2, min_rows=256)
    >>> y = layer(torch.randn(64, 16))  # in training mode: collected, grid updated once 256 rows are in
    '''

    def __init__(self, *args, update_at=UPDATE_AT, min_rows=1024, grid_update_mode='sample', **kwargs):
        KANLayer.__init__(self, *args, **kwargs)
        self.init_adaptive_grid(update_at, min_rows, {} if grid_update_mode == 'sample' else {'mode': grid_update_mode})

    def forward(self, x):
        self.collect_and_update(x)
        return super().forward(x)
