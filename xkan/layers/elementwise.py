"""The elementwise KAN layer: one edge per input, y_i = phi_i(x_i) (a learned univariate link per feature)."""
from .kan_layer import EdgeValues, KANLayer


class ElementwiseKANLayer(KANLayer):
    """
    dim inputs, dim outputs, output i = phi_i(x_i) = scale_base_i * b(x_i) + scale_sp_i * spline_i(x_i): the edges of
    KANLayer(dim, 1) without the sum over inputs, so its parameters, grid updates, refits and options (basis,
    out_of_grid, ...) are KANLayer's. A learned link per feature, for example on each output of a
    decoder. aggregation 'mean' and bias do not apply.

    layer(x) returns (batch, dim); layer(x, return_edges=True) returns EdgeValues whose y is (batch, dim) and whose
    per-edge values have one output, (batch, 1, dim).
    """

    def __init__(self, dim, num=5, k=3, *, device=None, dtype=None, **options):
        if options.get("aggregation", "sum") != "sum" or options.get("bias"):
            raise ValueError("an elementwise layer has no aggregation or bias")
        super().__init__(dim, 1, num, k, device=device, dtype=dtype, **options)

    def forward(self, x, return_edges=False):
        if return_edges or self._dropkan_active():
            edges = self._edge_values(x)
            y = edges.postacts[:, 0, :]
            return EdgeValues(y, edges.preacts, edges.postacts, edges.postspline) if return_edges else y
        spline = (self._spline_basis(x) * self.coef[:, 0, :][None]).sum(-1)  # (batch, dim)
        return spline * (self.scale_sp * self.mask)[:, 0] + self._base_values(x) * (self.scale_base * self.mask)[:, 0]
