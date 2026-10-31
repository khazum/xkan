"""Constitutive laws of the Physics 4 tutorials: the stress P of a batch of deformation gradients F (batch, 3, 3).

The module the tutorials import was not in the repository; this is a reconstruction from the laws they state and
from the formulas they recover (mu = E / (2 (1 + nu)), lambda = E nu / ((1 + nu) (1 - 2 nu))):

    linear elasticity: P = mu (F + F^T - 2 I) + lambda tr(F - I) I
        P11 = 2 mu (F11 - 1) + lambda (F11 + F22 + F33 - 3), P12 = mu (F12 + F21)
    neo-Hookean:       P = mu (F F^T - I) + lambda log(det F) I
        P11 = mu (F11^2 + F12^2 + F13^2 - 1) + lambda log(det F), P12 = mu (F11 F21 + F12 F22 + F13 F23)
"""
import torch


class _ConstitutiveLaw:
    def __init__(self, young_modulus=1.0, poisson_ratio=0.2):
        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio

    def get_lame_parameters(self):
        """(mu, lambda)"""
        E, nu = self.young_modulus, self.poisson_ratio
        return E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))

    def __call__(self, F):
        return self.forward(F)


class LinearElasticConstitutiveLaw(_ConstitutiveLaw):
    def forward(self, F):
        mu, lambda_ = self.get_lame_parameters()
        eye = torch.eye(3, dtype=F.dtype, device=F.device).expand_as(F)
        trace = torch.diagonal(F, dim1=1, dim2=2).sum(dim=1)
        return mu * (F + F.transpose(1, 2) - 2 * eye) + lambda_ * torch.diag_embed((trace - 3)[:, None].expand(-1, 3))


class NeoHookeanConstitutiveLaw(_ConstitutiveLaw):
    def forward(self, F):
        mu, lambda_ = self.get_lame_parameters()
        eye = torch.eye(3, dtype=F.dtype, device=F.device).expand_as(F)
        log_det = torch.log(torch.det(F))
        # lambda log(det F) on the diagonal only (no inf * 0 off the diagonal for a singular F)
        return mu * (F @ F.transpose(1, 2) - eye) + lambda_ * torch.diag_embed(log_det[:, None].expand(-1, 3))
