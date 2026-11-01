"""Ensembles: several fitted members, the best kept by validation loss, their predictions averaged."""
import torch


class Ensemble(torch.nn.Module):
    """
    The mean of its members' outputs. members: modules with the same output shape (KANNetworks, MultiHeads, any
    model); a dict output is averaged per key. Built directly or by fit_ensemble.
    """

    def __init__(self, members):
        super().__init__()
        if not members:
            raise ValueError("an ensemble needs at least one member")
        self.members = torch.nn.ModuleList(members)

    def forward(self, x, **kwargs):
        outs = [m(x, **kwargs) for m in self.members]
        if isinstance(outs[0], dict):
            return {key: torch.stack([o[key] for o in outs]).mean(0) for key in outs[0]}
        return torch.stack(outs).mean(0)

    @torch.no_grad()
    def agreement(self, x, **kwargs):
        """how alike the members' predictions are: per output column, 1 - mean squared difference over the sum of the
        two members' variances over the rows of x, the median over columns, the median over member pairs (1: the
        same function; None for one member). For a dict output, per key."""
        outs = [m(x, **kwargs) for m in self.members]
        if len(outs) < 2:
            return None
        if isinstance(outs[0], dict):
            return {key: _agreement([o[key] for o in outs]) for key in outs[0]}
        return _agreement(outs)


def _agreement(preds):
    vals = []
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            a, b = preds[i].reshape(len(preds[i]), -1), preds[j].reshape(len(preds[j]), -1)
            den = a.var(0) + b.var(0)
            ok = den > 1e-12
            vals.append(float((1 - ((a - b) ** 2).mean(0)[ok] / den[ok]).median()) if ok.any() else float("nan"))
    return float(torch.tensor(vals).median())


def fit_ensemble(build, fit, n=5, keep=None, score=None):
    """
    Fit n members and return the Ensemble of the best keep (default all) and the scores of all.

    build(i) -> a new member (seed it from i), fit(member, i) trains it (and may return anything); score(member) -> a
    float, lower is better (default: none; the first keep members are kept). The scores order the members for keep.
    """
    if keep is not None and not 1 <= keep <= n:
        raise ValueError(f"keep must be between 1 and n ({n}), got {keep}")
    members, scores = [], []
    for i in range(n):
        member = build(i)
        fit(member, i)
        members.append(member)
        scores.append(None if score is None else float(score(member)))
    order = list(range(n)) if score is None else sorted(range(n), key=lambda i: scores[i])
    chosen = sorted(order[:keep if keep is not None else n])
    return Ensemble([members[i] for i in chosen]), {"scores": scores, "kept": chosen}
