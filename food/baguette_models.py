"""Baguette: the roll's crust-softening and staling models (food/roll_models.py) apply -- same Hirte 2012 crust water
physics and Lagrain 2012 crumb staling (bread, cited); a baguette stales faster than a boule (wp_baguette) because
of its shape (surface / volume), so its crust clock runs faster by the surface-to-volume ratio relative to the roll
(DERIVED from the twin's geometry); tearing by hand is the tradition (wp_baguette)."""

from .roll_models import RollTruth, roll_beliefs, roll_truth


def sv_ratio(b):
    """Surface / volume ratio of the instance (from the label grid would be exact; ellipse estimate here)."""
    return 2 / b.b + 1 / b.H


def baguette_beliefs(b, mp):
    B = roll_beliefs(b, mp)
    B[""]["sdt:belief:shape"] = "baguette"
    return B


def baguette_truth(b):
    return roll_truth(b)


class BaguetteTruth(RollTruth):
    def advance(self, dt_h, storage=None):
        super().advance(dt_h * 1.5, storage)            # faster than the roll (surface / volume ~1.5x, derived)
