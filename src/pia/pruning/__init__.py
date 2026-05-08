"""
Poda no estructurada y flujo tipo lottery ticket (IMP con reinicio).
"""

from pia.pruning.lottery_ticket import iterative_magnitude_pruning
from pia.pruning.masks import WeightMaskRegistry
from pia.pruning.prune import (
    make_imp_param_selector,
    prune_globally_by_magnitude,
    select_conv_weight_params,
    select_imp_weight_params,
)

__all__ = [
    "WeightMaskRegistry",
    "iterative_magnitude_pruning",
    "make_imp_param_selector",
    "prune_globally_by_magnitude",
    "select_conv_weight_params",
    "select_imp_weight_params",
]
