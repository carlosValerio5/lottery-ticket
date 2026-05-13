"""
Poda no estructurada, poda estructurada por ancho y flujo tipo lottery ticket.
"""

from pia.pruning.lottery_ticket import iterative_magnitude_pruning
from pia.pruning.masks import WeightMaskRegistry
from pia.pruning.prune import (
    make_imp_param_selector,
    prune_globally_by_magnitude,
    select_conv_weight_params,
    select_imp_weight_params,
)

from pia.pruning.resnet18_slim import (
    parameter_and_buffer_bytes,
    slim_resnet18_cifar_from_state_dict,
)


__all__ = [
    "WeightMaskRegistry",
    "iterative_magnitude_pruning",
    "make_imp_param_selector",
    "parameter_and_buffer_bytes",
    "prune_globally_by_magnitude",
    "select_conv_weight_params",
    "select_imp_weight_params",
    "slim_resnet18_cifar_from_state_dict",
]
